"""
Scout - Catalog export to Cloud Storage for Vertex AI Search
=============================================================

Reads the catalog from Cloud SQL (product, product_image, review,
product_intelligence), turns each product into one JSON line in Vertex AI
Search's Document format ({"id": ..., "structData": {...}}), and uploads
them as a JSONL file to Cloud Storage. Import that file into a Vertex AI
Search data store of type "Structured data".

Nothing is written to local disk. All settings come from environment
variables (see .env.example), so it runs the same in Cloud Shell,
Cloud Run Jobs, or a laptop.

Environment variables:
  GOOGLE_CLOUD_PROJECT        e.g. scout-hackathon
  SCOUT_DB_INSTANCE           project:region:instance
  SCOUT_DB_NAME               e.g. scout_catalog
  SCOUT_DB_USER               e.g. scout_app
  SCOUT_DB_PASSWORD_SECRET    Secret Manager secret id holding the DB password
  SCOUT_EXPORT_BUCKET         e.g. scout-hackathon-catalog
  SCOUT_EXPORT_OBJECT         optional, default search/catalog.jsonl

Install:
  pip install "cloud-sql-python-connector[pg8000]" google-cloud-secret-manager google-cloud-storage
"""

import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from google.cloud import secretmanager, storage
from google.cloud.sql.connector import Connector

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"Missing environment variable: {name}")
    return value


@dataclass(frozen=True)
class ExportSettings:
    project_id: str
    db_instance: str
    db_name: str
    db_user: str
    db_password_secret: str
    bucket: str
    object_path: str
    review_excerpts_per_product: int = 3
    max_excerpt_chars: int = 220

    @classmethod
    def from_env(cls) -> "ExportSettings":
        return cls(
            project_id=_require_env("GOOGLE_CLOUD_PROJECT"),
            db_instance=_require_env("SCOUT_DB_INSTANCE"),
            db_name=_require_env("SCOUT_DB_NAME"),
            db_user=_require_env("SCOUT_DB_USER"),
            db_password_secret=_require_env("SCOUT_DB_PASSWORD_SECRET"),
            bucket=_require_env("SCOUT_EXPORT_BUCKET"),
            object_path=os.environ.get("SCOUT_EXPORT_OBJECT", "search/catalog.jsonl"),
        )


# ---------------------------------------------------------------------
# Database connection (only knows how to connect)
# ---------------------------------------------------------------------

class CloudSqlDatabase:
    """Opens a Cloud SQL connection with the password from Secret Manager."""

    def __init__(self, settings: ExportSettings):
        self._settings = settings
        self._connector = None
        self._connection = None

    def __enter__(self):
        self._connector = Connector()
        self._connection = self._connector.connect(
            self._settings.db_instance,
            "pg8000",
            user=self._settings.db_user,
            password=self._read_password(),
            db=self._settings.db_name,
        )
        return self._connection

    def __exit__(self, *exc):
        if self._connection:
            self._connection.close()
        if self._connector:
            self._connector.close()

    def _read_password(self) -> str:
        client = secretmanager.SecretManagerServiceClient()
        name = (f"projects/{self._settings.project_id}/secrets/"
                f"{self._settings.db_password_secret}/versions/latest")
        return client.access_secret_version(name=name).payload.data.decode("UTF-8").strip()


# ---------------------------------------------------------------------
# Catalog reading (only knows the SQL)
# ---------------------------------------------------------------------

class CatalogReader:
    """Reads products, images and top reviews. The only class that knows table names."""

    PRODUCTS_SQL = """
        SELECT p.product_id, p.title, p.store, p.brand, p.main_category, p.category,
               p.description, p.features, p.price, p.average_rating, p.rating_number,
               pi.visual_attributes AS attributes, pi.style_tags, pi.review_summary,
               pi.review_aspects AS review_points,
               pi.spoken_summary_short, pi.spoken_summary_detailed
        FROM product p
        LEFT JOIN product_intelligence pi ON pi.product_id = p.product_id
        ORDER BY p.product_id;
    """

    IMAGES_SQL = """
        SELECT product_id, array_agg(image_url ORDER BY ordinal)
        FROM product_image
        GROUP BY product_id;
    """

    REVIEWS_SQL = """
        SELECT product_id, text
        FROM (
            SELECT product_id, text,
                   row_number() OVER (
                       PARTITION BY product_id
                       ORDER BY verified_purchase DESC NULLS LAST, helpful_vote DESC
                   ) AS rank
            FROM review
            WHERE text IS NOT NULL AND length(trim(text)) > 0
        ) ranked
        WHERE rank <= %s
        ORDER BY product_id, rank;
    """

    def __init__(self, connection, reviews_per_product: int):
        self._cursor = connection.cursor()
        self._reviews_per_product = reviews_per_product

    def products(self) -> list[dict]:
        self._cursor.execute(self.PRODUCTS_SQL)
        columns = [d[0] for d in self._cursor.description]
        return [dict(zip(columns, row)) for row in self._cursor.fetchall()]

    def images_by_product(self) -> dict[str, list[str]]:
        self._cursor.execute(self.IMAGES_SQL)
        return {product_id: urls for product_id, urls in self._cursor.fetchall()}

    def reviews_by_product(self) -> dict[str, list[str]]:
        self._cursor.execute(self.REVIEWS_SQL, (self._reviews_per_product,))
        reviews: dict[str, list[str]] = {}
        for product_id, text in self._cursor.fetchall():
            reviews.setdefault(product_id, []).append(text)
        return reviews


# ---------------------------------------------------------------------
# Document building (no SQL, no cloud calls; works for any columns)
# ---------------------------------------------------------------------

class SearchDocumentBuilder:
    """Turns one product row into a clean, flat JSON object for Vertex AI Search.

    It does not depend on specific column names: every column is copied,
    JSON strings are parsed, numbers are made JSON-safe and empty values are
    dropped. Only SEARCH_TEXT_FIELDS decides what goes into `search_text`.
    """

    SEARCH_TEXT_FIELDS = (
        "title", "brand", "category", "description", "features",
        "attributes", "style_tags", "review_summary", "spoken_summary_detailed",
    )

    def __init__(self, max_excerpt_chars: int):
        self._max_excerpt_chars = max_excerpt_chars

    def build(self, row: dict, image_urls: list[str], reviews: list[str]) -> dict:
        document = {key: self._to_json_value(value) for key, value in row.items()}
        document["image_urls"] = image_urls
        document["review_excerpts"] = [self._excerpt(text) for text in reviews]
        document["search_text"] = self._search_text(document)
        return self._drop_empty(document)

    def _search_text(self, document: dict) -> str:
        words = [self._flatten_to_text(document.get(field)) for field in self.SEARCH_TEXT_FIELDS]
        return " ".join(" ".join(words).split())

    def _flatten_to_text(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return ""  # confidences, counts: noise for text search
        if isinstance(value, dict):
            return " ".join(self._flatten_to_text(v) for v in value.values())
        if isinstance(value, list):
            return " ".join(self._flatten_to_text(v) for v in value)
        return str(value)

    def _to_json_value(self, value):
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, str) and value[:1] in "{[":
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def _drop_empty(self, value):
        if isinstance(value, dict):
            cleaned = {k: self._drop_empty(v) for k, v in value.items()}
            return {k: v for k, v in cleaned.items() if v not in (None, "", [], {})}
        if isinstance(value, list):
            cleaned = [self._drop_empty(v) for v in value]
            return [v for v in cleaned if v not in (None, "", [], {})]
        return value

    def _excerpt(self, text: str) -> str:
        """First sentence or two of a review, short enough to say aloud."""
        text = " ".join((text or "").replace("<br />", " ").split())
        if len(text) <= self._max_excerpt_chars:
            return text
        cut = text[: self._max_excerpt_chars]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        return cut[: end + 1] if end > 40 else cut.rsplit(" ", 1)[0] + "..."


# ---------------------------------------------------------------------
# Upload (only knows Cloud Storage)
# ---------------------------------------------------------------------

class GcsUploader:
    def __init__(self, project_id: str):
        self._client = storage.Client(project=project_id)

    def upload_jsonl(self, documents: list[dict], bucket: str, object_path: str) -> str:
        body = "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents)
        self._client.bucket(bucket).blob(object_path).upload_from_string(
            body, content_type="application/json"
        )
        return f"gs://{bucket}/{object_path}"


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

class CatalogExporter:
    """Reads, builds and uploads. Each step is a class passed in, so any can be swapped."""

    def __init__(self, reader: CatalogReader, builder: SearchDocumentBuilder, uploader: GcsUploader):
        self._reader = reader
        self._builder = builder
        self._uploader = uploader

    def run(self, bucket: str, object_path: str) -> None:
        products = self._reader.products()
        images = self._reader.images_by_product()
        reviews = self._reader.reviews_by_product()
        print(f"Read {len(products)} products.")

        documents = [
            self._as_vertex_document(
                self._builder.build(p, images.get(p["product_id"], []), reviews.get(p["product_id"], []))
            )
            for p in products
        ]

        not_enriched = sum(1 for p in products if p.get("spoken_summary_short") is None)
        if not_enriched:
            print(f"Note: {not_enriched} products have no product_intelligence yet.")

        uri = self._uploader.upload_jsonl(documents, bucket, object_path)
        print(f"Uploaded {len(documents)} documents to {uri}")
        print("Next: import this file into a Vertex AI Search *structured* data store.")

    @staticmethod
    def _as_vertex_document(record: dict) -> dict:
        """Vertex AI Search Document format: one id plus the product fields under structData."""
        return {"id": record["product_id"], "structData": record}


def main() -> None:
    settings = ExportSettings.from_env()
    with CloudSqlDatabase(settings) as connection:
        exporter = CatalogExporter(
            reader=CatalogReader(connection, settings.review_excerpts_per_product),
            builder=SearchDocumentBuilder(settings.max_excerpt_chars),
            uploader=GcsUploader(settings.project_id),
        )
        exporter.run(settings.bucket, settings.object_path)


if __name__ == "__main__":
    main()
