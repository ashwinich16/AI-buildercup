"""
Scout - Gemini enrichment job
==============================

Run after ingestion.py. For each product it sends the images, the seller
text and real review text to Gemini in ONE multimodal call, and stores:

  - attributes          what can be seen or is stated, as a generic list
                        [{name, value, source, confidence}]  -> works for any
                        category (fashion, electronics, home, ...)
  - style_tags          Gemini's interpretation, kept apart from facts
  - review_summary      summary of the real reviews only
  - review_points       [{aspect, finding, sentiment, mentions}] from reviews
  - spoken_summary_short     2 sentences, for a shortlist
  - spoken_summary_detailed  5-7 sentences, so a blind shopper can picture it

Re-runnable: it only processes products that have no product_intelligence row
yet. To redo everything, delete the rows first: DELETE FROM product_intelligence;

Environment variables:
  GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION (Vertex AI region, e.g. us-central1)
  SCOUT_DB_INSTANCE, SCOUT_DB_NAME, SCOUT_DB_USER, SCOUT_DB_PASSWORD_SECRET
  SCOUT_ENRICH_MODEL  optional, default gemini-2.5-flash

Install:
  pip install "cloud-sql-python-connector[pg8000]" google-cloud-secret-manager google-genai pydantic requests
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import requests
from google import genai
from google.cloud import secretmanager
from google.cloud.sql.connector import Connector
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field
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
class EnrichSettings:
    project_id: str
    vertex_location: str
    model: str
    db_instance: str
    db_name: str
    db_user: str
    db_password_secret: str
    max_images: int = 3
    max_reviews: int = 15
    pause_seconds: float = 1.0

    @classmethod
    def from_env(cls) -> "EnrichSettings":
        return cls(
            project_id=_require_env("GOOGLE_CLOUD_PROJECT"),
            vertex_location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
            model=os.environ.get("SCOUT_ENRICH_MODEL", "gemini-2.5-flash"),
            db_instance=_require_env("SCOUT_DB_INSTANCE"),
            db_name=_require_env("SCOUT_DB_NAME"),
            db_user=_require_env("SCOUT_DB_USER"),
            db_password_secret=_require_env("SCOUT_DB_PASSWORD_SECRET"),
        )


# ---------------------------------------------------------------------
# Output schema (category-agnostic)
# ---------------------------------------------------------------------

class Attribute(BaseModel):
    name: str = Field(description="Attribute name in plain words, e.g. 'color', 'material', 'screen size'")
    value: str = Field(description="Value in everyday words")
    source: str = Field(description="'image' if seen in the photos, 'seller' if only stated in the text")
    confidence: float = Field(description="0.0 to 1.0; how sure you are")


class ReviewPoint(BaseModel):
    aspect: str = Field(description="What the point is about, e.g. 'sizing', 'comfort', 'battery life'")
    finding: str = Field(description="What reviewers say, in one short sentence")
    sentiment: str = Field(description="'positive', 'negative' or 'mixed'")
    mentions: int = Field(description="How many of the given reviews support this point")


class ProductIntelligence(BaseModel):
    attributes: list[Attribute]
    style_tags: list[str]
    review_summary: Optional[str] = None
    review_points: list[ReviewPoint] = []
    spoken_summary_short: str
    spoken_summary_detailed: str


# ---------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------

class CloudSqlDatabase:
    """Opens a Cloud SQL connection with the password from Secret Manager."""

    def __init__(self, settings: EnrichSettings):
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


class ProductRepository:
    """All SQL for this job lives here."""

    ADD_COLUMNS_SQL = """
        ALTER TABLE product_intelligence
            ADD COLUMN IF NOT EXISTS spoken_summary_short TEXT,
            ADD COLUMN IF NOT EXISTS spoken_summary_detailed TEXT;
    """

    PENDING_SQL = """
        SELECT p.product_id, p.title, p.category, p.brand, p.description, p.features, p.details
        FROM product p
        LEFT JOIN product_intelligence pi ON pi.product_id = p.product_id
        WHERE pi.product_id IS NULL
        ORDER BY p.product_id;
    """

    IMAGES_SQL = """
        SELECT image_url FROM product_image
        WHERE product_id = %s
        ORDER BY ordinal
        LIMIT %s;
    """

    REVIEWS_SQL = """
        SELECT rating, title, text FROM review
        WHERE product_id = %s AND text IS NOT NULL AND length(trim(text)) > 0
        ORDER BY verified_purchase DESC NULLS LAST, helpful_vote DESC
        LIMIT %s;
    """

    UPSERT_SQL = """
        INSERT INTO product_intelligence (
            product_id, visual_attributes, style_tags, review_summary, review_aspects,
            spoken_summary_short, spoken_summary_detailed, model_version
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (product_id) DO UPDATE SET
            visual_attributes = EXCLUDED.visual_attributes,
            style_tags = EXCLUDED.style_tags,
            review_summary = EXCLUDED.review_summary,
            review_aspects = EXCLUDED.review_aspects,
            spoken_summary_short = EXCLUDED.spoken_summary_short,
            spoken_summary_detailed = EXCLUDED.spoken_summary_detailed,
            model_version = EXCLUDED.model_version,
            generated_at = now();
    """

    def __init__(self, connection):
        self._connection = connection
        self._cursor = connection.cursor()

    def ensure_columns(self) -> None:
        self._cursor.execute(self.ADD_COLUMNS_SQL)
        self._connection.commit()

    def pending_products(self) -> list[dict]:
        self._cursor.execute(self.PENDING_SQL)
        columns = [d[0] for d in self._cursor.description]
        return [dict(zip(columns, row)) for row in self._cursor.fetchall()]

    def image_urls(self, product_id: str, limit: int) -> list[str]:
        self._cursor.execute(self.IMAGES_SQL, (product_id, limit))
        return [row[0] for row in self._cursor.fetchall()]

    def reviews(self, product_id: str, limit: int) -> list[dict]:
        self._cursor.execute(self.REVIEWS_SQL, (product_id, limit))
        return [{"rating": r[0], "title": r[1], "text": r[2]} for r in self._cursor.fetchall()]

    def save(self, product_id: str, result: ProductIntelligence, model: str) -> None:
        self._cursor.execute(self.UPSERT_SQL, (
            product_id,
            json.dumps([a.model_dump() for a in result.attributes]),
            result.style_tags,
            result.review_summary,
            json.dumps([p.model_dump() for p in result.review_points]),
            result.spoken_summary_short,
            result.spoken_summary_detailed,
            model,
        ))
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()


# ---------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------

class ImageFetcher:
    """Downloads images as Gemini parts; skips any that fail."""

    def fetch(self, urls: list[str]) -> list[types.Part]:
        parts = []
        for url in urls:
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                mime = response.headers.get("Content-Type", "image/jpeg").split(";")[0]
                parts.append(types.Part.from_bytes(data=response.content, mime_type=mime))
            except requests.RequestException as error:
                print(f"  ! image download failed ({url}): {error}")
        return parts


# ---------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------

PROMPT = """You describe one product from an online store for a voice shopping assistant used by blind and low-vision shoppers. A wrong description can mislead someone who cannot check the photos, so accuracy matters more than completeness.

PRODUCT
Title: {title}
Category: {category}
Brand: {brand}
Seller description: {description}
Seller features: {features}
Seller details: {details}

REAL CUSTOMER REVIEWS ({review_count})
{reviews}

RULES
1. attributes: list the attributes that matter for this kind of product. Use source "image" only for what you can clearly see in the photos; use source "seller" for what is only stated in the text. Never guess. If you cannot tell, leave the attribute out. Photos may show only one colour option: describe what is pictured.
2. style_tags: your own interpretation (e.g. "casual", "formal", "minimalist"). These are opinions, not facts.
3. review_summary and review_points: use ONLY the reviews above. If there are no reviews, set review_summary to null and review_points to an empty list. Count "mentions" honestly.
4. spoken_summary_short: at most 2 sentences, under 40 words. Say what it is and its single most distinctive feature.
5. spoken_summary_detailed: 5 to 7 sentences, under 130 words, in this order:
   a. overall impression (what it is and its feel),
   b. colour and pattern in everyday comparisons (e.g. "soft sky blue, like faded jeans"),
   c. material and texture, how it would feel to touch,
   d. shape and notable details,
   e. how it wears or is used, including what reviewers say about fit, comfort or quality,
   f. anything to watch out for, from reviews.
   Speak like a friendly, knowledgeable friend. Plain words, no lists, no markdown, no prices, no product codes. If unsure about something visual, say "it looks like" instead of stating it as fact.
"""


class GeminiEnricher:
    """Builds the prompt and calls Gemini with a structured response schema."""

    def __init__(self, settings: EnrichSettings, image_fetcher: ImageFetcher):
        self._client = genai.Client(vertexai=True, project=settings.project_id,
                                    location=settings.vertex_location)
        self._model = settings.model
        self._image_fetcher = image_fetcher

    def enrich(self, product: dict, image_urls: list[str], reviews: list[dict]) -> Optional[ProductIntelligence]:
        parts = self._image_fetcher.fetch(image_urls)
        if not parts:
            print("  ! no usable images, skipping")
            return None

        parts.append(types.Part.from_text(text=self._prompt(product, reviews)))
        response = self._generate_with_retry(parts)
        return ProductIntelligence.model_validate_json(response.text)

    RETRY_CODES = (429, 500, 503)          # quota / temporarily unavailable
    RETRY_WAITS_SECONDS = (10, 20, 40, 80)

    def _generate_with_retry(self, parts: list[types.Part]):
        """Call Gemini; on quota or overload errors wait and try again (10s, 20s, 40s, 80s)."""
        for attempt, wait in enumerate((*self.RETRY_WAITS_SECONDS, None), start=1):
            try:
                return self._client.models.generate_content(
                    model=self._model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ProductIntelligence,
                        temperature=0.2,
                    ),
                )
            except genai_errors.APIError as error:
                if error.code not in self.RETRY_CODES or wait is None:
                    raise
                print(f"  ~ Gemini returned {error.code}; retry {attempt} in {wait}s")
                time.sleep(wait)

    def _prompt(self, product: dict, reviews: list[dict]) -> str:
        review_lines = [
            f"- ({r['rating']} stars) {r['title'] or ''}: {' '.join((r['text'] or '').split())}"
            for r in reviews
        ]
        return PROMPT.format(
            title=product["title"],
            category=product.get("category") or "unknown",
            brand=product.get("brand") or "unknown",
            description=" ".join(product.get("description") or []) or "(none)",
            features=" | ".join(product.get("features") or []) or "(none)",
            details=self._as_text(product.get("details")),
            review_count=len(reviews),
            reviews="\n".join(review_lines) or "(none)",
        )

    @staticmethod
    def _as_text(value) -> str:
        if not value:
            return "(none)"
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


# ---------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------

class EnrichmentJob:
    """Loops over pending products. One failure never stops the batch."""

    def __init__(self, repository: ProductRepository, enricher: GeminiEnricher, settings: EnrichSettings):
        self._repository = repository
        self._enricher = enricher
        self._settings = settings

    def run(self) -> None:
        self._repository.ensure_columns()
        products = self._repository.pending_products()
        print(f"{len(products)} products need enrichment.")

        done = 0
        for i, product in enumerate(products, start=1):
            product_id = product["product_id"]
            print(f"[{i}/{len(products)}] {product_id} - {product['title'][:60]}")
            try:
                result = self._enricher.enrich(
                    product,
                    self._repository.image_urls(product_id, self._settings.max_images),
                    self._repository.reviews(product_id, self._settings.max_reviews),
                )
                if result:
                    self._repository.save(product_id, result, self._settings.model)
                    done += 1
                    print(f"  > {result.spoken_summary_short}")
            except Exception as error:  # keep going; the next run retries this product
                print(f"  ! failed: {error}")
                self._repository.rollback()
            time.sleep(self._settings.pause_seconds)

        print(f"Done. Enriched {done} of {len(products)}.")


def main() -> None:
    settings = EnrichSettings.from_env()
    with CloudSqlDatabase(settings) as connection:
        job = EnrichmentJob(
            repository=ProductRepository(connection),
            enricher=GeminiEnricher(settings, ImageFetcher()),
            settings=settings,
        )
        job.run()


if __name__ == "__main__":
    main()