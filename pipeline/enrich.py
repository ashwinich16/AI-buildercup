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

Install: see pipeline/common.py
"""

import json
import time
from typing import Optional

from google.genai import types
from pydantic import BaseModel, Field

from common import CloudSqlDatabase, Gemini, PipelineSettings, image_part

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
        WHERE pi.product_id IS NULL AND p.excluded_reason IS NULL   -- excluded products never reach the shop
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
   b. colour and pattern in everyday comparisons a listener can picture,
   c. material and texture, how it would feel to touch,
   d. shape and notable details,
   e. how it wears or is used, including what reviewers say about fit, comfort or quality,
   f. anything to watch out for, from reviews.
   Speak like a friendly, knowledgeable friend. Plain words, no lists, no markdown, no prices, no product codes. If unsure about something visual, say "it looks like" instead of stating it as fact.
"""


class GeminiEnricher:
    """Builds the prompt and asks Gemini for the product's intelligence, photos first."""

    def __init__(self, gemini: Gemini):
        self._gemini = gemini

    def enrich(self, product: dict, image_urls: list[str], reviews: list[dict]) -> Optional[ProductIntelligence]:
        parts = [part for part in (image_part(url) for url in image_urls) if part]
        if not parts:
            print("  ! no usable images, skipping")
            return None
        parts.append(types.Part.from_text(text=self._prompt(product, reviews)))
        return self._gemini.answer(parts, ProductIntelligence, temperature=0.2)

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

    MAX_IMAGES = 3
    MAX_REVIEWS = 15
    PAUSE_SECONDS = 1.0     # between products, to stay well inside the Gemini quota

    def __init__(self, repository: ProductRepository, enricher: GeminiEnricher, model: str):
        self._repository = repository
        self._enricher = enricher
        self._model = model

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
                    self._repository.image_urls(product_id, self.MAX_IMAGES),
                    self._repository.reviews(product_id, self.MAX_REVIEWS),
                )
                if result:
                    self._repository.save(product_id, result, self._model)
                    done += 1
                    print(f"  > {result.spoken_summary_short}")
            except Exception as error:  # keep going; the next run retries this product
                print(f"  ! failed: {error}")
                self._repository.rollback()
            time.sleep(self.PAUSE_SECONDS)

        print(f"Done. Enriched {done} of {len(products)}.")


def main() -> None:
    settings = PipelineSettings.from_env()
    gemini = Gemini(settings)
    with CloudSqlDatabase(settings) as connection:
        EnrichmentJob(ProductRepository(connection), GeminiEnricher(gemini), gemini.model).run()


if __name__ == "__main__":
    main()