"""
Scout - evidence for every review point: real quotes and real counts
====================================================================

enrich.py summarises reviews into points ("fit: fits true to size"). This
step backs each point with proof the shopper can trust:

  - quotes      1-2 short snippets copied WORD FOR WORD from the reviews.
                Every quote is checked against the review text in code; a
                quote that is not really there is dropped, never shown.
  - supporting  how many of the analysed reviews say it, counted in code from
                the review numbers Gemini cites (not Gemini's own estimate)

so Scout can say: "fits perfectly" (3 of 10 reviews).

Results are written into product_intelligence.review_aspects (each point gains
"quotes" and "supporting_reviews") plus product_intelligence.reviews_analyzed.

Run after enrich.py. Re-runnable: only products with no evidence yet are done.
To redo everything:  UPDATE product_intelligence SET reviews_analyzed = NULL;

Run:
  python pipeline/review_evidence.py
"""

import json
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

MAX_REVIEWS = 20          # ingestion keeps at most 20 per product
MAX_QUOTE_WORDS = 15      # short enough to read on a card and say out loud

ADD_COLUMN_SQL = "ALTER TABLE product_intelligence ADD COLUMN IF NOT EXISTS reviews_analyzed INTEGER;"

PENDING_SQL = """
    SELECT p.product_id, p.title, pi.review_aspects
    FROM product p JOIN product_intelligence pi USING (product_id)
    WHERE pi.reviews_analyzed IS NULL AND p.excluded_reason IS NULL
    ORDER BY p.product_id;
"""

REVIEWS_SQL = """
    SELECT rating, title, text FROM review
    WHERE product_id = %s AND text IS NOT NULL AND length(trim(text)) > 0
    ORDER BY verified_purchase DESC NULLS LAST, helpful_vote DESC
    LIMIT %s;
"""

SAVE_SQL = "UPDATE product_intelligence SET review_aspects = %s, reviews_analyzed = %s WHERE product_id = %s;"


class Quote(BaseModel):
    review: int = Field(description="Number of the review the quote is copied from")
    text: str = Field(description=f"Exact words from that review, at most {MAX_QUOTE_WORDS} words, unchanged")


class PointEvidence(BaseModel):
    point: int = Field(description="Number of the review point")
    supporting_reviews: list[int] = Field(description="Numbers of ALL reviews that say this")
    quotes: list[Quote] = Field(description="1 or 2 of the clearest short quotes that say this")


class EvidenceAnswer(BaseModel):
    points: list[PointEvidence]


PROMPT = """Below are customer reviews of one product, and points that summarise them.

For EACH point:
- supporting_reviews: the numbers of every review that clearly says this point (it may be none).
- quotes: one or two of the clearest short quotes that say it, each copied EXACTLY from one review:
  the same words in the same order, at most {max_words} words, no "..." and no changes.
  Pick the part of the sentence that carries the point. Leave quotes empty if no review says it.

PRODUCT: {title}

REVIEW POINTS
{points}

REVIEWS
{reviews}
"""


def normalise(text: str) -> str:
    """Lowercase, plain quotes, single spaces, no edge punctuation: for checking a quote is really there."""
    text = (text or "").lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.split()).strip(" .,!?;:\"'")


def verified_quotes(evidence: PointEvidence, reviews: list[dict]) -> list[str]:
    """Quotes that really appear in the review they cite, short and without duplicates."""
    kept = []
    for quote in evidence.quotes:
        if not 1 <= quote.review <= len(reviews):
            continue
        text = " ".join(quote.text.split()).strip(" \"'“”‘’")
        if not text or len(text.split()) > MAX_QUOTE_WORDS or "..." in text or "…" in text:
            continue
        if normalise(text) not in normalise(reviews[quote.review - 1]["text"]):
            continue                      # not word for word: never show it
        if normalise(text) not in {normalise(k) for k in kept}:
            kept.append(text)
    return kept[:2]


class GeminiEvidenceFinder:
    def __init__(self, project_id: str):
        self._client = genai.Client(vertexai=True, project=project_id,
                                    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
        self._model = os.environ.get("SCOUT_ENRICH_MODEL", "gemini-2.5-flash")

    def find(self, title: str, points: list[dict], reviews: list[dict]) -> EvidenceAnswer:
        prompt = PROMPT.format(
            max_words=MAX_QUOTE_WORDS,
            title=title,
            points="\n".join(f"{n}. {p.get('aspect')}: {p.get('finding')}" for n, p in enumerate(points, start=1)),
            reviews="\n".join(f"{n}. ({r['rating']} stars) {r['title'] or ''}: {' '.join((r['text'] or '').split())}"
                              for n, r in enumerate(reviews, start=1)),
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=EvidenceAnswer,
            ),
        )
        return EvidenceAnswer.model_validate_json(response.text)


def with_evidence(points: list[dict], answer: EvidenceAnswer, reviews: list[dict]) -> list[dict]:
    by_point = {e.point: e for e in answer.points}
    updated = []
    for number, point in enumerate(points, start=1):
        evidence = by_point.get(number)
        supporting = {n for n in (evidence.supporting_reviews if evidence else []) if 1 <= n <= len(reviews)}
        quotes = verified_quotes(evidence, reviews) if evidence else []
        # A quoted review supports the point by definition, even if Gemini forgot to list it.
        supporting |= {q.review for q in (evidence.quotes if evidence else [])
                       if 1 <= q.review <= len(reviews) and q.text.strip() and
                       normalise(q.text) in normalise(reviews[q.review - 1]["text"])}
        updated.append({**point, "quotes": quotes, "supporting_reviews": len(supporting)})
    return updated


def main():
    import ingestion   # reuses its Cloud SQL connection settings

    conn, connector = ingestion.get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(ADD_COLUMN_SQL)
        conn.commit()
        cursor.execute(PENDING_SQL)
        pending = cursor.fetchall()
        print(f"{len(pending)} products need review evidence.")

        finder = GeminiEvidenceFinder(ingestion.PROJECT_ID)
        dropped_total = kept_total = 0
        for number, (product_id, title, points) in enumerate(pending, start=1):
            points = points if isinstance(points, list) else json.loads(points or "[]")
            cursor.execute(REVIEWS_SQL, (product_id, MAX_REVIEWS))
            reviews = [{"rating": r[0], "title": r[1], "text": r[2]} for r in cursor.fetchall()]
            if not points or not reviews:
                cursor.execute(SAVE_SQL, (json.dumps(points), len(reviews), product_id))
                conn.commit()
                continue
            try:
                answer = finder.find(title, points, reviews)
            except Exception as error:   # one bad call must not lose the rest
                print(f"  [{number}/{len(pending)}] {product_id} FAILED: {error}")
                continue

            updated = with_evidence(points, answer, reviews)
            offered = sum(len(e.quotes) for e in answer.points)
            kept = sum(len(p["quotes"]) for p in updated)
            dropped_total += offered - kept
            kept_total += kept
            cursor.execute(SAVE_SQL, (json.dumps(updated), len(reviews), product_id))
            conn.commit()
            best = max(updated, key=lambda p: p["supporting_reviews"])
            sample = f'"{best["quotes"][0]}" ({best["supporting_reviews"]} of {len(reviews)})' if best["quotes"] else "-"
            print(f"  [{number}/{len(pending)}] {title[:45]:<45} {kept} quotes  e.g. {sample[:70]}")

        print(f"\nKept {kept_total} verified quotes; dropped {dropped_total} that were not word for word.")
    finally:
        cursor.close()
        conn.close()
        connector.close()


if __name__ == "__main__":
    main()
