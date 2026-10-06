"""
Scout - who each product is made for: men, women or unisex
===========================================================

Fills product.gender so Scout can keep men's and women's items apart. This is
a fact about the ITEM, not the shopper: anyone can shop for anyone.

Each product is labelled from the strongest evidence available:
  1. Amazon's own "Department" detail ("Womens", "Mens", "Unisex-adult")
  2. words in the title ("Women's", "Men's", "Unisex", or both)
  3. Gemini, for the rest: it reads the title, seller text and main photo

ingestion.py uses steps 1-2 for new products; run this script afterwards to
fill whatever they left empty. Re-runnable: it only touches rows with no
gender yet. To redo everything:  UPDATE product SET gender = NULL;

Run:
  python pipeline/gender.py
"""

import json
import re
from typing import Literal, Optional

from google.genai import types
from pydantic import BaseModel

from common import CloudSqlDatabase, Gemini, PipelineSettings, image_part

Gender = Literal["men", "women", "unisex"]

ADD_COLUMN_SQL = """
ALTER TABLE product ADD COLUMN IF NOT EXISTS gender TEXT
    CHECK (gender IN ('men', 'women', 'unisex'));
"""

# "women" is checked with its own pattern, and \bmen never matches inside "women".
# (?<!-) skips names like "Spider-Man".
WOMEN_PATTERN = re.compile(r"(?<!-)\b(women['’]?s?|woman['’]?s?|ladies|lady|female|girl['’]?s?)\b", re.I)
MEN_PATTERN = re.compile(r"(?<!-)\b(men['’]?s?|man['’]?s?|male|boy['’]?s?|gentlemen)\b", re.I)
UNISEX_PATTERN = re.compile(r"\bunisex\b", re.I)


def gender_from_text(text: str) -> Optional[Gender]:
    """men / women / unisex from free text, or None when it says nothing about it."""
    if not text:
        return None
    if UNISEX_PATTERN.search(text):
        return "unisex"
    women, men = WOMEN_PATTERN.search(text), MEN_PATTERN.search(text)
    if women and men:
        return "unisex"          # "for Men and Women"
    if women:
        return "women"
    if men:
        return "men"
    return None


def gender_from_listing(title: str, details) -> Optional[Gender]:
    """Steps 1-2: Amazon's Department detail first, then the title."""
    if isinstance(details, str):
        details = json.loads(details or "{}")
    department = (details or {}).get("Department") if isinstance(details, dict) else None
    # Department values look like "womens", "Mens", "Unisex-adult", "Baby-girls".
    from_department = gender_from_text(str(department or "").replace("-", " "))
    from_title = gender_from_text(title)
    # Sellers often leave Department as "Unisex-adult" on items titled "...for Women": the specific one wins.
    if from_department == "unisex" and from_title in ("men", "women"):
        return from_title
    return from_department or from_title


# ---------------------------------------------------------------------
# Step 3: Gemini, for products the listing doesn't settle
# ---------------------------------------------------------------------

class GenderAnswer(BaseModel):
    gender: Gender
    reason: str


PROMPT = """Who is this product designed and sold for?
Answer "men", "women" or "unisex" (made for anyone, or sold for both).
Judge the product itself: its cut, styling, sizing and how it is marketed. Ignore props, models' accessories
or other items shown around it in the photo. If nothing about the product points to one gender, answer "unisex".

Title: {title}
Category: {category}
Description: {description}
Features: {features}
"""


class GeminiGenderClassifier:
    def __init__(self, gemini: Gemini):
        self._gemini = gemini

    def classify(self, product: dict, image_url: Optional[str]) -> GenderAnswer:
        parts = [types.Part.from_text(text=PROMPT.format(
            title=product["title"],
            category=product["category"],
            description=" ".join(product["description"] or [])[:1500],
            features=" | ".join(product["features"] or [])[:1500],
        ))]
        image = image_part(image_url)
        if image:
            parts.append(image)
        return self._gemini.answer(parts, GenderAnswer)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    settings = PipelineSettings.from_env()
    with CloudSqlDatabase(settings) as conn:
        cursor = conn.cursor()
        cursor.execute(ADD_COLUMN_SQL)
        conn.commit()

        cursor.execute("""
            SELECT p.product_id, p.title, p.category, p.description, p.features, p.details,
                   (SELECT image_url FROM product_image i WHERE i.product_id = p.product_id
                     ORDER BY ordinal LIMIT 1)
            FROM product p WHERE p.gender IS NULL ORDER BY p.product_id;
        """)
        columns = ["product_id", "title", "category", "description", "features", "details", "image_url"]
        pending = [dict(zip(columns, row)) for row in cursor.fetchall()]
        print(f"{len(pending)} products have no gender yet.")

        from_listing, unresolved = 0, []
        for product in pending:
            gender = gender_from_listing(product["title"], product["details"])
            if gender:
                cursor.execute("UPDATE product SET gender = %s WHERE product_id = %s;",
                               (gender, product["product_id"]))
                from_listing += 1
            else:
                unresolved.append(product)
        conn.commit()
        print(f"Labelled {from_listing} from the listing; asking Gemini about {len(unresolved)}.")

        if unresolved:
            classifier = GeminiGenderClassifier(Gemini(settings))
            for number, product in enumerate(unresolved, start=1):
                try:
                    answer = classifier.classify(product, product["image_url"])
                except Exception as error:   # one bad call must not lose the rest
                    print(f"  [{number}/{len(unresolved)}] {product['product_id']} FAILED: {error}")
                    continue
                cursor.execute("UPDATE product SET gender = %s WHERE product_id = %s;",
                               (answer.gender, product["product_id"]))
                conn.commit()
                print(f"  [{number}/{len(unresolved)}] {answer.gender:<6} {product['title'][:60]}"
                      f"  ({answer.reason[:70]})")

        cursor.execute("SELECT category, gender, count(*) FROM product GROUP BY 1, 2 ORDER BY 1, 2;")
        print("\ncategory   gender   products")
        for category, gender, count in cursor.fetchall():
            print(f"{category:<10} {gender or '-':<8} {count:>8}")


if __name__ == "__main__":
    main()
