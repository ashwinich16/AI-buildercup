"""
Scout - what kind of product each one is (product.product_type)
=================================================================

Our categories are broad on purpose ("top" holds shirts, t-shirts, blouses,
polos...). product_type says which kind each product is, so a shopper who
asks for a shirt hears about shirts, not every top in the shop.

Gemini picks the type from a fixed list per category (TYPES_BY_CATEGORY), so
the values are predictable enough to filter on. It also flags products that
do not belong in their category at all (a wristlet filed under tops); those
are reported and left without a type for a person to deal with:
  move it:     UPDATE product SET category = 'bag', product_type = 'wristlet' WHERE product_id = '...';
  exclude it:  UPDATE product SET excluded_reason = 'why' WHERE product_id = '...';
Excluded products stay in the database but export.py leaves them out of the
shop; set excluded_reason back to NULL to bring one back.

Run after ingestion.py (and gender.py). Re-runnable: it only touches rows with
no product_type yet. To redo everything:  UPDATE product SET product_type = NULL;

Run:
  python pipeline/product_type.py
"""

import os
import sys
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gender import GeminiGenderClassifier  # noqa: E402  (reuses its photo download)

# Kinds of product per category, in the words a shopper would use.
TYPES_BY_CATEGORY = {
    "top": ["shirt", "t-shirt", "polo", "blouse", "tunic", "tank top", "sweatshirt", "athletic top", "thermal top"],
    "jeans": ["jeans"],
    "jacket": ["jacket", "coat", "blazer", "hoodie", "vest"],
    "shoe": ["sneaker", "boot", "sandal", "heel", "flat", "loafer", "slipper", "dress shoe"],
    "bag": ["handbag", "tote", "backpack", "crossbody bag", "clutch", "wristlet", "shopping bag"],
    "accessory": ["sunglasses", "hat", "belt", "scarf", "jewelry", "watch", "wallet", "socks", "hair accessory"],
}

# What each top type means, so "shirt" stays the button-up/collared kind people mean by it.
TYPE_HINTS = """- shirt: button-up or collared shirt (dress, casual, flannel, Hawaiian, camp collar)
- t-shirt: collarless knit tee, short or long sleeve, including graphic tees
- polo: knit shirt with a collar and a short button placket
- blouse: women's dressy top in woven or sheer fabric
- tunic: long top meant to cover the hips
- tank top: sleeveless top
- sweatshirt: pullover in sweatshirt or fleece fabric
- athletic top: compression, workout or running top
- thermal top: base layer worn for warmth"""

ADD_COLUMNS_SQL = (   # one statement per execute: pg8000 does not run several at once
    "ALTER TABLE product ADD COLUMN IF NOT EXISTS product_type TEXT;",
    "ALTER TABLE product ADD COLUMN IF NOT EXISTS excluded_reason TEXT;",
)


class TypeAnswer(BaseModel):
    belongs: bool = Field(description="False if the product is not really a kind of the given category")
    product_type: Optional[str] = Field(description="One of the allowed types, when it belongs")
    better_category: Optional[str] = Field(description="When it does not belong: the catalog category it "
                                                       "belongs to, or null if none fits")
    reason: str


PROMPT = """You are sorting products in a clothing shop's catalog.

The product is filed under the category "{category}". The allowed kinds for that category are:
{allowed}

For tops, these kinds mean:
{hints}

The shop's categories are: {categories}.

Decide:
- belongs: is this product really a {category}? (A bag, swimsuit, pyjamas, underwear or a dress is not a top;
  socks or shoelaces are not shoes.)
- product_type: if it belongs, the ONE allowed kind that fits best, spelled exactly as listed.
- better_category: if it does not belong, which of the shop's categories it should be in, or null.

Title: {title}
Description: {description}
Features: {features}
"""


class GeminiTypeClassifier:
    def __init__(self, project_id: str):
        self._client = genai.Client(vertexai=True, project=project_id,
                                    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
        self._model = os.environ.get("SCOUT_ENRICH_MODEL", "gemini-2.5-flash")

    def classify(self, product: dict, image_url: Optional[str]) -> TypeAnswer:
        category = product["category"]
        parts = [types.Part.from_text(text=PROMPT.format(
            category=category,
            allowed=", ".join(TYPES_BY_CATEGORY.get(category, [])),
            hints=TYPE_HINTS,
            categories=", ".join(TYPES_BY_CATEGORY),
            title=product["title"],
            description=" ".join(product["description"] or [])[:1500],
            features=" | ".join(product["features"] or [])[:1500],
        ))]
        image = GeminiGenderClassifier._fetch(image_url)
        if image:
            parts.append(image)
        response = self._client.models.generate_content(
            model=self._model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=TypeAnswer,
            ),
        )
        return TypeAnswer.model_validate_json(response.text)


def valid_type(category: str, product_type: Optional[str]) -> Optional[str]:
    """The allowed spelling of product_type for this category, or None if it isn't one."""
    allowed = {t.lower(): t for t in TYPES_BY_CATEGORY.get(category, [])}
    return allowed.get((product_type or "").strip().lower())


def main():
    import ingestion   # here, not at the top, to match gender.py

    conn, connector = ingestion.get_connection()
    cursor = conn.cursor()
    try:
        for statement in ADD_COLUMNS_SQL:
            cursor.execute(statement)
        conn.commit()

        cursor.execute("""
            SELECT p.product_id, p.title, p.category, p.description, p.features,
                   (SELECT image_url FROM product_image i WHERE i.product_id = p.product_id
                     ORDER BY ordinal LIMIT 1)
            FROM product p WHERE p.product_type IS NULL AND p.excluded_reason IS NULL
            ORDER BY p.category, p.product_id;
        """)
        columns = ["product_id", "title", "category", "description", "features", "image_url"]
        pending = [dict(zip(columns, row)) for row in cursor.fetchall()]
        print(f"{len(pending)} products have no product_type yet.")

        classifier = GeminiTypeClassifier(ingestion.PROJECT_ID)
        misfits = []
        for number, product in enumerate(pending, start=1):
            if len(TYPES_BY_CATEGORY.get(product["category"], [])) == 1:   # e.g. jeans: nothing to decide
                answer = TypeAnswer(belongs=True, product_type=TYPES_BY_CATEGORY[product["category"]][0],
                                    better_category=None, reason="only kind in its category")
            else:
                try:
                    answer = classifier.classify(product, product["image_url"])
                except Exception as error:   # one bad call must not lose the rest
                    print(f"  [{number}/{len(pending)}] {product['product_id']} FAILED: {error}")
                    continue

            product_type = valid_type(product["category"], answer.product_type) if answer.belongs else None
            if product_type:
                cursor.execute("UPDATE product SET product_type = %s WHERE product_id = %s;",
                               (product_type, product["product_id"]))
                conn.commit()
                print(f"  [{number}/{len(pending)}] {product['category']:<9} {product_type:<13} "
                      f"{product['title'][:60]}")
            else:
                misfits.append((product, answer))
                print(f"  [{number}/{len(pending)}] {product['category']:<9} {'?? MISFIT':<13} "
                      f"{product['title'][:60]}")

        if misfits:
            print("\nNot a fit for their category (left without a type; move or remove them):")
            for product, answer in misfits:
                print(f"  {product['product_id']}  [{product['category']} -> {answer.better_category or 'none'}]"
                      f"  {product['title'][:70]}\n      {answer.reason[:140]}")
    finally:
        cursor.close()
        conn.close()
        connector.close()


if __name__ == "__main__":
    main()
