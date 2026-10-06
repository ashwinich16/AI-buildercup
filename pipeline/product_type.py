"""
Scout - what kind of product each one is (product.product_type)
=================================================================

Our categories are broad on purpose ("trousers" holds dress pants, chinos and
wide-leg trousers...). product_type says which kind each product is, so a
shopper who asks for chinos hears about chinos, not every pair of trousers.

Gemini picks the type from a fixed list per category (CATALOG), so the
values are predictable enough to filter on. It also checks that each product
really fits its category's description, and sorts out the ones that don't:
  - it fits another of the shop's categories -> moved there, with its kind
  - it fits none of them (e.g. a dress)       -> excluded, with the reason
Excluded products stay in the database but export.py leaves them out of the
shop, and ingestion.py tops the category up again on its next run. To bring
one back:  UPDATE product SET excluded_reason = NULL WHERE product_id = '...';

Run after ingestion.py (and gender.py). Re-runnable: it only touches rows with
no product_type yet. To redo everything:  UPDATE product SET product_type = NULL;

Run:
  python pipeline/product_type.py
"""

from typing import Optional

from google.genai import types
from pydantic import BaseModel, Field

from common import CloudSqlDatabase, Gemini, PipelineSettings, image_part

# THE CATALOG: what each category is and the kinds (product_type) it holds, each in plain words. Gemini
# checks every product against its category's description and picks its kind from this list, so a new
# catalog (electronics, home...) means editing this dictionary, not the code.
CATALOG = {
    "shirt": {
        "description": "Shirts for adults, men's or women's: collared woven button-up shirts and knit polo shirts.",
        "types": {
            "shirt": "button-up or collared woven shirt (dress, office, casual, oxford, flannel, linen)",
            "polo": "knit shirt with a collar and a short button placket",
        },
    },
    "top": {
        "description": "Women's tops for adults that are not collared button-up shirts: blouses, t-shirts, "
                       "tunics, tank tops, sweatshirts and similar.",
        "types": {
            "t-shirt": "collarless knit tee, short or long sleeve, including graphic tees",
            "blouse": "women's dressy top in woven or sheer fabric, usually without a full button-up collar",
            "tunic": "long top meant to cover the hips",
            "tank top": "sleeveless top",
            "sweatshirt": "pullover in sweatshirt or fleece fabric",
            "athletic top": "compression, workout or running top",
            "thermal top": "base layer worn for warmth",
        },
    },
    "trousers": {
        "description": "Trousers and pants for adults, men's or women's, in any length or fit, that are not "
                       "denim jeans, shorts, leggings or sweatpants.",
        "types": {
            "dress pants": "tailored trousers for office or formal wear, including suit trousers and slacks",
            "chinos": "cotton twill trousers, smart-casual",
            "wide-leg trousers": "loose, flowing trousers with a wide leg, usually women's",
            "casual trousers": "any other everyday trousers: elastic-waist, linen, cropped, capri, flare, outdoor",
        },
    },
    "jeans": {
        "description": "Full-length denim jeans for adults, men's or women's.",
        "types": {"jeans": ""},
    },
    "jacket": {
        "description": "Jackets and outerwear for adults: jackets, coats, blazers, hoodies and vests.",
        "types": {"jacket": "", "coat": "", "blazer": "", "hoodie": "", "vest": ""},
    },
    "shoe": {
        "description": "Shoes and other footwear for adults.",
        "types": {"sneaker": "", "boot": "", "sandal": "", "heel": "", "flat": "", "loafer": "", "slipper": "",
                  "dress shoe": "formal lace-up or slip-on leather shoe, such as an oxford or derby"},
    },
    "bag": {
        "description": "Bags for adults: handbags, totes, backpacks, crossbody bags, laptop bags and briefcases.",
        "types": {
            "handbag": "", "tote": "", "backpack": "", "crossbody bag": "", "clutch": "", "wristlet": "",
            "shopping bag": "",
            "laptop bag": "bag or backpack made to carry a laptop (padded laptop sleeve or compartment)",
            "briefcase": "structured business case for documents and a laptop",
        },
    },
    "accessory": {
        "description": "Wearable fashion accessories for adults: ties, cufflinks, belts, watches, wallets, "
                       "scarves, hats, jewellery and sunglasses.",
        "types": {"sunglasses": "", "hat": "", "belt": "", "scarf": "", "jewelry": "", "watch": "", "wallet": "",
                  "socks": "", "hair accessory": "", "tie": "necktie or bow tie", "cufflinks": "",
                  "tie clip": "tie clip or tie bar"},
    },
}

# Kinds of product per category (the spelling stored in product.product_type).
TYPES_BY_CATEGORY = {category: list(spec["types"]) for category, spec in CATALOG.items()}


def kinds_text(category: str) -> str:
    """The allowed kinds of a category, each with what it means when that is not obvious."""
    return "\n".join(f"- {kind}: {meaning}" if meaning else f"- {kind}"
                      for kind, meaning in CATALOG.get(category, {}).get("types", {}).items())


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


PROMPT = """You are sorting products in a shop's catalog.

The product is filed under the category "{category}", which the shop describes as:
{category_description}

The allowed kinds for that category are:
{allowed}

The shop's categories are:
{categories}

Decide:
- belongs: does this product really fit the description of "{category}"? Judge what the product itself is,
  not words that happen to appear in its title.
- product_type: if it belongs, the ONE allowed kind that fits best, spelled exactly as listed.
- better_category: if it does not belong, which of the shop's categories it fits instead, or null.

Title: {title}
Description: {description}
Features: {features}
"""


class GeminiTypeClassifier:
    def __init__(self, gemini: Gemini):
        self._gemini = gemini

    def classify(self, product: dict, image_url: Optional[str]) -> TypeAnswer:
        category = product["category"]
        parts = [types.Part.from_text(text=PROMPT.format(
            category=category,
            category_description=CATALOG.get(category, {}).get("description", "(not in the catalog)"),
            allowed=kinds_text(category),
            categories="\n".join(f"- {name}: {spec['description']}" for name, spec in CATALOG.items()),
            title=product["title"],
            description=" ".join(product["description"] or [])[:1500],
            features=" | ".join(product["features"] or [])[:1500],
        ))]
        image = image_part(image_url)
        if image:
            parts.append(image)
        return self._gemini.answer(parts, TypeAnswer)


def valid_type(category: str, product_type: Optional[str]) -> Optional[str]:
    """The allowed spelling of product_type for this category, or None if it isn't one."""
    allowed = {t.lower(): t for t in TYPES_BY_CATEGORY.get(category, [])}
    return allowed.get((product_type or "").strip().lower())


def main():
    settings = PipelineSettings.from_env()
    with CloudSqlDatabase(settings) as conn:
        cursor = conn.cursor()
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

        classifier = GeminiTypeClassifier(Gemini(settings))
        outcomes = {"typed": 0, "moved": 0, "unsure": 0, "excluded": 0, "failed": 0}
        for number, product in enumerate(pending, start=1):
            try:
                outcome = sort_product(cursor, classifier, product)
                conn.commit()
            except Exception as error:   # one bad call must not lose the rest; the next run retries it
                conn.rollback()
                outcome = f"failed ({error})"
            outcomes[outcome.split()[0]] += 1
            print(f"  [{number}/{len(pending)}] {product['category']:<9} {outcome:<30} {product['title'][:55]}")
        print(f"\nDone: {outcomes}")
        if outcomes["excluded"]:
            print("Excluded products leave gaps: run `python pipeline/ingestion.py` again to top the categories up.")


def sort_product(cursor, classifier: GeminiTypeClassifier, product: dict) -> str:
    """Give the product its kind; move it if it belongs in another category; exclude it if it fits none."""
    category = product["category"]
    answer = check(classifier, product, category)
    if answer.belongs and valid_type(category, answer.product_type):
        cursor.execute("UPDATE product SET product_type = %s WHERE product_id = %s;",
                       (valid_type(category, answer.product_type), product["product_id"]))
        return f"typed {valid_type(category, answer.product_type)}"

    if answer.belongs:   # it fits, Gemini just named a kind that isn't on the list: leave it for the next run
        return "unsure (kind not on the list)"

    better = answer.better_category if answer.better_category in CATALOG and answer.better_category != category else None
    if better:
        moved = check(classifier, {**product, "category": better}, better)
        kind = valid_type(better, moved.product_type) if moved.belongs else None
        if kind:
            cursor.execute("UPDATE product SET category = %s, product_type = %s WHERE product_id = %s;",
                           (better, kind, product["product_id"]))
            return f"moved -> {better} / {kind}"

    reason = f"does not fit {category}: {answer.reason}"[:300]
    cursor.execute("UPDATE product SET excluded_reason = %s WHERE product_id = %s;", (reason, product["product_id"]))
    return "excluded (fits no category)"


def check(classifier: GeminiTypeClassifier, product: dict, category: str) -> TypeAnswer:
    """Gemini's verdict for the product in this category (no call needed when the category has one kind)."""
    kinds = TYPES_BY_CATEGORY.get(category, [])
    if len(kinds) == 1:   # e.g. jeans: nothing to decide
        return TypeAnswer(belongs=True, product_type=kinds[0], better_category=None, reason="only kind in its category")
    return classifier.classify(product, product["image_url"])


if __name__ == "__main__":
    main()
