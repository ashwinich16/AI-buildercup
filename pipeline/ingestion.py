"""
Scout — Catalog Ingestion Script
================================

Run this in Cloud Shell or a laptop with normal internet access. It needs to reach the
McAuley Lab dataset (Hugging Face mirror) and your Cloud SQL instance. Settings come from .env.

What it does, in order:
  1. Connects to Cloud SQL Postgres via the Cloud SQL Python Connector, pulling the app
     password from Secret Manager.
  2. Streams the Amazon Reviews 2023 item-metadata file (Amazon_Fashion subset) line by line.
  3. Tops every category up to TARGET_PER_CATEGORY products (office wear for shirts, trousers,
     accessories and bags), half men's and half women's, skipping products already in the DB.
  4. Buckets each item into our own `category` instead of trusting Amazon's raw taxonomy.
  5. Inserts the curated products + their catalog images.
  6. Streams the matching reviews file, keeps the best reviews of our curated products, inserts them.
  7. Gives every product with no price (new and existing) a price within ±PRICE_SPREAD of its
     category's average, marked price_estimated = TRUE.

Install: see pipeline/common.py
"""

import gzip
import json
import random
import re
import urllib.request
from collections import defaultdict

from common import CloudSqlDatabase, PipelineSettings
from gender import ADD_COLUMN_SQL as ADD_GENDER_COLUMN_SQL, gender_from_listing


# ---------------------------------------------------------------------
# CONFIG (Cloud SQL settings come from .env, see common.py)
# ---------------------------------------------------------------------

# Dataset source (McAuley Lab, Amazon Reviews 2023), Hugging Face mirror, uncompressed JSONL.
# Amazon_Fashion rather than Clothing_Shoes_and_Jewelry: its reviews file is ~1 GB vs ~28 GB.
HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
METADATA_URL = f"{HF_BASE}/meta_categories/meta_Amazon_Fashion.jsonl"
REVIEWS_URL = f"{HF_BASE}/review_categories/Amazon_Fashion.jsonl"

# Curation knobs
# Every category is topped up to this many products (half men's, half women's). Products already
# in the DB count towards it and are never added twice, so re-running only fills what is missing.
TARGET_PER_CATEGORY = 50
MIN_RATING_NUMBER = 15        # item must have at least this many ratings to be "real" enough to compare
MAX_REVIEWS_PER_PRODUCT = 20  # cap so the review corpus per item stays small enough for Gemini summarization
PRICE_SPREAD = 10.0           # missing prices are filled within category average ± this


# ---------------------------------------------------------------------
# 1. CATEGORY BUCKETING (our own taxonomy)
# ---------------------------------------------------------------------
# Amazon's `categories` breadcrumb is inconsistent and too granular to be a comparison grouping,
# so we bucket by the title. First match wins. The shirt, trousers, bag and accessory buckets only take
# office wear, which is what this run adds for the demo. pipeline/product_type.py then has Gemini check
# every product against its category and moves or excludes any that the keywords got wrong.

CATEGORY_RULES = [
    # Office shirts, men's and women's. "polo" is not here: it's not office wear, and it matched the
    # brand "U.S. Polo Assn.".
    ("shirt", re.compile(r"\b(dress shirts?|formal shirts?|office shirts?|work shirts?|business shirts?|"
                         r"oxford shirts?|button[- ](down|up) shirts?|non[- ]iron)\b", re.I)),
    # Office trousers, men's and women's. Checked before shoes, so "Flat Front Dress Pants" isn't a flat.
    ("trousers", re.compile(r"\b(dress pants?|formal pants?|office pants?|work pants?|suit pants?|"
                            r"business pants?|slacks|trousers?|chinos?)\b", re.I)),
    ("shoe", re.compile(r"\b(shoes?|sneakers?|boots?|sandals?|heels?|loafers?|flats)\b", re.I)),
    ("jacket", re.compile(r"\b(jackets?|coats?|blazers?|hoodies?|parkas?)\b", re.I)),
    # No bare "top" keyword: it matched "Top Zip Wristlet" and "Tankini Top".
    # Women's tops (other shirts belong in "shirt", which only takes office shirts this run).
    ("top", re.compile(r"\b(t-shirts?|tees?|blouses?|tunics?|henley)\b", re.I)),
    # After the others so brand names like "Calvin Klein Jeans Men's Dress Shirt" aren't jeans.
    # "jean(?!-)" so "Jean-Michel Jarre" (a concert DVD in this file) isn't jeans.
    ("jeans", re.compile(r"\b(jeans|jean(?!-)|denim pants?)\b", re.I)),
    # Bags and accessories come last: garments often mention them as extras ("coat with belt",
    # "jacket with laptop pocket"), and those are still garments.
    # Office bags: laptop bags, briefcases and work bags.
    ("bag", re.compile(r"\b(laptop|computer|briefcases?|messenger bags?|work (bag|tote)s?|business bags?)\b", re.I)),
    # Office accessories: ties, cufflinks, tie clips, belts, watches, wallets, card holders.
    ("accessory", re.compile(r"\b(neck ?ties?|ties?(?![- ]dye)|tie (clips?|bars?)|cufflinks?|belts?|watch(es)?|wallets?|"
                             r"card holders?)\b", re.I)),
]

# Titles that keyword-match a bucket but aren't the garment itself: swimwear, kids' sizes, shorts,
# skirts, dresses (but not "dress shirt" / "dress pants"), sportswear and sleepwear, add-ons "for" a garment.
EXCLUDE_PATTERN = re.compile(
    r"\b(swim\w*|tankini|bikini|keychain|charms?|pendant|costume|baby|toddler|girl'?s|boy'?s|kids|"
    r"shorts|skirts?|dresses|shirt ?dress|dress(?![\s-]+(shirts?|pants?|shoes?|slacks|trousers?|belts?|watch(es)?|socks?))|overalls?|jeggings?|leggings?|joggers?|sweatpants|yoga|cargo|"
    r"tanks?|a-shirts?|camis?|nightshirts?|nightgowns?|sleepwear|pajamas?|pyjamas?|nightwear|lederhosen|sweaters?|cardigans?|"
    r"belt bags?|fanny packs?|surgery|mastectomy|recovery|leather pants|aprons?|extenders?|stretchers?|patch(es)?|hangers?)\b",
    re.I,
)


def bucket_category(title: str, categories: list) -> str:
    if EXCLUDE_PATTERN.search(title or ""):
        return "other"
    haystack = " ".join([title or ""] + (categories or []))
    for label, pattern in CATEGORY_RULES:
        if pattern.search(haystack):
            return label
    return "other"


# ---------------------------------------------------------------------
# 2. STREAMING JSONL READER
# ---------------------------------------------------------------------
# The files are large, so they are read line by line: memory stays flat; only the curated
# candidates are kept.

def stream_jsonl(url: str):
    # Handles both the gzipped McAuley Lab files and the plain .jsonl Hugging Face mirror.
    req = urllib.request.Request(url, headers={"User-Agent": "scout-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        lines = gzip.GzipFile(fileobj=resp) if url.endswith(".gz") else resp
        for line in lines:
            yield json.loads(line)


# ---------------------------------------------------------------------
# 3. PRODUCT CURATION
# ---------------------------------------------------------------------
# Filters: enough ratings (so there is real review text), an image (Scout describes photos),
# a description or features (so Gemini has text to ground attributes in). Most-rated first,
# half men's and half women's per bucket.

def clean_text(value):
    # Postgres TEXT rejects NUL bytes, which do appear in some scraped Amazon text.
    return value.replace("\x00", "") if isinstance(value, str) else value


def parse_price(value):
    # The 2023 dump stores price as a float, null, or occasionally a string like "None" / "$12.99".
    if isinstance(value, (int, float)):
        return value
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def curate_products(metadata_iter, existing_ids: set, needed: dict):
    candidates = defaultdict(list)

    for item in metadata_iter:
        parent_asin = item.get("parent_asin")
        title = item.get("title")
        if not parent_asin or not title or parent_asin in existing_ids:
            continue

        rating_number = item.get("rating_number") or 0
        if rating_number < MIN_RATING_NUMBER:
            continue

        images = item.get("images") or []
        if not images:
            continue

        description = item.get("description") or []
        features = item.get("features") or []
        if not description and not features:
            continue

        categories = item.get("categories") or []
        category = bucket_category(title, categories)
        if not needed.get(category):
            continue  # only fill the categories that still need products

        details = item.get("details") or {}
        brand = None
        if isinstance(details, dict):
            brand = details.get("Brand") or details.get("brand")

        candidates[category].append({
            "product_id": parent_asin,
            "title": clean_text(title),
            "store": item.get("store"),
            "brand": brand,
            "main_category": item.get("main_category"),
            "category": category,
            "categories": categories,
            "description": description,
            "features": features,
            "price": parse_price(item.get("price")),
            "average_rating": item.get("average_rating"),
            "rating_number": rating_number,
            "details": details,
            "images": images,
            # None when the listing doesn't say; pipeline/gender.py asks Gemini about those.
            "gender": gender_from_listing(title, details),
        })

    selected = []
    for category, items in candidates.items():
        # Sellers often list each size/colour as its own parent_asin with the same or a reworded
        # title ("...Shirt(02Purple-S)", "...Shirt(02Grey-M)"), so dedupe on the title and allow
        # one product per store per bucket, which also keeps the comparisons varied.
        seen = set()
        ranked = []
        for p in sorted(items, key=lambda p: p["rating_number"], reverse=True):
            keys = {p["product_id"], title_key(p["title"])}
            if p["store"]:
                keys.add("store:" + p["store"].lower())
            if not keys & seen:
                seen |= keys
                ranked.append(p)
        selected.extend(half_men_half_women(ranked, needed[category]))
    return selected


def half_men_half_women(ranked: list, count: int) -> list:
    """The best `count` products, half women's and half men's; unisex/unknown items fill any gap."""
    women = [p for p in ranked if p["gender"] == "women"][: count // 2]
    men = [p for p in ranked if p["gender"] == "men"][: count - len(women)]
    picked = women + men
    rest = [p for p in ranked if p not in picked]
    return picked + rest[: count - len(picked)]


def title_key(title: str) -> str:
    # Drop bracketed size/colour suffixes and punctuation; the first 60 chars identify the product.
    stripped = re.sub(r"[\(\[].*?[\)\]]", " ", title.lower())
    return re.sub(r"[^a-z0-9]+", "", stripped)[:60]


# ---------------------------------------------------------------------
# 4. REVIEW CURATION
# ---------------------------------------------------------------------
# The review file is not sorted by product, and we want the BEST reviews per product (verified +
# most helpful), so reviews are buffered per product, ranked, and capped at MAX_REVIEWS_PER_PRODUCT.

def curate_reviews(reviews_iter, wanted_parent_asins: set):
    buffer = defaultdict(list)

    for review in reviews_iter:
        parent_asin = review.get("parent_asin")
        if parent_asin not in wanted_parent_asins:
            continue

        rating = review.get("rating")
        if rating is None:
            continue

        buffer[parent_asin].append({
            "product_id": parent_asin,
            "asin": review.get("asin"),
            "rating": rating,
            "title": clean_text(review.get("title")),
            "text": clean_text(review.get("text")),
            "verified_purchase": review.get("verified_purchase"),
            "helpful_vote": review.get("helpful_vote") or 0,
            "review_timestamp": review.get("timestamp"),
        })

    final = []
    for parent_asin, reviews in buffer.items():
        ranked = sorted(
            reviews,
            key=lambda r: (r["verified_purchase"] is True, r["helpful_vote"]),
            reverse=True,
        )
        final.extend(ranked[:MAX_REVIEWS_PER_PRODUCT])

    return final


# ---------------------------------------------------------------------
# 5. INSERTS AND PRICES
# ---------------------------------------------------------------------
# ON CONFLICT DO NOTHING on product_id makes the script safe to re-run.

INSERT_PRODUCT_SQL = """
INSERT INTO product (
    product_id, title, store, brand, main_category, category,
    categories, description, features, price, average_rating,
    rating_number, details, gender
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
ON CONFLICT (product_id) DO NOTHING;
"""

INSERT_IMAGE_SQL = """
INSERT INTO product_image (product_id, image_url, variant, ordinal)
VALUES (%s, %s, %s, %s);
"""

INSERT_REVIEW_SQL = """
INSERT INTO review (
    product_id, asin, rating, title, text,
    verified_purchase, helpful_vote, review_timestamp
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s
);
"""

ADD_PRICE_ESTIMATED_SQL = "ALTER TABLE product ADD COLUMN IF NOT EXISTS price_estimated BOOLEAN NOT NULL DEFAULT FALSE;"


def insert_products(cursor, products):
    for p in products:
        cursor.execute(INSERT_PRODUCT_SQL, (
            p["product_id"], p["title"], p["store"], p["brand"],
            p["main_category"], p["category"], p["categories"],
            p["description"], p["features"], p["price"],
            p["average_rating"], p["rating_number"], json.dumps(p["details"]), p["gender"],
        ))

        # product_image has no unique key, so clear first to keep re-runs from duplicating rows.
        cursor.execute("DELETE FROM product_image WHERE product_id = %s;", (p["product_id"],))
        for ordinal, img in enumerate(p["images"]):
            url = img.get("hi_res") or img.get("large") or img.get("thumb")
            variant = img.get("variant") or ("hi_res" if img.get("hi_res") else "large")
            if url:
                cursor.execute(INSERT_IMAGE_SQL, (p["product_id"], url, variant, ordinal))


def insert_reviews(cursor, reviews):
    # Same as images: review has no natural unique key, so replace each product's reviews wholesale.
    for product_id in {r["product_id"] for r in reviews}:
        cursor.execute("DELETE FROM review WHERE product_id = %s;", (product_id,))
    for r in reviews:
        cursor.execute(INSERT_REVIEW_SQL, (
            r["product_id"], r["asin"], r["rating"], r["title"], r["text"],
            r["verified_purchase"], r["helpful_vote"], r["review_timestamp"],
        ))


def fill_missing_prices(cursor) -> int:
    """Every product with no price (new and existing) gets a random price within ±PRICE_SPREAD of its
    category's average, so budget questions work for every product. Marked price_estimated = TRUE."""
    cursor.execute(ADD_PRICE_ESTIMATED_SQL)
    cursor.execute("SELECT category, avg(price) FROM product WHERE price > 0 GROUP BY category;")
    averages = {category: float(average) for category, average in cursor.fetchall()}
    cursor.execute("SELECT product_id, category FROM product WHERE price IS NULL OR price <= 0;")
    filled = 0
    for product_id, category in cursor.fetchall():
        if category not in averages:
            continue   # no priced product in this category to take an average from
        average = averages[category]
        price = round(random.uniform(max(1.0, average - PRICE_SPREAD), average + PRICE_SPREAD), 2)
        cursor.execute("UPDATE product SET price = %s, price_estimated = TRUE WHERE product_id = %s;",
                       (price, product_id))
        filled += 1
    return filled


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("Connecting to Cloud SQL...")
    with CloudSqlDatabase(PipelineSettings.from_env()) as conn:
        cursor = conn.cursor()
        cursor.execute(ADD_GENDER_COLUMN_SQL)   # no-ops once the columns exist
        cursor.execute("ALTER TABLE product ADD COLUMN IF NOT EXISTS excluded_reason TEXT;")
        conn.commit()

        cursor.execute("SELECT product_id FROM product;")
        existing_ids = {row[0] for row in cursor.fetchall()}
        # Excluded products (product_type.py / by hand) stay in the DB but don't count towards the 50.
        cursor.execute("SELECT category, count(*) FROM product WHERE excluded_reason IS NULL GROUP BY category;")
        counts = dict(cursor.fetchall())
        categories = list(dict.fromkeys(label for label, _ in CATEGORY_RULES))
        needed = {c: max(0, TARGET_PER_CATEGORY - counts.get(c, 0)) for c in categories}
        print(f"Products per category now: {counts}")
        print(f"Adding: {needed}")

        print(f"Streaming metadata from {METADATA_URL} ...")
        products = curate_products(stream_jsonl(METADATA_URL), existing_ids, needed)
        counts = defaultdict(int)
        for p in products:
            counts[(p["category"], p["gender"] or "unknown")] += 1
        print(f"Curated {len(products)} new products: {dict(counts)}")

        print("Inserting products + images...")
        insert_products(cursor, products)
        conn.commit()

        wanted_ids = {p["product_id"] for p in products}

        print(f"Streaming reviews from {REVIEWS_URL} ...")
        reviews = curate_reviews(stream_jsonl(REVIEWS_URL), wanted_ids)
        print(f"Curated {len(reviews)} reviews across {len(wanted_ids)} products.")

        print("Inserting reviews...")
        insert_reviews(cursor, reviews)
        conn.commit()

        filled = fill_missing_prices(cursor)
        conn.commit()
        print(f"Filled in an estimated price for {filled} products (price_estimated = TRUE).")

        unlabelled = sum(1 for p in products if not p["gender"])
        print(f"Done. {unlabelled} new products need `python pipeline/gender.py` to label their gender.")


if __name__ == "__main__":
    main()
