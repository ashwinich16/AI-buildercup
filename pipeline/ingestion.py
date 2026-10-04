"""
Scout — Catalog Ingestion Script (Step 3)
==========================================

Run this in YOUR OWN Cloud Shell / a VM with normal internet access —
not in a network-restricted sandbox. It needs to reach the McAuley Lab
dataset host and your Cloud SQL instance.

What it does, in order:
  1. Connects to Cloud SQL Postgres securely via the Cloud SQL Python
     Connector, pulling the app password from Secret Manager.
  2. Streams the Amazon Reviews 2023 item-metadata file (Amazon_Fashion
     subset) line by line — never loads the whole file into memory.
  3. Applies curation criteria to pick a small, demo-quality catalog
     (TARGET_BUCKETS, skipping products already in the DB) instead of
     ingesting the full noisy long-tail.
  4. Buckets each item into our own `category` (top/jeans/jacket/shoe/bag/
     accessory) instead of trusting Amazon's raw taxonomy.
  5. Inserts the curated products + their catalog images.
  6. Streams the matching reviews file, keeps only reviews whose
     parent_asin is one of our curated products, caps + ranks reviews
     per product, and inserts them.

Install first:
  pip install "cloud-sql-python-connector[pg8000]" google-cloud-secret-manager
"""

import gzip
import json
import re
import urllib.request
from collections import defaultdict

from google.cloud.sql.connector import Connector
from google.cloud import secretmanager
import pg8000

from gender import ADD_COLUMN_SQL as ADD_GENDER_COLUMN_SQL, gender_from_listing


# ---------------------------------------------------------------------
# CONFIG — fill these in for your project
# ---------------------------------------------------------------------
PROJECT_ID = "scout-hackathon"
REGION = "asia-south2"                 # region your Cloud SQL instance is in
INSTANCE_NAME = "scoutdb1"             # Cloud SQL instance id
DB_NAME = "scout_catalog"
DB_USER = "scout_app"
DB_PASSWORD_SECRET_ID = "scoutdb1-password"   # secret created in Step 1

# Dataset source (McAuley Lab, Amazon Reviews 2023)
# Hugging Face mirror (https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), uncompressed JSONL.
# Amazon_Fashion rather than Clothing_Shoes_and_Jewelry: its reviews file is ~1 GB vs ~28 GB, and
# curate_reviews has to stream the whole thing.
HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
METADATA_URL = f"{HF_BASE}/meta_categories/meta_Amazon_Fashion.jsonl"
REVIEWS_URL = f"{HF_BASE}/review_categories/Amazon_Fashion.jsonl"

# Curation knobs
# How many NEW products to add per bucket on this run. Products already in the DB are skipped,
# so re-running with a bigger number tops a bucket up rather than re-adding the same items.
# (The original 125-item catalog was 25 each of shirt/jacket/shoe/bag/accessory; "shirt" is now "top",
# since that bucket holds t-shirts, blouses and polos too. product_type.py says which kind each one is.)
TARGET_BUCKETS = {"top": 25, "jeans": 25}
MIN_RATING_NUMBER = 15        # item must have at least this many ratings to be "real" enough to compare
MAX_REVIEWS_PER_PRODUCT = 20  # cap so the review corpus per item stays small enough for Gemini summarization


# ---------------------------------------------------------------------
# 1. DATABASE CONNECTION
# ---------------------------------------------------------------------
# WHY the Cloud SQL Python Connector instead of a plain host:port
# connection: it authenticates using IAM + the instance connection name
# rather than a public IP you have to allowlist, so this script works
# unmodified whether it's run from Cloud Shell, a laptop, or later from
# a Cloud Run job — one less thing to misconfigure under deadline
# pressure.
#
# WHY the password comes from Secret Manager instead of being pasted
# here: this file is going into the GitHub repo for the hackathon
# submission. A hardcoded DB password in a public repo is the single
# most common accidental leak in projects like this.

def get_db_password() -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{DB_PASSWORD_SECRET_ID}/versions/latest"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8").strip()


def get_connection():
    connector = Connector()
    instance_connection_name = f"{PROJECT_ID}:{REGION}:{INSTANCE_NAME}"

    def getconn():
        return connector.connect(
            instance_connection_name,
            "pg8000",
            user=DB_USER,
            password=get_db_password(),
            db=DB_NAME,
        )

    conn = getconn()
    return conn, connector


# ---------------------------------------------------------------------
# 2. CATEGORY BUCKETING (our own taxonomy, decided in Step 2)
# ---------------------------------------------------------------------
# WHY: Amazon's `categories` breadcrumb is inconsistent and too granular
# ("Women > Clothing > Active > Active Tops > ...") to be useful as a
# comparison grouping. The comparison agent needs a small, predictable
# set of buckets so it only ever compares like-with-like.

CATEGORY_RULES = [
    ("shoe", re.compile(r"\b(shoe|sneaker|boot|sandal|heel|loafer|flat)\b", re.I)),
    ("bag", re.compile(r"\b(bag|backpack|tote|purse|handbag|clutch|wristlet)\b", re.I)),
    ("jacket", re.compile(r"\b(jacket|coat|blazer|hoodie|parka)\b", re.I)),
    ("accessory", re.compile(r"\b(belt|scarf|hat|sunglasses|jewelry|watch|wallet)\b", re.I)),
    # No bare "top" keyword: it matched "Top Zip Wristlet" and "Tankini Top".
    # No bare "polo" either: it matched the brand "U.S. Polo Assn." on joggers.
    ("top", re.compile(r"\b(shirts?|t-shirts?|tees?|blouses?|button[- ]down|henley)\b", re.I)),
    # After "top" so brand names like "Calvin Klein Jeans Men's T-Shirt" stay tops.
    # "jean(?!-)" so "Jean-Michel Jarre" (a concert DVD in this file) isn't jeans.
    ("jeans", re.compile(r"\b(jeans|jean(?!-)|denim pants?)\b", re.I)),
]

# Titles that keyword-match a bucket but aren't the garment itself: swimwear "tops", keychain
# "bag" charms, kids' sizes, jean shorts/skirts, jeggings, tank tops and sleepwear sold as
# "shirts", medical/collar add-ons, and sewing/fit accessories "for jeans".
EXCLUDE_PATTERN = re.compile(
    r"\b(swim\w*|tankini|bikini|keychain|charms?|pendant|costume|baby|toddler|girl'?s|boy'?s|kids|"
    r"shorts?|skirts?|dress(es)?|overalls?|jeggings?|leggings?|joggers?|sweatpants|"
    r"tanks?|a-shirts?|camis?|nightshirts?|nightgowns?|sleepwear|pajamas?|sweaters?|cardigans?|"
    r"surgery|mastectomy|recovery|collars?|leather pants|aprons?|extenders?|stretchers?|patch(es)?|hangers?)\b",
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
# 3. STREAMING JSONL READER
# ---------------------------------------------------------------------
# WHY streaming: the raw metadata/review files are several GB each.
# Reading them into memory would either crash or make this script
# unusable on a small Cloud Shell VM. Line-by-line iteration keeps
# memory flat regardless of file size; only the curated candidates are
# kept in memory.

def stream_jsonl(url: str):
    # Handles both the gzipped McAuley Lab files and the plain .jsonl Hugging Face mirror.
    req = urllib.request.Request(url, headers={"User-Agent": "scout-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        lines = gzip.GzipFile(fileobj=resp) if url.endswith(".gz") else resp
        for line in lines:
            yield json.loads(line)


# ---------------------------------------------------------------------
# 4. PRODUCT CURATION
# ---------------------------------------------------------------------
# WHY these specific filters, and why a per-category cap:
#   - rating_number >= MIN_RATING_NUMBER: filters out near-zero-review
#     listings, so every curated product actually has real review text
#     for the narration/comparison agents to draw on (we never fabricate
#     review evidence, so if there's no review text, the feature is dead
#     weight for the demo).
#   - must have at least one image: this is a vision-dependent shopping
#     assistant — an item Gemini can't see is useless in the demo.
#   - non-empty description or features: gives the visual-attributes
#     extraction something real to ground against, not just a title.
#   - capped per bucket: we want the demo catalog spread across
#     shirts/jackets/shoes/bags/accessories, not 125 random shirts,
#     because the comparison agent's value only shows up when there
#     are multiple comparable items in the same bucket.
#   - most-rated first: we scan the whole file and keep the top
#     TARGET_BUCKETS[bucket] items by rating_number, instead of the
#     first matches in file order, so the additions are well-known,
#     review-rich products.

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


def curate_products(metadata_iter, existing_ids: set):
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
        if category not in TARGET_BUCKETS:
            continue  # only fill the buckets this run is targeting

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
        selected.extend(ranked[:TARGET_BUCKETS[category]])
    return selected


def title_key(title: str) -> str:
    # Drop bracketed size/colour suffixes and punctuation; the first 60 chars identify the product.
    stripped = re.sub(r"[\(\[].*?[\)\]]", " ", title.lower())
    return re.sub(r"[^a-z0-9]+", "", stripped)[:60]


# ---------------------------------------------------------------------
# 5. REVIEW CURATION
# ---------------------------------------------------------------------
# WHY we buffer reviews per product before inserting, instead of
# inserting as we go: the review file is not sorted by product, and we
# want the BEST reviews per product (verified + most helpful), not just
# the first N we happen to encounter. WHY cap at MAX_REVIEWS_PER_PRODUCT:
# bounds how much text the Comparison/Narration agents load per item,
# keeping prompt size and latency predictable in the demo.

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
# 6. INSERTS
# ---------------------------------------------------------------------
# WHY ON CONFLICT DO NOTHING on product_id: makes the script safe to
# re-run (idempotent) while you're iterating on curation rules, without
# needing to truncate tables every time.

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


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("Connecting to Cloud SQL...")
    conn, connector = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(ADD_GENDER_COLUMN_SQL)   # no-op once the column exists
        conn.commit()

        cursor.execute("SELECT product_id FROM product;")
        existing_ids = {row[0] for row in cursor.fetchall()}
        print(f"{len(existing_ids)} products already in the catalog; adding {TARGET_BUCKETS}.")

        print(f"Streaming metadata from {METADATA_URL} ...")
        products = curate_products(stream_jsonl(METADATA_URL), existing_ids)
        counts = defaultdict(int)
        for p in products:
            counts[p["category"]] += 1
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

        unlabelled = sum(1 for p in products if not p["gender"])
        print(f"Done. {unlabelled} new products need `python pipeline/gender.py` to label their gender.")
    finally:
        cursor.close()
        conn.close()
        connector.close()


if __name__ == "__main__":
    main()