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
     (~125 items) instead of ingesting the full noisy long-tail.
  4. Buckets each item into our own `category` (shirt/jacket/shoe/bag/
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
METADATA_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Amazon_Fashion.jsonl.gz"
REVIEWS_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Amazon_Fashion.jsonl.gz"

# Curation knobs
TARGET_CATALOG_SIZE = 125
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
    ("bag", re.compile(r"\b(bag|backpack|tote|purse|handbag|clutch)\b", re.I)),
    ("jacket", re.compile(r"\b(jacket|coat|blazer|hoodie|parka)\b", re.I)),
    ("accessory", re.compile(r"\b(belt|scarf|hat|sunglasses|jewelry|watch|wallet)\b", re.I)),
    ("shirt", re.compile(r"\b(shirt|top|blouse|tee|t-shirt)\b", re.I)),
]


def bucket_category(title: str, categories: list) -> str:
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
# memory flat regardless of file size, and we stop early once we have
# enough curated products.

def stream_jsonl_gz(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "scout-ingest/1.0"})
    with urllib.request.urlopen(req) as resp:
        with gzip.GzipFile(fileobj=resp) as gz:
            for line in gz:
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

def curate_products(metadata_iter):
    per_bucket_cap = max(1, TARGET_CATALOG_SIZE // len(CATEGORY_RULES))
    bucket_counts = defaultdict(int)
    selected = []

    for item in metadata_iter:
        if len(selected) >= TARGET_CATALOG_SIZE:
            break

        parent_asin = item.get("parent_asin")
        title = item.get("title")
        if not parent_asin or not title:
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
        if category == "other":
            continue  # keep the demo catalog inside our five comparison buckets
        if bucket_counts[category] >= per_bucket_cap:
            continue

        details = item.get("details") or {}
        brand = None
        if isinstance(details, dict):
            brand = details.get("Brand") or details.get("brand")

        selected.append({
            "product_id": parent_asin,
            "title": title,
            "store": item.get("store"),
            "brand": brand,
            "main_category": item.get("main_category"),
            "category": category,
            "categories": categories,
            "description": description,
            "features": features,
            "price": item.get("price"),
            "average_rating": item.get("average_rating"),
            "rating_number": rating_number,
            "details": details,
            "images": images,
        })
        bucket_counts[category] += 1

    return selected


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
            "title": review.get("title"),
            "text": review.get("text"),
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
    rating_number, details
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
            p["average_rating"], p["rating_number"], json.dumps(p["details"]),
        ))

        for ordinal, img in enumerate(p["images"]):
            url = img.get("hi_res") or img.get("large") or img.get("thumb")
            variant = img.get("variant") or ("hi_res" if img.get("hi_res") else "large")
            if url:
                cursor.execute(INSERT_IMAGE_SQL, (p["product_id"], url, variant, ordinal))


def insert_reviews(cursor, reviews):
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
        print(f"Streaming metadata from {METADATA_URL} ...")
        products = curate_products(stream_jsonl_gz(METADATA_URL))
        print(f"Curated {len(products)} products.")

        print("Inserting products + images...")
        insert_products(cursor, products)
        conn.commit()

        wanted_ids = {p["product_id"] for p in products}

        print(f"Streaming reviews from {REVIEWS_URL} ...")
        reviews = curate_reviews(stream_jsonl_gz(REVIEWS_URL), wanted_ids)
        print(f"Curated {len(reviews)} reviews across {len(wanted_ids)} products.")

        print("Inserting reviews...")
        insert_reviews(cursor, reviews)
        conn.commit()

        print("Done.")
    finally:
        cursor.close()
        conn.close()
        connector.close()


if __name__ == "__main__":
    main()