"""Print what's in the Cloud SQL catalog: products, images, reviews and enrichment per category.

  python scripts/check_catalog.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import ingestion  # noqa: E402  (reuses its Cloud SQL connection settings)

SQL = """
SELECT p.category,
       count(*)                                                       AS products,
       (SELECT count(*) FROM product_image i JOIN product q USING (product_id)
         WHERE q.category IS NOT DISTINCT FROM p.category)            AS images,
       (SELECT count(*) FROM review r JOIN product q USING (product_id)
         WHERE q.category IS NOT DISTINCT FROM p.category)            AS reviews,
       count(pi.product_id)                                           AS enriched
FROM product p
LEFT JOIN product_intelligence pi USING (product_id)
GROUP BY p.category
ORDER BY p.category;
"""


def main():
    conn, connector = ingestion.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(SQL)
        print(f"{'category':<10} {'products':>8} {'images':>7} {'reviews':>8} {'enriched':>9}")
        for category, products, images, reviews, enriched in cursor.fetchall():
            print(f"{category or '-':<10} {products:>8} {images:>7} {reviews:>8} {enriched:>9}")
    finally:
        conn.close()
        connector.close()


if __name__ == "__main__":
    main()
