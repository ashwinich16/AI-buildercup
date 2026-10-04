"""
Try Retrieval against the real catalog with a few sample needs.

From the repository root:
    PYTHONPATH=src python scripts/check_retrieval.py
"""

from scout.catalog.vertex_search import VertexSearchCatalog
from scout.config import get_settings
from scout.domain import ShoppingNeed
from scout.tools import ProductRetriever

SAMPLE_NEEDS = [
    ShoppingNeed(category="top", product_types=["shirt"], gender="men", looking_for="relaxed but smart shirt",
                 purpose="beach wedding", budget_max=50),
    ShoppingNeed(category="top", product_types=["shirt"], gender="women", looking_for="casual shirt"),
    ShoppingNeed(category="jeans", gender="men", looking_for="slim fit jeans"),
    ShoppingNeed(category="bag", looking_for="leather tote", purpose="work", avoid=["black"]),
    ShoppingNeed(category="shoe", looking_for="comfortable walking shoes", budget_max=5),   # forces relaxing
]


def main() -> None:
    settings = get_settings()
    retriever = ProductRetriever(VertexSearchCatalog(settings), limit=settings.search_result_limit)

    for need in SAMPLE_NEEDS:
        result = retriever.find(need)
        print(f"\nNEED: {need.summary()}")
        print(f"  query: {result.query_text!r}   relaxed: {result.relaxed or 'nothing'}"
              f"   other kinds offered: {result.other_types or 'none'}")
        for product in result.products[:3]:
            price = f"${product.price:.2f}" if product.price is not None else "price unknown"
            print(f"  - [{product.gender}, {product.product_type}] {product.title[:60]}  ({price})")


if __name__ == "__main__":
    main()
