"""
Retrieval: turns a ShoppingNeed into a shortlist of products.

Plain Python, no LLM: the Orchestrator calls it directly, which is faster and
more predictable than letting a model decide how to search.

Exact matches come first. If there are only a few, it tops the list up with
related products from broader searches (details relaxed, budget kept), and
records which products are exact and what was relaxed for the rest, so Scout
can be honest ("I only have one glittery pair, but these flats would also
suit a wedding"). The budget is only relaxed when nothing at all fits.

It depends only on the CatalogRepository interface, so any catalog works.
"""

from typing import Optional

from pydantic import BaseModel, Field

from scout.catalog import CatalogRepository, SearchQuery
from scout.domain import Product, ShoppingNeed


class RetrievalResult(BaseModel):
    products: list[Product] = Field(default_factory=list)    # exact matches first, then related ones
    exact_ids: list[str] = Field(default_factory=list)       # products that match everything asked for
    relaxed: list[str] = Field(default_factory=list)         # what was relaxed for the related ones, e.g. ["details"]
    query_text: str = ""

    @property
    def found(self) -> bool:
        return bool(self.products)


class ProductRetriever:
    def __init__(self, catalog: CatalogRepository, limit: int = 8, target: int = 6):
        self._catalog = catalog
        self._limit = limit      # products per search call
        self._target = target    # stop topping up once the shortlist has this many

    def find(self, need: ShoppingNeed, exclude_ids: Optional[set[str]] = None) -> RetrievalResult:
        """Exact matches first, topped up with related products. `exclude_ids` skips products already shown."""
        excluded = set(exclude_ids or ())
        collected: list[Product] = []
        exact_ids: list[str] = []
        relaxed_for_rest: list[str] = []

        for relaxed, query in self._attempts(need):
            if collected and "budget" in relaxed:
                break                          # never trade the budget away just to top up
            seen = excluded | {p.id for p in collected}
            hits = [p for p in self._without_avoided(self._catalog.search(query), need.avoid)
                    if p.id not in seen]
            if not hits:
                continue
            if relaxed:
                relaxed_for_rest = relaxed_for_rest or relaxed
            else:
                exact_ids += [p.id for p in hits]
            collected += hits
            if len(collected) >= self._target:
                break

        if not collected:
            return RetrievalResult(relaxed=["everything"], query_text=need.search_text())
        return RetrievalResult(products=collected[: self._limit], exact_ids=exact_ids,
                               relaxed=relaxed_for_rest, query_text=need.search_text())

    def _attempts(self, need: ShoppingNeed) -> list[tuple[list[str], SearchQuery]]:
        """Search plans from strictest to loosest.

        Budget is treated as a firm limit and details as nice-to-have, so details
        are relaxed first and the budget only as a last resort.
        """
        full_text = need.search_text()
        short_text = need.looking_for or need.category or full_text
        broad_text = " ".join(p for p in [need.category or short_text, need.purpose or ""] if p)
        category_only = need.category or short_text
        budget = need.budget_max

        plans = [
            ([], full_text, budget),                        # everything the shopper said
            (["some details"], short_text, budget),         # just what they want, within budget
            (["details"], broad_text, budget),              # the category for the same purpose, within budget
            (["details"], category_only, budget),           # anything in the category, within budget
            (["budget"], full_text, None),                  # full need, any price
            (["budget", "details"], category_only, None),   # anything in the category
        ]

        attempts, seen = [], set()
        for relaxed, text, max_price in plans:
            key = (text, max_price)
            if text and key not in seen:                    # skip duplicate searches
                seen.add(key)
                attempts.append((relaxed, SearchQuery(text=text, category=need.category,
                                                      max_price=max_price, limit=self._limit)))
        return attempts

    @staticmethod
    def _without_avoided(products: list[Product], avoid: list[str]) -> list[Product]:
        """Drop products whose stated facts mention something the shopper wants to avoid.

        Checks attributes and style tags only (not titles), so "avoid heels" does not
        remove a shoe whose title mentions "heel pain relief". If everything would be
        removed, the original list is kept and the narrator can mention the conflict.
        """
        if not avoid:
            return products
        terms = [term.lower() for term in avoid]

        def mentions_avoided(product: Product) -> bool:
            facts = " ".join([a.value for a in product.attributes] + product.style_tags).lower()
            return any(term in facts for term in terms)

        kept = [p for p in products if not mentions_avoided(p)]
        return kept or products


def find_products(retriever: ProductRetriever, need: Optional[dict]) -> RetrievalResult:
    """Convenience for callers holding the need as a plain dict (e.g. from session state)."""
    return retriever.find(ShoppingNeed.model_validate(need or {}))
