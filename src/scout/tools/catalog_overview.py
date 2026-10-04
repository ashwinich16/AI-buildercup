"""
CatalogOverview: a short summary of what the catalog actually has that is
close to what the shopper is asking for.

The Elicitation agent uses it so its questions always lead to products that
exist ("soft ballet flats or cushioned loafers?" only if both are in stock),
and to guess the category from the shopper's own words. It also shows who the
matches are made for and which kinds they are, so Scout knows when "men's or
women's?" or "a shirt or a t-shirt?" would really change what it shows.

Scales with catalog size: it summarises the top-N search results (an index
lookup, roughly constant time), never the whole catalog. For very large
catalogs this class can be swapped for one using Vertex AI Search facets,
behind the same `build()` call.
"""

from collections import Counter
from typing import Optional

from pydantic import BaseModel, Field

from scout.catalog import CatalogRepository, SearchQuery
from scout.domain import Product

_MAX_VALUE_CHARS = 40   # skip long sentence-style values; they do not summarise well


class CatalogOverview(BaseModel):
    sample_size: int = 0
    categories: dict[str, int] = Field(default_factory=dict)          # category -> count in sample
    kinds: dict[str, dict[str, int]] = Field(default_factory=dict)    # category -> {product type -> count}
    genders: dict[str, int] = Field(default_factory=dict)             # men / women / unisex -> count
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    common_attributes: dict[str, list[str]] = Field(default_factory=dict)  # e.g. color -> [black, tan]
    common_styles: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)                 # short product names

    def to_prompt_text(self) -> str:
        if not self.sample_size:
            return "No close matches found in the catalog."
        lines = [f"Closest {self.sample_size} products in the catalog:"]
        lines.append("Categories: " + ", ".join(f"{c} ({n})" for c, n in self.categories.items()))
        for category, kinds in self.kinds.items():
            lines.append(f"Kinds of {category}: " + ", ".join(f"{k} ({n})" for k, n in kinds.items()))
        if self.genders:
            lines.append("Made for: " + ", ".join(f"{g} ({n})" for g, n in self.genders.items()))
        if self.price_min is not None:
            lines.append(f"Prices: ${self.price_min:.0f} to ${self.price_max:.0f}")
        for name, values in self.common_attributes.items():
            lines.append(f"{name}: {', '.join(values)}")
        if self.common_styles:
            lines.append("Styles: " + ", ".join(self.common_styles))
        if self.examples:
            lines.append("Examples: " + "; ".join(self.examples))
        return "\n".join(lines)


class CatalogOverviewBuilder:
    def __init__(self, catalog: CatalogRepository, sample_size: int = 20,
                 values_per_attribute: int = 4, max_attributes: int = 6, max_examples: int = 5):
        self._catalog = catalog
        self._sample_size = sample_size
        self._values_per_attribute = values_per_attribute
        self._max_attributes = max_attributes
        self._max_examples = max_examples

    def build(self, text: str, category: Optional[str] = None, gender: Optional[str] = None) -> CatalogOverview:
        products = self._catalog.search(SearchQuery(text=text, category=category, gender=gender,
                                                    limit=self._sample_size))
        if not products:
            return CatalogOverview()

        prices = [p.price for p in products if p.price is not None]
        return CatalogOverview(
            sample_size=len(products),
            categories=dict(Counter(p.category for p in products if p.category).most_common()),
            kinds=self._kinds(products),
            genders=dict(Counter(p.gender for p in products if p.gender).most_common()),
            price_min=min(prices) if prices else None,
            price_max=max(prices) if prices else None,
            common_attributes=self._common_attributes(products),
            common_styles=self._top([t.lower() for p in products for t in p.style_tags], 8),
            examples=[_short_name(p) for p in products[: self._max_examples]],
        )

    def _common_attributes(self, products: list[Product]) -> dict[str, list[str]]:
        values_by_name: dict[str, list[str]] = {}
        for product in products:
            for attribute in product.attributes:
                value = attribute.value.strip().lower()
                if value and len(value) <= _MAX_VALUE_CHARS:
                    values_by_name.setdefault(attribute.name.strip().lower(), []).append(value)

        # Attributes that many products share are the useful ones to ask about.
        names = sorted(values_by_name, key=lambda n: len(values_by_name[n]), reverse=True)
        return {name: self._top(values_by_name[name], self._values_per_attribute)
                for name in names[: self._max_attributes] if len(values_by_name[name]) >= 2}

    @staticmethod
    def _kinds(products: list[Product]) -> dict[str, dict[str, int]]:
        by_category: dict[str, Counter] = {}
        for product in products:
            if product.category and product.product_type:
                by_category.setdefault(product.category, Counter())[product.product_type] += 1
        return {category: dict(counts.most_common()) for category, counts in by_category.items()}

    @staticmethod
    def _top(values: list[str], n: int) -> list[str]:
        return [value for value, _ in Counter(values).most_common(n)]


def _short_name(product: Product, max_words: int = 8) -> str:
    words = product.title.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")
