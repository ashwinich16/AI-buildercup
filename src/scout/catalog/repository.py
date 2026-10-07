"""
The catalog contract that agents and tools depend on.

Agents only know `CatalogRepository` and `SearchQuery`. Which backend answers
(Vertex AI Search today, something else later) is decided when the app is
wired together, never inside an agent (Dependency Inversion).
"""

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field

from scout.domain import Product


class SearchQuery(BaseModel):
    """What to look for. Every constraint is optional."""

    text: str = ""                           # natural language, e.g. "breathable shirt for a beach wedding"
    category: Optional[str] = None
    product_types: list[str] = Field(default_factory=list)   # any of these kinds, e.g. ["shirt", "polo"]
    gender: Optional[str] = None             # "men" or "women": their items plus unisex ones
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    limit: int = Field(default=8, ge=1, le=50)

    def genders(self) -> list[str]:
        """Product genders that suit this query: a women's search also fits unisex items."""
        return [self.gender, "unisex"] if self.gender in ("men", "women") else []

    def matches(self, product: Product) -> bool:
        """Check the structured constraints locally (used when the backend cannot filter)."""
        if self.category and (product.category or "").lower() != self.category.lower():
            return False
        if self.product_types and (product.product_type or "").lower() not in {t.lower() for t in self.product_types}:
            return False
        if self.genders() and (product.gender or "") not in self.genders():
            return False
        if self.min_price is not None and (product.price is None or product.price < self.min_price):
            return False
        if self.max_price is not None and (product.price is None or product.price > self.max_price):
            return False
        if self.min_rating is not None and (product.rating is None or product.rating < self.min_rating):
            return False
        return True


class CatalogRepository(ABC):
    """Read-only access to the product catalog."""

    @abstractmethod
    def search(self, query: SearchQuery) -> list[Product]:
        """Best-matching products for the query, most relevant first."""

    @abstractmethod
    def get_product(self, product_id: str) -> Optional[Product]:
        """One product by id, or None if it does not exist (e.g. a saved cart item)."""

    @abstractmethod
    def categories(self) -> list[str]:
        """Categories present in the catalog, so agents never hard-code them."""

    @abstractmethod
    def product_types(self) -> list[str]:
        """Kinds of product present in the catalog (e.g. shirt, polo, sneaker), for the same reason."""
