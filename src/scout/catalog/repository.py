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
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    limit: int = Field(default=8, ge=1, le=50)

    def matches(self, product: Product) -> bool:
        """Check the structured constraints locally (used when the backend cannot filter)."""
        if self.category and (product.category or "").lower() != self.category.lower():
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
        """One product by id, or None if it does not exist."""

    @abstractmethod
    def categories(self) -> list[str]:
        """Categories present in the catalog, so agents never hard-code them."""
