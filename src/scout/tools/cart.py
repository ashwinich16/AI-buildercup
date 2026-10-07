"""
Cart: the products the shopper has chosen.

Plain data plus simple operations. The Orchestrator changes it in session state;
ScoutChatService saves it with the ShopperRepository (Cloud SQL), so it follows
the shopper across conversations. Nothing is written to local disk.
"""

from typing import Optional

from pydantic import BaseModel, Field

from scout.domain import Product


class CartItem(BaseModel):
    product_id: str
    name: str
    price: Optional[float] = None
    kind: Optional[str] = None          # e.g. "jeans": long titles are cut, and the kind must not be guessed


class Cart(BaseModel):
    items: list[CartItem] = Field(default_factory=list)

    def add(self, product: Product) -> bool:
        """False if it was already in the cart."""
        if any(item.product_id == product.id for item in self.items):
            return False
        self.items.append(CartItem(product_id=product.id, name=" ".join(product.title.split()[:10]),
                                   price=product.price, kind=product.kind or None))
        return True

    def remove(self, positions: list[int]) -> list[CartItem]:
        """Take out the items at these 1-based positions; returns what was removed."""
        wanted = {p for p in positions if 1 <= p <= len(self.items)}
        removed = [item for n, item in enumerate(self.items, start=1) if n in wanted]
        self.items = [item for n, item in enumerate(self.items, start=1) if n not in wanted]
        return removed

    def clear(self) -> list[CartItem]:
        removed, self.items = self.items, []
        return removed

    @property
    def total(self) -> float:
        return round(sum(item.price or 0.0 for item in self.items), 2)

    @property
    def has_unpriced(self) -> bool:
        return any(item.price is None for item in self.items)

    def to_prompt_text(self) -> str:
        if not self.items:
            return "The cart is empty."
        lines = [f"{n}. {item.kind + ': ' if item.kind else ''}{item.name} "
                 f"({f'${item.price:.2f}' if item.price is not None else 'price not listed'})"
                 for n, item in enumerate(self.items, start=1)]
        note = " (some prices not listed)" if self.has_unpriced else ""
        lines.append(f"Total: ${self.total:.2f}{note}")
        return "\n".join(lines)
