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
    image_url: Optional[str] = None     # a small photo for the cart list
    size: Optional[str] = None          # as shown and said, e.g. "UK 9", "32 waist", "M"; None if it needs none


class Cart(BaseModel):
    items: list[CartItem] = Field(default_factory=list)

    def add(self, product: Product, size: Optional[str] = None) -> bool:
        """False if it was already in the cart."""
        if any(item.product_id == product.id for item in self.items):
            return False
        self.items.append(CartItem(product_id=product.id, name=" ".join(product.title.split()[:10]),
                                   price=product.price, kind=product.kind or None,
                                   image_url=product.image_urls[0] if product.image_urls else None, size=size))
        return True

    def set_size(self, product_id: str, size: str) -> bool:
        """Change the size of an item in the cart; False if it is not there."""
        item = next((i for i in self.items if i.product_id == product_id), None)
        if item is not None:
            item.size = size
        return item is not None

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

    def spoken(self) -> str:
        """The cart in a few spoken words: each item by its kind (or a short name), size and price, then the total."""
        def one(item: CartItem) -> str:
            name = (item.kind or " ".join(item.name.split()[:6])) + (f", size {item.size}," if item.size else "")
            return f"the {name} at ${item.price:.2f}" if item.price is not None else f"the {name}"
        names = [one(item) for item in self.items]
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        return f"{listed}, ${self.total:.2f} in total" if len(names) > 1 else listed

    def to_prompt_text(self) -> str:
        if not self.items:
            return "The cart is empty."
        lines = [f"{n}. {item.kind + ': ' if item.kind else ''}{item.name}{', size ' + item.size if item.size else ''} "
                 f"({f'${item.price:.2f}' if item.price is not None else 'price not listed'})"
                 for n, item in enumerate(self.items, start=1)]
        note = " (some prices not listed)" if self.has_unpriced else ""
        lines.append(f"Total: ${self.total:.2f}{note}")
        return "\n".join(lines)
