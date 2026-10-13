"""
Wishlist: products the shopper saved to come back to, compare or buy later.

Scout presents three options at a time, so an option heard a few searches ago is gone from the screen. Saving
it keeps it: the wishlist can be shown again at any time, its products compared or added to the cart. Like the
cart, ScoutChatService saves it with the ShopperRepository (Cloud SQL), so it follows the shopper across
conversations. Only short facts are kept per item; the full product comes from the catalog when it is shown.
"""

from typing import Optional

from pydantic import BaseModel, Field

from scout.domain import Product


class WishlistItem(BaseModel):
    product_id: str
    name: str
    price: Optional[float] = None
    kind: Optional[str] = None          # e.g. "loafers", so Scout can say it in a few words
    image_url: Optional[str] = None


class Wishlist(BaseModel):
    items: list[WishlistItem] = Field(default_factory=list)

    def add(self, product: Product) -> bool:
        """False if it was already saved."""
        if self.has(product.id):
            return False
        self.items.append(WishlistItem(product_id=product.id, name=" ".join(product.title.split()[:10]),
                                       price=product.price, kind=product.kind or None,
                                       image_url=product.image_urls[0] if product.image_urls else None))
        return True

    def has(self, product_id: str) -> bool:
        return any(item.product_id == product_id for item in self.items)

    def ids(self, positions: Optional[list[int]] = None) -> list[str]:
        """The product ids at these 1-based positions (all of them when none are given)."""
        if not positions:
            return [item.product_id for item in self.items]
        return list(dict.fromkeys(self.items[n - 1].product_id for n in positions if 1 <= n <= len(self.items)))

    def remove(self, product_ids: list[str]) -> list[WishlistItem]:
        """Take these products out; returns what was removed."""
        removed = [item for item in self.items if item.product_id in product_ids]
        self.items = [item for item in self.items if item.product_id not in product_ids]
        return removed

    @staticmethod
    def short_name(item: WishlistItem) -> str:
        return item.kind or " ".join(item.name.split()[:6])

    def spoken(self) -> str:
        """Every item in a few spoken words, numbered: "1, the loafers at $40.00; 2, the blazer at $85.00"."""
        def one(n: int, item: WishlistItem) -> str:
            price = f" at ${item.price:.2f}" if item.price is not None else ""
            return f"{n}, the {self.short_name(item)}{price}"
        return "; ".join(one(n, item) for n, item in enumerate(self.items, start=1))

    def to_prompt_text(self) -> str:
        if not self.items:
            return "The wishlist is empty."
        return "\n".join(f"{n}. {item.kind + ': ' if item.kind else ''}{item.name} "
                         f"({f'${item.price:.2f}' if item.price is not None else 'price not listed'})"
                         for n, item in enumerate(self.items, start=1))
