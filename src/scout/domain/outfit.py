"""
An outfit for an occasion, put together one piece at a time.

The Stylist agent plans it (OutfitPlan): the pieces to choose, in order, and a
few extras that would finish the look. Extras are only ever offered, once, at
the end: the shopper decides what goes into their outfit.

Outfit is the plan plus what the shopper has picked so far, kept in session
state while they choose.
"""

from typing import Optional

from pydantic import BaseModel, Field

from scout.domain.product import Product
from scout.domain.shopping_need import ShoppingNeed

MAX_PIECES = 4


class OutfitPiece(BaseModel):
    category: str = Field(description="One of the shop's categories")
    product_types: list[str] = Field(
        default_factory=list, max_length=3,
        description="Specific kinds from the shop's product types that suit it; empty when any kind will do")
    looks_like: str = Field(description="What to look for, in plain words a product description would use "
                                        "(colour, fit, material, style)")
    why: str = Field(description="One short reason it suits the occasion")


class OutfitPlan(BaseModel):
    occasion: str = Field(description="The occasion or event, in the shopper's words")
    pieces: list[OutfitPiece] = Field(
        default_factory=list, max_length=MAX_PIECES,
        description="The pieces to choose, most important first")
    extras: list[OutfitPiece] = Field(
        default_factory=list, max_length=2,
        description="Pieces that would finish the look; offered at the end, never required")


class Outfit(BaseModel):
    plan: OutfitPlan
    picked: list[Optional[Product]] = Field(default_factory=list)   # one per piece, None until chosen
    budget: Optional[float] = None                                 # for the whole outfit, if given
    introduced: bool = False                                       # Scout has shared the plan
    extras_offered: bool = False                                   # an extra is only ever offered once
    offered_extra: Optional[OutfitPiece] = None                    # the extra Scout is offering right now

    @classmethod
    def start(cls, plan: OutfitPlan, budget: Optional[float]) -> "Outfit":
        return cls(plan=plan, picked=[None] * len(plan.pieces), budget=budget)

    def current(self) -> Optional[int]:
        """Number of the piece being chosen now, or None when every piece is picked."""
        return next((i for i, product in enumerate(self.picked) if product is None), None)

    def chosen(self) -> list[Product]:
        return [product for product in self.picked if product is not None]

    def total(self) -> float:
        return sum(product.price or 0 for product in self.chosen())

    def pick(self, product: Product) -> bool:
        """Put the product in a piece still to choose that it belongs to (the current one first).
        False if it fits none, e.g. once every piece is picked: then it is an ordinary choice."""
        current = self.current()
        if current is None:
            return False
        open_pieces = [current] + [i for i, picked in enumerate(self.picked) if picked is None and i != current]
        for i in open_pieces:
            if self.plan.pieces[i].category.lower() == (product.category or "").lower():
                self.picked[i] = product
                return True
        return False

    def skip_current(self) -> None:
        index = self.current()
        if index is not None:
            del self.plan.pieces[index]
            del self.picked[index]

    def offer_extra(self) -> Optional[OutfitPiece]:
        """The extra to offer now that every piece is picked: the first one, and only the first time."""
        if not self.extras_offered and self.plan.extras:
            self.extras_offered = True
            self.offered_extra = self.plan.extras[0]
        return self.offered_extra

    def accept_extra(self) -> None:
        """The shopper accepted the offered extra: it becomes the next piece to choose."""
        if self.offered_extra:
            self.plan.pieces.append(self.offered_extra)
            self.picked.append(None)
            self.offered_extra = None

    def decline_extra(self) -> None:
        self.offered_extra = None

    def need_for_current(self, base: ShoppingNeed) -> Optional[ShoppingNeed]:
        """What to search for the piece being chosen: in the outfit's style, going with what is picked."""
        index = self.current()
        if index is None:
            return None
        piece = self.plan.pieces[index]
        goes_with = ", ".join(" ".join(p.title.split()[:8]) for p in self.chosen())   # titles carry colour, style
        budget_left = max(self.budget - self.total(), 0) if self.budget is not None else None
        return ShoppingNeed(
            category=piece.category,
            product_types=piece.product_types,
            gender=base.gender,
            looking_for=f"{piece.looks_like} to go with {goes_with}" if goes_with else piece.looks_like,
            purpose=self.plan.occasion,
            budget_max=budget_left,
            avoid=base.avoid,
        )

    def to_prompt_text(self) -> str:
        lines = [f"Occasion: {self.plan.occasion}"]
        for i, piece in enumerate(self.plan.pieces):
            product = self.picked[i]
            if product is None:
                lines.append(f"- {piece.category}: still to choose ({piece.looks_like}; {piece.why})")
            else:
                price = f"${product.price:.2f}" if product.price is not None else "price not listed"
                lines.append(f"- {piece.category}: picked {' '.join(product.title.split()[:10])} ({price})")
        if self.chosen():
            lines.append(f"Total so far: ${self.total():.2f}"
                         + (f" of a ${self.budget:.0f} budget" if self.budget is not None else ""))
        return "\n".join(lines)
