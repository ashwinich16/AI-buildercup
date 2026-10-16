"""
Planning outfits for an occasion: looks made of the shopper's own clothes and new pieces from the shop.

  WardrobeItem  a piece of clothing the shopper owns (from a photo or their own words)
  OutfitPlan    the Stylist's plan: the looks, each piece either one they own or a new one described in words,
                and their own pieces that do not suit the occasion
  Outfit        the plan on the board. Every look has its own pieces, so changing, removing or adding a piece
                in one look never touches another look. A new piece holds the product found for it.

Changing a piece takes two turns: Scout offers a few products (`changing` says for which piece and looks),
then the one they choose goes into those looks. Adding a kind of piece a look does not have yet is the same
step: the chosen product goes in as a new piece.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from scout.domain.product import Product
from scout.domain.shopping_need import ShoppingNeed

MAX_LOOKS = 7      # a week of looks: the most one plan makes


class WardrobeItem(BaseModel):
    id: str
    category: str                       # one of the shop's categories, so plans can mix owned and new pieces
    description: str                    # e.g. "navy slim chinos, cotton, smart casual"
    thumbnail: Optional[str] = None     # a small photo of it (data URL), when it was added from a photo


# ---------------------------------------------------------------------
# The Stylist's plan (its output schema)
# ---------------------------------------------------------------------

class PlannedPiece(BaseModel):
    owned: Optional[str] = Field(default=None, description="Id of a WARDROBE item, when they already own the piece")
    category: str = Field(default="", description="For a new piece: one of the shop's categories")
    product_types: list[str] = Field(default_factory=list,
                                     description="For a new piece: kinds from the shop's product types that suit "
                                                 "it; may be empty")
    looks_like: str = Field(default="", description="For a new piece: what to look for, in words a product "
                                                    "description would use (colour, fit, material, how formal)")


class PlannedLook(BaseModel):
    title: str = Field(description="When it is worn, e.g. 'Day 1 · client meeting'")
    pieces: list[PlannedPiece] = Field(default_factory=list)
    why: str = Field(default="", description="One short line on why it works for that moment")


class LeaveAtHome(BaseModel):
    owned: str = Field(description="Id of a WARDROBE item that does not suit this occasion")
    why: str = Field(description="One short, kind reason, e.g. 'too warm for the heat'")


class OutfitPlan(BaseModel):
    occasion: str = Field(description="The occasion and what they will do there, in the shopper's words")
    looks: list[PlannedLook] = Field(default_factory=list)
    leave_at_home: list[LeaveAtHome] = Field(default_factory=list)


# ---------------------------------------------------------------------
# The outfit on the board
# ---------------------------------------------------------------------

class Piece(BaseModel):
    """One piece of one look: one they own, or a new one (with the product found for it, if any)."""
    category: str
    owned: Optional[str] = None             # wardrobe id, when it is theirs
    product_types: list[str] = Field(default_factory=list)
    looks_like: str = ""                    # what to look for (new pieces)
    product: Optional[Product] = None       # the product chosen for a new piece; None if none was found

    def is_new(self) -> bool:
        return self.owned is None


class Look(BaseModel):
    title: str
    why: str = ""
    pieces: list[Piece] = Field(default_factory=list)

    def index_of(self, category: str) -> Optional[int]:
        """Where this look's piece of the category is, or None if it has none."""
        return next((n for n, piece in enumerate(self.pieces) if _same(piece.category, category)), None)


class Change(BaseModel):
    """A piece being changed (or added) in some looks: options are on screen, the chosen one goes into them."""
    action: Literal["change", "remove"] = "change"
    category: str
    looks: list[int] = Field(default_factory=list)   # look indexes (0 = the first); empty: waiting to hear which
    wish: str = ""                                   # what they want instead, in their words
    price: tuple[Optional[float], Optional[float]] = (None, None)   # a price they gave for this piece alone
    shown: list[str] = Field(default_factory=list)   # products already offered, never offered again
    replaces: Optional[str] = None                   # the category it takes the place of (jeans for the trousers)

    def looks_text(self) -> str:
        return " and ".join(f"look {n + 1}" for n in self.looks)


class Outfit(BaseModel):
    occasion: str
    looks: list[Look] = Field(default_factory=list)
    leave_at_home: list[LeaveAtHome] = Field(default_factory=list)
    wardrobe: list[WardrobeItem] = Field(default_factory=list)   # their own pieces (words only, no photos)
    budget: Optional[float] = None                               # for all the new pieces together, if given
    changing: Optional[Change] = None
    fit_notes: dict[str, str] = Field(default_factory=dict)      # product id -> why it suits the occasion

    @classmethod
    def from_plan(cls, plan: OutfitPlan, wardrobe: list[WardrobeItem], budget: Optional[float]) -> "Outfit":
        """The board for a plan: owned pieces that are really in the wardrobe, new pieces with a category."""
        owned = {item.id: item for item in wardrobe}
        looks = []
        for planned in plan.looks[:MAX_LOOKS]:
            pieces = []
            for piece in planned.pieces:
                if piece.owned:
                    if piece.owned in owned:
                        pieces.append(Piece(category=owned[piece.owned].category, owned=piece.owned))
                elif piece.category:
                    pieces.append(Piece(category=piece.category, product_types=piece.product_types[:3],
                                        looks_like=piece.looks_like or piece.category))
            if pieces:
                looks.append(Look(title=planned.title, why=planned.why, pieces=pieces))
        used = {piece.owned for look in looks for piece in look.pieces if piece.owned}
        left = [left for left in plan.leave_at_home if left.owned in owned and left.owned not in used]
        return cls(occasion=plan.occasion, looks=looks, leave_at_home=left,
                   wardrobe=[item.model_copy(update={"thumbnail": None}) for item in wardrobe], budget=budget)

    # --- reading it ---

    def owned_item(self, item_id: Optional[str]) -> Optional[WardrobeItem]:
        return next((item for item in self.wardrobe if item.id == item_id), None)

    def looks_with(self, category: str) -> list[int]:
        """Indexes of the looks that have a piece of this category."""
        return [n for n, look in enumerate(self.looks) if any(_same(p.category, category) for p in look.pieces)]

    def categories(self) -> list[str]:
        return list(dict.fromkeys(p.category for look in self.looks for p in look.pieces))

    def unfound(self) -> list[Piece]:
        """New pieces that have no product yet."""
        return [p for look in self.looks for p in look.pieces if p.is_new() and p.product is None]

    def products_to_buy(self) -> list[Product]:
        """Every product in the looks, once each (a product worn in two looks is bought once)."""
        found = {}
        for look in self.looks:
            for piece in look.pieces:
                if piece.product is not None:
                    found.setdefault(piece.product.id, piece.product)
        return list(found.values())

    def look_products(self, number: int) -> list[Product]:
        return [p.product for p in self.looks[number].pieces if p.product is not None]

    def total(self) -> float:
        return round(sum(p.price or 0 for p in self.products_to_buy()), 2)

    # --- changing it ---

    def put(self, product: Product, change: Change) -> None:
        """The chosen product goes into the looks being changed: it takes the place of that look's piece of this
        category (theirs or new), or is added to a look that has none."""
        for number in change.looks:
            look = self.looks[number]
            n = look.index_of(change.category)
            if n is None and change.replaces:
                n = look.index_of(change.replaces)
                if n is not None:                    # jeans in place of the trousers: a new piece of its own kind
                    look.pieces[n] = Piece(category=change.category, looks_like=change.wish, product=product)
                    continue
            if n is None:
                look.pieces.append(Piece(category=change.category, looks_like=change.wish, product=product))
            else:
                old = look.pieces[n]
                look.pieces[n] = Piece(category=old.category, product_types=old.product_types,
                                       looks_like=change.wish or old.looks_like, product=product)

    def remove(self, category: str, looks: list[int]) -> None:
        """Take that look's piece of this category out of each look."""
        for number in looks:
            n = self.looks[number].index_of(category)
            if n is not None:
                self.looks[number].pieces.pop(n)

    # --- searching for it ---

    def need_for(self, change: Change, base: ShoppingNeed) -> ShoppingNeed:
        """What to search for to change (or add) a piece: what they wished for, else what the piece was meant to
        be; within the price they gave for it, else what is left of the budget once the other pieces are paid."""
        current = [look.pieces[i] for look in (self.looks[n] for n in change.looks)
                   if (i := look.index_of(change.category)) is not None]
        new = next((p for p in current if p.is_new()), None)
        owned = next((self.owned_item(p.owned) for p in current if p.owned), None)
        looks_like = change.wish or (new.looks_like if new else "") or (owned.description if owned else "")
        lowest, highest = change.price
        if highest is None and lowest is None and self.budget is not None:
            replaced = {p.product.id for p in current if p.product is not None}
            others = sum(p.price or 0 for p in self.products_to_buy() if p.id not in replaced)
            highest = round(max(self.budget - others, 0), 2)
        return self._need(base, change.category, new.product_types if new else [], looks_like, lowest, highest)

    def need_for_piece(self, piece: Piece, base: ShoppingNeed, share: Optional[float]) -> ShoppingNeed:
        return self._need(base, piece.category, piece.product_types, piece.looks_like, None, share)

    def _need(self, base: ShoppingNeed, category: str, product_types: list[str], looks_like: str,
              lowest: Optional[float], highest: Optional[float]) -> ShoppingNeed:
        return ShoppingNeed(category=category, product_types=product_types, gender=base.gender,
                            looking_for=looks_like or category, purpose=self.occasion,
                            budget_min=lowest, budget_max=highest, avoid=base.avoid)

    # --- for the agents ---

    def to_prompt_text(self) -> str:
        lines = [f"Occasion: {self.occasion}", "LOOKS ([category] then the piece)"]
        for number, look in enumerate(self.looks, start=1):
            parts = [f"[{piece.category}] {self._piece_text(piece)}" for piece in look.pieces]
            lines.append(f"Look {number} ({look.title}): {'; '.join(parts) or 'no pieces'}"
                         + (f" ({look.why})" if look.why else ""))
        if self.leave_at_home:
            lines.append("THEIR OWN PIECES NOT USED (they do not suit it)")
            lines += [f"- {item.description}: {left.why}"
                      for left in self.leave_at_home if (item := self.owned_item(left.owned))]
        lines.append(f"New pieces to buy: {len(self.products_to_buy())}, total ${self.total():.2f}"
                     + (f" of a ${self.budget:.0f} budget" if self.budget else ""))
        if self.changing and self.changing.looks:
            lines.append(f"Choosing now: a {self.changing.category} for {self.changing.looks_text()}")
        return "\n".join(lines)

    def _piece_text(self, piece: Piece) -> str:
        if piece.owned:
            item = self.owned_item(piece.owned)
            return f"their own {item.description if item else piece.category}"
        if piece.product is None:
            return f"new {piece.looks_like}: nothing suitable found in the shop"
        price = f"${piece.product.price:.2f}" if piece.product.price is not None else "price not listed"
        suits = f"; for the occasion: {self.fit_notes[piece.product.id]}" if piece.product.id in self.fit_notes else ""
        return f"NEW {' '.join(piece.product.title.split()[:8])}, {price}{suits}"


def _same(category: str, other: str) -> bool:
    return category.strip().lower() == other.strip().lower()
