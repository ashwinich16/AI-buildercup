"""
Planning outfits for any occasion (a trip, a week, a day at the office), from the shopper's own wardrobe.

  WardrobeItem  a piece of clothing the shopper already owns (from a photo or their own words)
  CapsulePlan   the Stylist's plan: one look per day or occasion, built from owned pieces first, the owned
                pieces better left at home (and why), and the few new pieces worth buying (each one used in
                as many looks as possible)
  Capsule       the plan plus the shop's product chosen for each new piece, kept in session state

The plan is made in one go and shown as a board. Two ways to fill it, chosen by code:
  - they own pieces for it: the shop's best match for each new piece is chosen at once, so the whole
    picture is there straight away (they can swap any piece)
  - everything is new (guided): they choose each new piece themselves, one at a time, from a few options
"""

from typing import Optional

from pydantic import BaseModel, Field

from scout.domain.product import Product
from scout.domain.shopping_need import ShoppingNeed

MAX_LOOKS = 7
MAX_NEW_PIECES = 4
MAX_LEAVE_AT_HOME = 12


class WardrobeItem(BaseModel):
    id: str
    category: str                       # one of the shop's categories, so plans can mix owned and new pieces
    description: str                    # e.g. "navy slim chinos, cotton, smart casual"
    thumbnail: Optional[str] = None     # a small photo of it (data URL), when it was added from a photo


class NewPiece(BaseModel):
    key: str = Field(description="Short id for this piece, e.g. 'white-linen-shirt'")
    category: str = Field(description="One of the shop's categories")
    product_types: list[str] = Field(default_factory=list, max_length=3,
                                     description="Kinds from the shop's product types that suit it; may be empty")
    looks_like: str = Field(description="What to look for, in words a product description would use")
    why: str = Field(description="Why it is worth buying: what it adds and how many looks it is in")


class LeaveAtHome(BaseModel):
    owned: str = Field(description="Id of a WARDROBE item that does not suit this trip")
    why: str = Field(description="One short, kind reason, e.g. 'too warm for the heat'")


class LookPiece(BaseModel):
    owned: Optional[str] = Field(default=None, description="Id of a WARDROBE item, when they already own the piece")
    new: Optional[str] = Field(default=None, description="Key of a new piece, when it has to be bought")


class Look(BaseModel):
    title: str = Field(description="When it is worn, e.g. 'Day 1 · client meeting'")
    pieces: list[LookPiece] = Field(default_factory=list, max_length=6)
    why: str = Field(description="One short line on why it works for that moment")


class CapsulePlan(BaseModel):
    occasion: str = Field(description="The trip, week or event and what they will do there, in the shopper's words")
    looks: list[Look] = Field(default_factory=list, max_length=MAX_LOOKS)
    leave_at_home: list[LeaveAtHome] = Field(default_factory=list, max_length=MAX_LEAVE_AT_HOME)
    new_pieces: list[NewPiece] = Field(default_factory=list, max_length=MAX_NEW_PIECES)


class Capsule(BaseModel):
    plan: CapsulePlan
    wardrobe: list[WardrobeItem] = Field(default_factory=list)   # the owned pieces the plan uses
    picks: dict[str, Product] = Field(default_factory=dict)      # new piece key -> the product chosen for it
    shown: dict[str, list[str]] = Field(default_factory=dict)    # new piece key -> products already offered
    budget: Optional[float] = None                               # for all the new pieces together, if given
    guided: bool = False                                         # they choose each new piece, one at a time
    current: Optional[str] = None                                # guided: key of the piece being chosen now
    skipped: list[str] = Field(default_factory=list)             # guided: keys of pieces they did not want
    price_limits: dict[str, tuple[Optional[float], Optional[float]]] = Field(default_factory=dict)
    # new piece key -> (lowest, highest) price they gave for that piece alone

    def new_piece(self, key_or_category: str) -> Optional[NewPiece]:
        """The new piece the shopper means, by its key or its category."""
        wanted = key_or_category.lower()
        return next((p for p in self.plan.new_pieces if wanted in (p.key.lower(), p.category.lower())), None)

    def need_for(self, piece: NewPiece, base: ShoppingNeed, wish: str = "") -> ShoppingNeed:
        """What to search for one new piece, in the capsule's style: within the price they gave for it, else
        within its share of what is left of the budget (shared equally by the pieces not chosen yet, so all
        of them together stay within it)."""
        spent_elsewhere = sum(p.price or 0 for key, p in self.picks.items() if key != piece.key)
        open_pieces = sum(1 for p in self.plan.new_pieces if p.key == piece.key or p.key not in self.picks)
        share = max(self.budget - spent_elsewhere, 0) / max(open_pieces, 1) if self.budget is not None else None
        lowest, highest = self.price_limits.get(piece.key, (None, None))
        if lowest is not None or highest is not None:
            share = highest
        return ShoppingNeed(
            category=piece.category,
            product_types=piece.product_types,
            gender=base.gender,
            looking_for=f"{piece.looks_like} {wish}".strip(),
            purpose=self.plan.occasion,
            budget_min=lowest,
            budget_max=round(share, 2) if share is not None else None,
            avoid=base.avoid,
        )

    def choose(self, key: str, product: Product) -> None:
        self.picks[key] = product
        self.offer(key, [product.id])

    def offer(self, key: str, product_ids: list[str]) -> None:
        """Remember products offered for a piece, so it is never offered the same ones again."""
        self.shown[key] = list(dict.fromkeys([*self.shown.get(key, []), *product_ids]))

    def add_piece(self, category: str, product_types: list[str], looks_like: str) -> NewPiece:
        """A piece they asked to add to the outfit (worn in every look), to choose now."""
        keys = {p.key for p in self.plan.new_pieces}
        key = next(k for k in (f"{category}-{n}".lower().replace(" ", "-") for n in range(1, 100)) if k not in keys)
        piece = NewPiece(key=key, category=category, product_types=product_types[:3],
                         looks_like=looks_like or category, why="they asked to add it to the outfit")
        self.plan.new_pieces.append(piece)
        for look in self.plan.looks:
            look.pieces.append(LookPiece(new=key))
        self.current = key
        return piece

    def keep_choices_from(self, earlier: "Capsule") -> None:
        """Planned again: pieces already chosen stay chosen, matched by category."""
        chosen = {earlier.new_piece(key).category.lower(): product for key, product in earlier.picks.items()
                  if earlier.new_piece(key)}
        for piece in self.plan.new_pieces:
            if piece.category.lower() in chosen:
                self.choose(piece.key, chosen.pop(piece.category.lower()))

    def skip(self, key: str) -> None:
        self.skipped = list(dict.fromkeys([*self.skipped, key]))

    def move_on(self) -> Optional[NewPiece]:
        """Guided: the next piece still to choose (now `current`), or None when every piece is settled."""
        piece = next((p for p in self.plan.new_pieces if p.key not in self.picks and p.key not in self.skipped), None)
        self.current = piece.key if piece else None
        return piece

    def products_to_buy(self) -> list[Product]:
        return [self.picks[p.key] for p in self.plan.new_pieces if p.key in self.picks]

    def look_products(self, look: Look) -> list[Product]:
        return [self.picks[piece.new] for piece in look.pieces if piece.new in self.picks]

    def to_prompt_text(self) -> str:
        owned = {item.id: item for item in self.wardrobe}
        lines = [f"Occasion: {self.plan.occasion}", "LOOKS"]
        for look in self.plan.looks:
            parts = []
            for piece in look.pieces:
                if piece.owned in owned:
                    parts.append(f"their own {owned[piece.owned].description}")
                elif piece.new in self.picks:
                    parts.append(f"NEW {' '.join(self.picks[piece.new].title.split()[:8])}")
                elif (new := self.new_piece(piece.new or "")) and new.key not in self.skipped:
                    parts.append(f"{new.category} still to choose")
            lines.append(f"- {look.title}: {'; '.join(parts)} ({look.why})")
        if self.plan.leave_at_home:
            lines.append("THEIR OWN PIECES BETTER LEFT AT HOME")
            lines += [f"- {owned[left.owned].description}: {left.why}"
                      for left in self.plan.leave_at_home if left.owned in owned]
        lines.append("NEW PIECES TO BUY")
        for piece in self.plan.new_pieces:
            product = self.picks.get(piece.key)
            if product is None and piece.key in self.skipped:
                lines.append(f"- {piece.category}: they chose to skip it")
            elif product is None and self.guided:
                now = " (choosing now)" if piece.key == self.current else ""
                lines.append(f"- {piece.category}: still to choose{now} ({piece.looks_like}; {piece.why})")
            elif product is None:
                lines.append(f"- {piece.category}: nothing suitable found in the shop ({piece.why})")
            else:
                price = f"${product.price:.2f}" if product.price is not None else "price not listed"
                lines.append(f"- {piece.category}: {' '.join(product.title.split()[:10])}, {price} ({piece.why})")
        total = sum(p.price or 0 for p in self.products_to_buy())
        lines.append(f"New pieces total: ${total:.2f}" + (f" of a ${self.budget:.0f} budget" if self.budget else ""))
        return "\n".join(lines)
