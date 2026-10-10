"""
ShoppingNeed: what the shopper is looking for, built up over the conversation.

Category-agnostic on purpose: "purpose", "must_have" and "avoid" hold free
text, so "size 9, wide fit" and "16 GB RAM" fit equally well. The
Elicitation agent fills it; the Retrieval tool turns it into a catalog query.

`gender` is about the ITEM (men's or women's style), never about the person
talking: anyone can shop for anyone, so it is only ever set from what the
shopper says.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Lists are capped: the cap is part of the JSON schema Gemini writes against, so a model that starts
# repeating items ("soft", "comfortable", "soft", ...) is stopped instead of writing until it times out.
MAX_LIST_ITEMS = 8


class ShoppingNeed(BaseModel):
    category: Optional[str] = Field(default=None, description="One of the catalog categories, or null if unknown")
    product_types: list[str] = Field(
        default_factory=list, max_length=MAX_LIST_ITEMS,
        description="The specific kinds they want, spelled exactly as in the shop's product types; "
                    "empty when any kind in the category is fine")
    gender: Optional[Literal["men", "women", "any"]] = Field(
        default=None,
        description="Whose style the item is: men's, women's, or 'any' if the shopper said it does not matter. "
                    "Only from what the shopper said, never guessed from who is talking; null if not said")
    looking_for: str = Field(default="", description="What they want, in their own words")
    purpose: Optional[str] = Field(default=None, description="Occasion or use, if given")
    budget_min: Optional[float] = Field(default=None, description="Lower price limit in dollars, if given")
    budget_max: Optional[float] = Field(default=None, description="Upper price limit in dollars, if given")
    must_have: list[str] = Field(default_factory=list, max_length=MAX_LIST_ITEMS, description="Features they need, in plain words a product description would use")
    avoid: list[str] = Field(default_factory=list, max_length=MAX_LIST_ITEMS, description="Qualities they do not want, in plain words a product description would use")

    def gender_filter(self) -> Optional[str]:
        """'men' or 'women' when the search should be limited to that style (plus unisex)."""
        return self.gender if self.gender in ("men", "women") else None

    def search_text(self) -> str:
        """Free text for semantic search."""
        parts = [self.looking_for, *self.product_types, self.purpose or "", *self.must_have]
        return " ".join(p for p in parts if p) or (self.category or "")

    def summary(self) -> str:
        """One plain sentence, used in prompts and to confirm understanding."""
        kind = self.looking_for or " or ".join(self.product_types) or self.category or "something"
        if self.gender_filter():
            kind = f"{self.gender_filter()}'s {kind}"
        parts = [kind]
        if self.purpose:
            parts.append(f"for {self.purpose}")
        if self.must_have:
            parts.append("with " + ", ".join(self.must_have))
        if self.budget_min is not None and self.budget_max is not None:
            parts.append(f"between ${self.budget_min:g} and ${self.budget_max:g}")
        elif self.budget_min is not None:
            parts.append(f"over ${self.budget_min:g}")
        elif self.budget_max is not None:
            parts.append(f"under ${self.budget_max:g}")
        if self.avoid:
            parts.append("avoiding " + ", ".join(self.avoid))
        return " ".join(parts)
