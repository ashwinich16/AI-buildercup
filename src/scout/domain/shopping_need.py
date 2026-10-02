"""
ShoppingNeed: what the shopper is looking for, built up over the conversation.

Category-agnostic on purpose: "purpose", "must_have" and "avoid" hold free
text, so "size 9, wide fit" and "16 GB RAM" fit equally well. The
Elicitation agent fills it; the Retrieval tool turns it into a catalog query.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ShoppingNeed(BaseModel):
    category: Optional[str] = Field(default=None, description="One of the catalog categories, or null if unknown")
    looking_for: str = Field(default="", description="What they want, in their own words")
    purpose: Optional[str] = Field(default=None, description="Occasion or use, if given")
    budget_max: Optional[float] = Field(default=None, description="Upper price limit in dollars, if given")
    must_have: list[str] = Field(default_factory=list, description="Features they need, in plain words a product description would use")
    avoid: list[str] = Field(default_factory=list, description="Qualities they do not want, in plain words a product description would use")

    def search_text(self) -> str:
        """Free text for semantic search."""
        parts = [self.looking_for, self.purpose or "", *self.must_have]
        return " ".join(p for p in parts if p) or (self.category or "")

    def summary(self) -> str:
        """One plain sentence, used in prompts and to confirm understanding."""
        parts = [self.looking_for or self.category or "something"]
        if self.purpose:
            parts.append(f"for {self.purpose}")
        if self.must_have:
            parts.append("with " + ", ".join(self.must_have))
        if self.budget_max is not None:
            parts.append(f"under ${self.budget_max:g}")
        if self.avoid:
            parts.append("avoiding " + ", ".join(self.avoid))
        return " ".join(parts)
