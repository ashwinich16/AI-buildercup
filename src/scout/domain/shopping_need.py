"""
ShoppingNeed: what the shopper is looking for, built up over the conversation.

Category-agnostic on purpose: "purpose", "must_have" and "avoid" hold free
text, so "size 9, wide fit" and "16 GB RAM" fit equally well. The
Elicitation agent fills it; the Retrieval tool turns it into a catalog query.

`gender` is about the ITEM (men's or women's style), never about the person
talking: anyone can shop for anyone, so it is only ever set from what the
shopper says.
"""

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

# Words that say whose style an item is. "\bmen" never matches inside "women".
_WOMEN = re.compile(r"\b(women'?s?|woman'?s?|womens|ladies|lady|female|girls?'?|"
                    r"(my|for|her) (wife|mother|mom|mum|sister|daughter|girlfriend|aunt|grandma|grandmother|niece))\b", re.I)
_MEN = re.compile(r"\b(men'?s?|mens|man'?s?|male|gents?|gentlemen|guys?|boys?'?|"
                  r"(my|for|his) (husband|father|dad|brother|son|boyfriend|uncle|grandpa|grandfather|nephew))\b", re.I)


def gender_in_words(text: str) -> Optional[str]:
    """'men' or 'women' when the shopper's words clearly name one style (or who it is for); else None.

    A safety net for the Elicitation model, which sometimes misses a short answer like "Women's jeans."
    Mentioning both ("I'm a man buying for my wife") is left to the model.
    """
    text = (text or "").replace("’", "'")
    women, men = bool(_WOMEN.search(text)), bool(_MEN.search(text))
    if women == men:
        return None
    return "women" if women else "men"


COLOURS = {
    "black", "white", "grey", "gray", "silver", "blue", "navy", "red", "maroon", "burgundy", "pink", "purple",
    "green", "olive", "yellow", "orange", "brown", "tan", "beige", "khaki", "cream", "ivory", "gold",
}
_NEGATION = re.compile(r"(no|not|don'?t|doesn'?t|never|avoid|without|except|hate|dislike)", re.I)


def colours_wanted(text: str) -> list[str]:
    """Colours the shopper asks for in their own words ("black jeans", "I asked for black").

    A colour right after a negation ("not black", "I don't want black") is not counted, unless they
    also ask for it elsewhere in the message ("They are not black. I asked for black jeans.").
    """
    words = re.findall(r"[a-z']+", (text or "").lower().replace("’", "'"))
    wanted = []
    for i, word in enumerate(words):
        if word in COLOURS and not any(_NEGATION.fullmatch(w) for w in words[max(0, i - 3):i]):
            wanted.append("grey" if word == "gray" else word)
    return list(dict.fromkeys(wanted))


# Lists are capped: the cap is part of the JSON schema Gemini writes against, so a model that starts
# repeating items ("soft", "comfortable", "soft", ...) is stopped instead of writing until it times out.
MAX_LIST_ITEMS = 8


class ShoppingNeed(BaseModel):
    category: Optional[str] = Field(default=None, description="One of the catalog categories, or null if unknown")
    product_types: list[str] = Field(
        default_factory=list, max_length=MAX_LIST_ITEMS,
        description="The specific kinds they want, spelled as in the shop's product types (e.g. ['shirt'] or "
                    "['shirt', 'polo']); empty when any kind in the category is fine")
    gender: Optional[Literal["men", "women", "any"]] = Field(
        default=None,
        description="Whose style the item is: men's, women's, or 'any' if the shopper said it does not matter. "
                    "Only from what the shopper said, never guessed from who is talking; null if not said")
    looking_for: str = Field(default="", description="What they want, in their own words")
    purpose: Optional[str] = Field(default=None, description="Occasion or use, if given")
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
        if self.budget_max is not None:
            parts.append(f"under ${self.budget_max:g}")
        if self.avoid:
            parts.append("avoiding " + ", ".join(self.avoid))
        return " ".join(parts)
