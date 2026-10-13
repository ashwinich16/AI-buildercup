"""
The shopper's taste, learned over time from what they own and what they shop.

Each item tells a little: its colours, materials, patterns and styles (Traits), read once when it is bought,
saved or added to the wardrobe. Every trait value adds weight, more for stronger signals (buying says more
than owning). A taste is a PATTERN: a value only counts once it has come from more than one item, so one
navy shirt says nothing and navy across several pieces does.

Taste is kept per style (men's / women's), because people shop for others: what someone likes in women's
style is used only when they shop women's style. It never decides the style: Scout always asks.
What the shopper SAYS about their taste is a Preference (preferences.py) and always wins over this.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

TRAITS = ("colour", "material", "pattern", "style")
Signal = Literal["bought", "saved", "owned"]
SIGNAL_WEIGHT: dict[str, float] = {"bought": 3.0, "saved": 2.0, "owned": 1.0}   # how much each one says
STYLES = ("men", "women")          # the styles taste is kept apart for; anything else counts for both ("any")
MAX_MARKS = 200                    # the lightest are dropped beyond this, so the profile stays small
SHOWN_PER_TRAIT = 3                # the strongest few values of each trait are what agents see


class Traits(BaseModel):
    """What one item looks and feels like, in plain words a product description would use."""
    colours: list[str] = Field(default_factory=list, description="Main colours, e.g. 'navy', 'beige'")
    materials: list[str] = Field(default_factory=list, description="Materials or textures, e.g. 'linen', 'denim'")
    patterns: list[str] = Field(default_factory=list, description="Patterns, e.g. 'striped', 'floral'; empty if plain")
    styles: list[str] = Field(default_factory=list, description="Style words, e.g. 'smart casual', 'minimal'")

    def by_trait(self) -> dict[str, list[str]]:
        return {"colour": self.colours, "material": self.materials, "pattern": self.patterns, "style": self.styles}


class TasteMark(BaseModel):
    trait: str                     # one of TRAITS
    value: str                     # e.g. "navy"
    style: str = "any"             # "men", "women", or "any" (counts for both)
    weight: float = 0.0            # how much evidence, by signal strength
    items: int = 0                 # how many items it came from

    def key(self) -> tuple[str, str, str]:
        return (self.trait, self.value, self.style)


class TasteProfile(BaseModel):
    marks: list[TasteMark] = Field(default_factory=list)

    def learn(self, traits: Traits, style: Optional[str], signal: Signal) -> None:
        """One item's traits, counted once."""
        style = style if style in STYLES else "any"
        found = {mark.key(): mark for mark in self.marks}
        for trait, values in traits.by_trait().items():
            for value in dict.fromkeys(_plain(v) for v in values):
                if not value:
                    continue
                mark = found.get((trait, value, style))
                if mark is None:
                    mark = found[(trait, value, style)] = TasteMark(trait=trait, value=value, style=style)
                mark.weight = round(mark.weight + SIGNAL_WEIGHT[signal], 2)
                mark.items += 1
        self.marks = sorted(found.values(), key=lambda m: m.weight, reverse=True)[:MAX_MARKS]

    def forget(self, values: list[str]) -> bool:
        """They asked Scout to forget these ("forget that I like navy"); True if anything was dropped."""
        gone = {_plain(v) for v in values}
        kept = [mark for mark in self.marks if mark.value not in gone]
        changed = len(kept) != len(self.marks)
        self.marks = kept
        return changed

    def patterns(self, style: Optional[str], avoid: set[str] = frozenset()) -> dict[str, list[TasteMark]]:
        """The strongest repeated values of each trait for this style (and those that count for any style),
        leaving out anything they said they avoid."""
        styles = {"any", style} if style in STYLES else {"any", *STYLES}
        merged: dict[tuple[str, str], TasteMark] = {}
        for mark in self.marks:
            if mark.style in styles and mark.value not in avoid:
                seen = merged.setdefault((mark.trait, mark.value), TasteMark(trait=mark.trait, value=mark.value))
                seen.weight += mark.weight
                seen.items += mark.items
        shown: dict[str, list[TasteMark]] = {}
        for trait in TRAITS:
            repeated = [m for m in merged.values() if m.trait == trait and m.items > 1]   # a pattern, not one item
            if repeated:
                shown[trait] = sorted(repeated, key=lambda m: m.weight, reverse=True)[:SHOWN_PER_TRAIT]
        return shown

    def to_prompt_text(self, style: Optional[str], avoid: set[str] = frozenset()) -> str:
        found = self.patterns(style, avoid)
        if not found:
            return "Nothing clear yet."
        return "\n".join(f"- {trait}s they lean towards: " + ", ".join(m.value for m in marks)
                         for trait, marks in found.items())


def _plain(value: str) -> str:
    return " ".join(value.lower().split())
