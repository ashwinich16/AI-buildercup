"""
The shopper's long-term taste profile.

A Preference is one lasting like or dislike ("avoid: colour black, for shoes").
It is either stated (the shopper said it about themselves: full confidence) or
learned (inferred from choices and rejections: low confidence that grows with
repeated evidence), so one rejected product never becomes a firm rule.

Category-agnostic: attribute and value are free text.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

MAX_PREFERENCES = 40
LEARNED_START = 0.4      # confidence of a preference inferred once
LEARNED_STEP = 0.2       # each repeated piece of evidence adds this
LEARNED_MAX = 0.9        # learned preferences never become as certain as stated ones
USE_THRESHOLD = 0.5      # below this, a learned preference is not shown to other agents


class PreferenceKey(BaseModel):
    attribute: str = Field(description="What it is about, in plain words a product description would use")
    value: str = Field(description="The value, in plain words")
    scope: Optional[str] = Field(default=None, description="Catalog category it applies to; null for all")

    def key(self) -> tuple[str, str, str]:
        return (self.attribute.strip().lower(), self.value.strip().lower(), (self.scope or "").strip().lower())


class Preference(PreferenceKey):
    polarity: Literal["like", "avoid"]
    source: Literal["stated", "learned"]
    confidence: float = Field(default=LEARNED_START, description="0.0 to 1.0")
    evidence: int = 1


class PreferenceProfile(BaseModel):
    preferences: list[Preference] = Field(default_factory=list)

    def merge(self, updates: list[Preference], forget: list[PreferenceKey]) -> "PreferenceProfile":
        items = {p.key(): p for p in self.preferences}
        for item in forget:
            items.pop(item.key(), None)

        for update in updates:
            old = items.get(update.key())
            stated = update.source == "stated" or (old is not None and old.source == "stated")
            evidence = (old.evidence + 1) if old else 1
            if stated:
                confidence = 1.0
            elif old:
                confidence = min(LEARNED_MAX, old.confidence + LEARNED_STEP)
            else:
                confidence = min(update.confidence, LEARNED_START)
            items[update.key()] = update.model_copy(update={   
                "source": "stated" if stated else "learned",
                "confidence": round(confidence, 2),
                "evidence": evidence,
            })

        kept = sorted(items.values(), key=lambda p: (p.source == "stated", p.confidence), reverse=True)
        return PreferenceProfile(preferences=kept[:MAX_PREFERENCES])

    def to_prompt_text(self) -> str:
        """Only preferences confident enough to act on."""
        lines = []
        for p in self.preferences:
            if p.source == "stated" or p.confidence >= USE_THRESHOLD:
                scope = f" (for {p.scope})" if p.scope else ""
                lines.append(f"- {p.polarity} {p.attribute}: {p.value}{scope} [{p.source}]")
        return "\n".join(lines) or "Nothing known yet."
