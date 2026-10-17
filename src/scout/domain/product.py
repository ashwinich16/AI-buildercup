"""
Product: the one data model every Scout agent and tool works with.

It is deliberately category-agnostic (hybrid design):
  - core fields that any product in any catalog has (id, title, price, ...)
  - flexible lists for what differs by category (attributes, review points)
  - `extra` for anything else the source data carries, so nothing is lost

Agents never see search results or table columns. A catalog adapter maps the
source data into Product; switching from fashion to electronics only changes
that adapter, not this model or the agents.
"""

from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field


class Attribute(BaseModel):
    """One fact about a product, e.g. color = sky blue, or screen size = 15.6 inch."""

    name: str
    value: str
    source: str = "seller"                # "image" (seen in photos) or "seller" (stated in text)
    confidence: Optional[float] = None    # 0.0 to 1.0 when known

    def as_text(self) -> str:
        hedge = " (looks like)" if self.source == "image" and (self.confidence or 1.0) < 0.7 else ""
        return f"{self.name}: {self.value}{hedge} [{self.source}]"


class ReviewPoint(BaseModel):
    """One thing reviewers say, e.g. sizing: runs a size small (7 mentions)."""

    aspect: str
    finding: str
    sentiment: Optional[str] = None       # positive / negative / mixed
    mentions: Optional[int] = None        # Gemini's estimate (older data)
    supporting_reviews: Optional[int] = None   # counted from the reviews that say it (review_evidence.py)
    quotes: list[str] = Field(default_factory=list)   # word-for-word snippets checked against the reviews

    def count(self) -> Optional[int]:
        return self.supporting_reviews if self.supporting_reviews is not None else self.mentions


class Product(BaseModel):
    # Core: present for any product in any category (only id and title are required)
    id: str
    title: str
    category: Optional[str] = None
    product_type: Optional[str] = None   
    gender: Optional[str] = None         
    brand: Optional[str] = None
    price: Optional[float] = None
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    image_urls: list[str] = Field(default_factory=list)

    # Descriptions
    description: Optional[str] = None
    spoken_summary_short: Optional[str] = None
    spoken_summary_detailed: Optional[str] = None
    style_tags: list[str] = Field(default_factory=list)

    # Flexible, category-specific facts
    attributes: list[Attribute] = Field(default_factory=list)

    # Reviews
    review_summary: Optional[str] = None
    review_points: list[ReviewPoint] = Field(default_factory=list)
    review_excerpts: list[str] = Field(default_factory=list)
    reviews_analyzed: Optional[int] = None     

    # Anything else from the source data
    extra: dict[str, Any] = Field(default_factory=dict)

    SHORT_TEXT_CHARS: ClassVar[int] = 300          
    SHORT_FINDINGS_PER_ASPECT: ClassVar[int] = 3   

    @property
    def kind(self) -> str:
        """What sort of product it is: its kind within the category, else the category ("" if unknown)."""
        return self.product_type or self.category or ""

    @property
    def short_description(self) -> str:
        """Best available short summary, falling back when enrichment is missing."""
        if self.spoken_summary_short:
            return self.spoken_summary_short
        if self.description:
            return _clip(self.description, self.SHORT_TEXT_CHARS)
        return self.title

    def review_digest(self, findings_per_aspect: Optional[int] = None) -> str:
        """Review points grouped by aspect: 'fit: too narrow / fits perfectly (7 mentions)'."""
        grouped: dict[str, list[str]] = {}
        for point in self.review_points:
            count = f" ({point.count()} mentions)" if point.count() else ""
            grouped.setdefault(point.aspect, []).append(point.finding + count)
        parts = []
        for aspect, findings in grouped.items():
            shown = findings[:findings_per_aspect] if findings_per_aspect else findings
            more = len(findings) - len(shown)
            parts.append(f"{aspect}: " + " / ".join(shown) + (f" (+{more} more)" if more else ""))
        return "; ".join(parts)

    def to_prompt_text(self, detailed: bool = False) -> str:
        """Compact, token-efficient text for Gemini. Empty fields are left out.

        detailed=False: enough to shortlist and compare.
        detailed=True:  everything needed to describe one product vividly.
        """
        lines = [f"[{self.id}] {self.title}"]
        lines += self._core_lines()
        lines.append(f"Summary: {self.short_description}")

        if self.attributes:
            lines.append("Attributes: " + "; ".join(a.as_text() for a in self.attributes))
        if self.style_tags:
            lines.append("Style (interpretation, not fact): " + ", ".join(self.style_tags))
        if self.review_points:
            per_aspect = None if detailed else self.SHORT_FINDINGS_PER_ASPECT
            lines.append("Reviewers say: " + self.review_digest(per_aspect))

        if detailed:
            if self.spoken_summary_detailed:
                lines.append(f"Detailed description: {self.spoken_summary_detailed}")
            elif self.description:
                lines.append(f"Seller description: {self.description}")
            if self.review_summary:
                lines.append(f"Review summary: {self.review_summary}")
            if self.review_excerpts:
                lines.append("Review quotes: " + " | ".join(f'"{q}"' for q in self.review_excerpts))

        return "\n".join(lines)

    def _core_lines(self) -> list[str]:
        facts = []
        if self.category:
            facts.append(f"category {self.category}")
        if self.product_type:
            facts.append(f"type {self.product_type}")
        if self.gender:
            facts.append("unisex" if self.gender == "unisex" else f"{self.gender}'s")
        if self.brand:
            facts.append(f"brand {self.brand}")
        facts.append(f"price ${self.price:.2f}" if self.price is not None else "price unknown")
        if self.rating is not None:
            count = f" from {self.rating_count} ratings" if self.rating_count else ""
            facts.append(f"rated {self.rating:.1f}/5{count}")
        return [", ".join(facts)]


def _clip(text: str, max_chars: int) -> str:
    """Shorten text at a sentence (or word) boundary."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > max_chars // 3 else cut.rsplit(" ", 1)[0] + "..."
