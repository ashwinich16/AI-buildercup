"""
ReviewEvidence: the 2-3 real review quotes that best back up a recommendation
for THIS shopper, e.g.  "fits perfectly" (3 of 10 reviews).

Plain Python, no LLM call, so it adds no delay. The quotes themselves were
copied word for word from reviews and checked in the data pipeline
(pipeline/review_evidence.py); the counts are how many analysed reviews say it.

Choice of quotes:
  - points that share words with what the shopper asked for (what they want,
    purpose, must-haves and things to avoid) come first, then the most-supported ones
  - positive points justify the recommendation; one well-supported negative
    point is added as an honest heads-up, so evidence is never one-sided
"""

from typing import Optional

from pydantic import BaseModel

from scout.domain import Product, ReviewPoint, ShoppingNeed
from scout.tools.retrieval import word_stems


class Evidence(BaseModel):
    quote: str
    aspect: str
    sentiment: str                       # "positive", or "negative" for a heads-up
    supporting: int                      # reviews that say it
    analyzed: Optional[int] = None       # reviews it was drawn from

    def count_text(self) -> str:
        if self.analyzed:
            return f"{self.supporting} of {self.analyzed} review{'s' if self.analyzed != 1 else ''}"
        return f"{self.supporting} review{'s' if self.supporting != 1 else ''}"

    def as_text(self) -> str:
        label = "Heads-up" if self.sentiment == "negative" else self.aspect
        return f'{label}: "{self.quote}" ({self.count_text()})'


class EvidencePicker:
    def __init__(self, supporting: int = 2, heads_up_min_reviews: int = 2):
        self._supporting = supporting                 # quotes that justify the recommendation
        self._heads_up_min = heads_up_min_reviews     # a downside needs this many reviews to be mentioned

    def pick(self, product: Product, need: Optional[ShoppingNeed] = None) -> list[Evidence]:
        points = [p for p in product.review_points if p.quotes and (p.count() or 0) > 0]
        if not points:
            return []
        terms = _need_terms(need)

        def score(point: ReviewPoint) -> tuple:
            return (_relevance(point, terms), point.count() or 0)

        positives = sorted((p for p in points if p.sentiment != "negative"), key=score, reverse=True)
        negatives = sorted((p for p in points if p.sentiment == "negative" and (p.count() or 0) >= self._heads_up_min),
                           key=score, reverse=True)

        chosen = positives[: self._supporting]
        if negatives:
            chosen.append(negatives[0])
        elif len(positives) > self._supporting:
            chosen.append(positives[self._supporting])
        return [Evidence(quote=p.quotes[0], aspect=p.aspect, sentiment=p.sentiment or "positive",
                         supporting=p.count() or 0, analyzed=product.reviews_analyzed) for p in chosen]


def _need_terms(need: Optional[ShoppingNeed]) -> set[str]:
    if need is None:
        return set()
    return word_stems(" ".join([need.looking_for, need.purpose or "", *need.must_have, *need.avoid]))


def _relevance(point: ReviewPoint, terms: set[str]) -> int:
    if not terms:
        return 0
    return len(terms & word_stems(" ".join([point.aspect, point.finding, *point.quotes])))
