"""
Buying advice from reviews, said when a shopper picks a size or is about to add things to the cart: does it run
small or large, and is there a downside many reviewers share (handles that crack, a buckle that feels cheap).

Only patterns count: something at least MIN_REVIEWS reviews say, with how many said it, so one review is never
presented as a trend. The words come from the review points read from the reviews (pipeline/review_evidence.py);
nothing here is invented.
"""

import re
from typing import Literal, Optional

from pydantic import BaseModel

from scout.domain import Product
from scout.domain.sizes import SizeScale, size_scale

MIN_REVIEWS = 2          # a pattern, not one person's experience

_SIZE_ASPECT = re.compile(r"siz|fit", re.I)
_RUNS_SMALL = re.compile(r"size up|sizing up|runs? small|too small|too tight|tight|snug|narrow|smaller than", re.I)
_RUNS_LARGE = re.compile(r"size down|sizing down|runs? (large|big)|too (large|big|loose)|loose|larger than|bigger than|boxy", re.I)
_TRUE_TO_SIZE = re.compile(r"true to size|fits? (as expected|well|perfectly)|good fit", re.I)


class SizeAdvice(BaseModel):
    direction: Literal["up", "down", "true", "mixed"]
    count: int                     # reviews that say it
    analyzed: Optional[int] = None # reviews it was drawn from

    def reviews(self) -> str:
        return f"{self.count} of {self.analyzed} reviews" if self.analyzed else f"{self.count} reviews"


def size_advice(product: Product) -> Optional[SizeAdvice]:
    """What reviewers say about its size, from the strongest sizing point that enough of them share. Only for
    things chosen in a size ("a laptop bag runs large" is about capacity, not a size to pick)."""
    if size_scale(product) is None:
        return None
    points = [p for p in product.review_points
              if _SIZE_ASPECT.search(p.aspect) and (p.count() or 0) >= MIN_REVIEWS]
    if not points:
        return None
    point = max(points, key=lambda p: p.count() or 0)
    small, large = bool(_RUNS_SMALL.search(point.finding)), bool(_RUNS_LARGE.search(point.finding))
    if small and large:
        direction = "mixed"
    elif small:
        direction = "up"
    elif large:
        direction = "down"
    elif _TRUE_TO_SIZE.search(point.finding):
        direction = "true"
    else:
        return None
    return SizeAdvice(direction=direction, count=point.count() or 0, analyzed=product.reviews_analyzed)


def plural(kind: str) -> bool:
    """"jeans", "dress pants", "chinos" are spoken of as "they"; "shirt", "dress" as "it"."""
    return kind.endswith("s") and not kind.endswith("ss")


def possessive(kind: str) -> str:
    return f"{kind}'" if plural(kind) else f"{kind}'s"


def size_sentence(product: Product) -> str:
    """One sentence to say before they choose its size, or "" when reviewers agree on nothing."""
    advice = size_advice(product)
    kind = product.kind or "item"
    if advice is None:
        return ""
    run, fit = ("run", "fit") if plural(kind) else ("runs", "fits")
    return {
        "up": f"Reviewers often say the {kind} {run} small ({advice.reviews()}), so you might go one size up.",
        "down": f"Reviewers often say the {kind} {run} large ({advice.reviews()}), so you might go one size down.",
        "true": f"Reviewers say the {kind} {fit} true to size ({advice.reviews()}).",
        "mixed": f"Reviewers find the {possessive(kind)} sizing a little inconsistent ({advice.reviews()}), so check your usual size.",
    }[advice.direction]


def better_size(product: Product, scale: SizeScale, size: str) -> Optional[str]:
    """The size reviewers' experience points to instead of `size` (one up when it runs small), or None."""
    advice = size_advice(product)
    if advice is None or advice.direction not in ("up", "down") or size not in scale.options:
        return None
    n = scale.options.index(size) + (1 if advice.direction == "up" else -1)
    return scale.options[n] if 0 <= n < len(scale.options) else None


def heads_up(product: Product) -> Optional[str]:
    """The downside most reviewers share (not about size), e.g. "7 of 20 reviewers flag the belt's durability"."""
    points = [p for p in product.review_points
              if p.sentiment == "negative" and not _SIZE_ASPECT.search(p.aspect) and (p.count() or 0) >= MIN_REVIEWS]
    if not points:
        return None
    point = max(points, key=lambda p: p.count() or 0)
    who = f"{point.count()} of {product.reviews_analyzed} reviewers" if product.reviews_analyzed else f"{point.count()} reviewers"
    return f"{who} flag the {possessive(product.kind or 'item')} {point.aspect.lower()}"
