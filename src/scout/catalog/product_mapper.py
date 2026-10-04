"""
Turns one raw catalog record (a plain dict) into a Product.

This is the ONLY place that knows source field names. Each Product field has
a list of candidate source names; the first one present wins. To use a new
catalog (e.g. electronics with "name" and "stars"), pass a different FieldMap;
agents and the Product model do not change.

It also accepts both shapes the Scout pipeline has produced:
  attributes   - dict   {"color": "navy", ...}                 (older enrich)
               - list   [{"name", "value", "source", ...}]     (newer enrich)
  review points- dict   {"fit_feedback": ["runs small", ...]}  (older enrich)
               - list   [{"aspect", "finding", ...}]           (newer enrich)
"""

from dataclasses import dataclass
from typing import Any, Optional

from scout.domain import Attribute, Product, ReviewPoint

_EMPTY_VALUES = {"", "unknown", "n/a", "none", "not visible", "not specified", "null"}


@dataclass(frozen=True)
class FieldMap:
    """Candidate source field names for each Product field, in priority order."""

    id: tuple[str, ...] = ("product_id", "id")
    title: tuple[str, ...] = ("title", "name")
    category: tuple[str, ...] = ("category",)
    product_type: tuple[str, ...] = ("product_type",)
    gender: tuple[str, ...] = ("gender",)
    brand: tuple[str, ...] = ("brand", "store")
    price: tuple[str, ...] = ("price",)
    rating: tuple[str, ...] = ("average_rating", "rating")
    rating_count: tuple[str, ...] = ("rating_number", "rating_count")
    image_urls: tuple[str, ...] = ("image_urls", "images")
    description: tuple[str, ...] = ("description", "features")
    spoken_summary_short: tuple[str, ...] = ("spoken_summary_short",)
    spoken_summary_detailed: tuple[str, ...] = ("spoken_summary_detailed",)
    style_tags: tuple[str, ...] = ("style_tags",)
    attributes: tuple[str, ...] = ("attributes", "visual_attributes")
    review_summary: tuple[str, ...] = ("review_summary",)
    review_points: tuple[str, ...] = ("review_points", "review_aspects")
    review_excerpts: tuple[str, ...] = ("review_excerpts",)
    reviews_analyzed: tuple[str, ...] = ("reviews_analyzed",)
    ignored: tuple[str, ...] = ("search_text",)   # index-only fields, never shown to agents


class ProductMapper:
    def __init__(self, field_map: Optional[FieldMap] = None):
        self._fields = field_map or FieldMap()

    def to_product(self, record: dict[str, Any], fallback_id: str = "") -> Product:
        used: set[str] = set(self._fields.ignored)

        def pick(field: str) -> Any:
            for name in getattr(self._fields, field):
                value = record.get(name)
                if not _is_empty(value):
                    used.add(name)
                    return value
            return None

        product = Product(
            id=str(pick("id") or fallback_id),
            title=_text(pick("title")) or "Untitled product",
            category=_text(pick("category")),
            product_type=_text(pick("product_type")),
            gender=_text(pick("gender")),
            brand=_text(pick("brand")),
            price=_number(pick("price")),
            rating=_number(pick("rating")),
            rating_count=_integer(pick("rating_count")),
            image_urls=_string_list(pick("image_urls")),
            description=_text(pick("description")),
            spoken_summary_short=_text(pick("spoken_summary_short")),
            spoken_summary_detailed=_text(pick("spoken_summary_detailed")),
            style_tags=_string_list(pick("style_tags")),
            attributes=_attributes(pick("attributes")),
            review_summary=_text(pick("review_summary")),
            review_points=_review_points(pick("review_points")),
            review_excerpts=_string_list(pick("review_excerpts")),
            reviews_analyzed=_integer(pick("reviews_analyzed")),
        )
        product.extra = {k: v for k, v in record.items() if k not in used and not _is_empty(v)}
        return product


# ---------------------------------------------------------------------
# Small, forgiving converters
# ---------------------------------------------------------------------

def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _EMPTY_VALUES
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _text(value: Any) -> Optional[str]:
    if _is_empty(value):
        return None
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if not _is_empty(v)) or None
    return str(value).strip()


def _string_list(value: Any) -> list[str]:
    if _is_empty(value):
        return []
    items = value if isinstance(value, list) else [value]
    return [str(v).strip() for v in items if not _is_empty(v)]


def _number(value: Any) -> Optional[float]:
    if _is_empty(value):
        return None
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


def _integer(value: Any) -> Optional[int]:
    number = _number(value)
    return int(number) if number is not None else None


def _attributes(value: Any) -> list[Attribute]:
    if isinstance(value, dict):   # {"color": "navy", ...} -> facts observed in photos
        return [Attribute(name=k.replace("_", " "), value=_text(v), source="image")
                for k, v in value.items() if not _is_empty(v)]
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict) and not _is_empty(item.get("name")) and not _is_empty(item.get("value")):
                result.append(Attribute(
                    name=str(item["name"]),
                    value=_text(item["value"]),
                    source=str(item.get("source") or "seller"),
                    confidence=_number(item.get("confidence")),
                ))
        return result
    return []


def _review_points(value: Any) -> list[ReviewPoint]:
    if isinstance(value, dict):   # {"fit_feedback": ["runs small", ...]}
        aspect_name = lambda key: key.replace("_feedback", "").replace("_", " ")
        return [ReviewPoint(aspect=aspect_name(k), finding=_text(f))
                for k, findings in value.items()
                for f in _string_list(findings)]
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict) and not _is_empty(item.get("finding")):
                result.append(ReviewPoint(
                    aspect=str(item.get("aspect") or "general"),
                    finding=_text(item["finding"]),
                    sentiment=item.get("sentiment"),
                    mentions=_integer(item.get("mentions")),
                    supporting_reviews=_integer(item.get("supporting_reviews")),
                    quotes=_string_list(item.get("quotes")),
                ))
        return result
    return []
