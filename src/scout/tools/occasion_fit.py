"""
OccasionRanker: puts the products found for an outfit piece in order of how well they suit the occasion
("a beach weekend: lounging by the water, dinner in town"), so a linen shirt comes before a flannel one.

Search finds products that match the piece; it does not judge them against the occasion. One quick
Gemini call per piece does that, and gives a short reason Scout can say ("light and breathable for the
heat"). If the call fails, the search order is kept: the outfit is never held up by it.
"""
import json
import logging
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from scout.config import Settings
from scout.domain import Product

logger = logging.getLogger(__name__)

MIN_OPTIONS = 3      # the shopper is always offered a choice of three, when the shop has them


class OccasionFit(BaseModel):
    product_id: str
    why: str = ""          # why it suits the occasion, in a few words


class _Ranking(BaseModel):
    ranked: list[OccasionFit] = Field(default_factory=list)


RANK_PROMPT = """You are a stylist choosing a {piece} for this occasion: {occasion}
What the piece should be like: {looks_like}

PRODUCTS FOUND IN THE SHOP
{products}

Put the products in order, the best for THIS occasion first: think of the setting, the weather it implies,
what they will be doing, and how dressed up it is. Leave out any product that would clearly be wrong for
it (for example heavy wool for a beach, or gym wear for a wedding). For each one you keep, give why it suits
the occasion in at most eight plain words, from its real details. Use the ids exactly as given."""


class OccasionRanker:
    def __init__(self, settings: Settings):
        self._client = genai.Client(vertexai=True, project=settings.project_id, location=settings.vertex_location)
        self._model = settings.model_for("occasion")
        tuning = settings.tuning("occasion")
        self._config = types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=_Ranking,
            thinking_config=types.ThinkingConfig(thinking_budget=tuning.thinking_budget),
        )

    def rank(self, occasion: str, piece: str, looks_like: str,
             products: list[Product]) -> tuple[list[Product], dict[str, str]]:
        """The products best first for the occasion (clearly wrong ones left out, unless that would leave
        nothing), and why each suits it, by product id."""
        if not occasion or len(products) < 2:
            return products, {}
        listed = "\n".join(f"[{p.id}] {_describe(p)}" for p in products)
        prompt = RANK_PROMPT.format(piece=piece, occasion=occasion, looks_like=looks_like or piece, products=listed)
        try:
            response = self._client.models.generate_content(model=self._model, contents=prompt, config=self._config)
            ranking = _Ranking.model_validate(json.loads(response.text)).ranked
        except Exception as error:   # the search order is still a reasonable answer
            logger.warning("Could not rank %s for the occasion: %s", piece, error)
            return products, {}
        by_id = {p.id: p for p in products}
        kept = [fit for fit in ranking if fit.product_id in by_id]
        kept = list({fit.product_id: fit for fit in kept}.values())     # each product once
        if not kept:
            return products, {}
        ranked = [by_id[fit.product_id] for fit in kept]
        # Too few left to choose from: the rest follow in search order, after the ones that suit it.
        ranked += [p for p in products if p not in ranked][:max(MIN_OPTIONS - len(ranked), 0)]
        return ranked, {fit.product_id: fit.why for fit in kept if fit.why}


def _describe(product: Product) -> str:
    facts = [product.title]
    if product.product_type:
        facts.append(f"type: {product.product_type}")
    if product.style_tags:
        facts.append("style: " + ", ".join(product.style_tags[:6]))
    facts += [f"{a.name}: {a.value}" for a in product.attributes[:8]]
    summary: Optional[str] = product.spoken_summary_short or (product.description or "")[:200]
    if summary:
        facts.append(summary)
    return "; ".join(facts)
