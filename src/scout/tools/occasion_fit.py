"""
OccasionRanker: puts the products found for an outfit piece in order of how well they suit the occasion
("a beach weekend: lounging by the water, dinner in town"), so a linen shirt comes before a flannel one.

Search finds products that match the piece; it does not judge them against the occasion. One quick
Gemini call per piece does that, and gives a short reason Scout can say ("light and breathable for the
heat"). If the call fails, the search order is kept: the outfit is never held up by it.
"""
import json
import logging
from typing import Literal, Optional

from google.genai import types
from pydantic import BaseModel, Field

from scout.config import Settings
from scout.domain import Product

logger = logging.getLogger(__name__)

JUDGE_TIMEOUT_S = 12    # usually 2-5 s; past this the search order is used for that piece instead

class OccasionFit(BaseModel):
    product_id: str
    fit: Literal["good", "close", "wrong"] = "good"   # would a well-dressed friend wear it there?
    why: str = ""          # why it suits the occasion (or how it falls short), in a few words


class _Ranking(BaseModel):
    ranked: list[OccasionFit] = Field(default_factory=list)


RANK_PROMPT = """You are a stylist choosing a {piece} for this occasion: {occasion}
What the piece should be like: {looks_like}

PRODUCTS FOUND IN THE SHOP
{products}

Judge every product for THIS occasion: think of the setting, the weather it implies,
what they will be doing, and how dressed up it is. Judge each one like a well-dressed friend:
- fit "good": a sensible person would happily wear it there.
- fit "close": it would do, but is a step off (a little less formal, a little more casual); say how in why.
- fit "wrong": made for another activity (gym, running, football, hiking, ski or winter sports, safety work,
  sleep or lounging) when the occasion is not that activity, or clearly out of place there. For anything smart,
  formal or for work, hoodies, sweatshirts, track or athletic jackets, graphic tees and flip-flops are wrong.
How dressed up the occasion is sets the bar: for a formal one, proper formal pieces are good, smart-casual ones
(chinos, loafers) are close, and casual, outdoor, tactical, cargo or sporty ones are wrong.
Judge each product on its own against the occasion, never against the others: if none of them is good, none is
marked good. Judge by the product's real details, not only its kind.
List ONLY the good and close ones, best first; leave the wrong ones out entirely. For each, give in at most six plain
words why it suits the occasion (or, for a close one, how it falls short). Use the ids exactly as given."""


class OccasionRanker:
    def __init__(self, settings: Settings):
        self._client = settings.genai_client(timeout_s=JUDGE_TIMEOUT_S)
        self._model = settings.model_for("occasion")
        tuning = settings.tuning("occasion")
        self._config = types.GenerateContentConfig(
            temperature=tuning.temperature,
            response_mime_type="application/json",
            response_schema=_Ranking,
            thinking_config=types.ThinkingConfig(thinking_budget=tuning.thinking_budget),
        )

    def rank(self, occasion: str, piece: str, looks_like: str,
             products: list[Product]) -> tuple[list[Product], Optional[dict[str, str]]]:
        """The products best first for the occasion (ones that do not suit it last, never left out: they are
        still there for "more options"), and why each that suits it does, by product id (every product that
        suits it is there, so its length says how many do). None when it could not judge."""
        if not occasion or len(products) < 2:
            return products, {}
        listed = "\n".join(f"[{p.id}] {_describe(p)}" for p in products)
        prompt = RANK_PROMPT.format(piece=piece, occasion=occasion, looks_like=looks_like or piece, products=listed)
        try:
            response = self._client.models.generate_content(model=self._model, contents=prompt, config=self._config)
            ranking = _Ranking.model_validate(json.loads(response.text)).ranked
        except Exception as error:   # the search order is still a reasonable answer
            logger.warning("Could not rank %s for the occasion: %s", piece, error)
            return products, None
        by_id = {p.id: p for p in products}
        fits = list({fit.product_id: fit for fit in ranking if fit.product_id in by_id}.values())   # each once
        if not fits:                              # judged, and none of them suits it
            return products, {}
        fits.sort(key=lambda fit: ("good", "close", "wrong").index(fit.fit))   # stable: its order within each
        ranked = [by_id[fit.product_id] for fit in fits]
        ranked += [p for p in products if p.id not in {fit.product_id for fit in fits}]   # any it skipped
        # Good ones, and close ones said honestly ("a step less formal"); wrong ones are never noted.
        return ranked, {fit.product_id: (fit.why if fit.fit == "good" else f"a close match: {fit.why}")
                        for fit in fits if fit.fit != "wrong"}


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
