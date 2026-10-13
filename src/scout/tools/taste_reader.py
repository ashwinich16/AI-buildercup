"""
TasteReader: what items look and feel like (colours, materials, patterns, styles), for learning the
shopper's taste. One quick Gemini call for all the items of a turn, made in the background after Scout has
replied, so the shopper never waits for it. Values the profile already has are offered to the model, so the
same colour is always written the same way ("navy", not "navy blue" one time and "dark blue" the next).
"""

import json
import logging

from google.genai import types
from pydantic import BaseModel, Field

from scout.config import Settings
from scout.domain import Traits

logger = logging.getLogger(__name__)


class _ItemTraits(Traits):
    number: int = Field(description="The item's number in the list")


class _Read(BaseModel):
    items: list[_ItemTraits] = Field(default_factory=list)


PROMPT = """For each item below, write what it looks and feels like, in plain everyday words a product
description would use: its main colours, its materials or textures, its patterns (none if it is plain) and a
few style words (how formal, the look). One or two words per value. Only what the item's own details show;
never guess.
When a value means the same as one already used, use that one exactly: {known}

ITEMS
{items}"""


class TasteReader:
    def __init__(self, settings: Settings):
        self._client = settings.genai_client()
        self._model = settings.model_for("taste")
        tuning = settings.tuning("taste")
        self._config = types.GenerateContentConfig(
            temperature=tuning.temperature, response_mime_type="application/json", response_schema=_Read,
            thinking_config=types.ThinkingConfig(thinking_budget=tuning.thinking_budget))

    def read(self, items: list[str], known: list[str]) -> list[Traits]:
        """The traits of each item, in order (empty traits for one that could not be read)."""
        if not items:
            return []
        listed = "\n".join(f"{n}. {text}" for n, text in enumerate(items, start=1))
        prompt = PROMPT.format(known=", ".join(known) or "none yet", items=listed)
        try:
            response = self._client.models.generate_content(model=self._model, contents=prompt, config=self._config)
            read = {item.number: item for item in _Read.model_validate(json.loads(response.text)).items}
        except Exception as error:      # learning is a bonus: a failed read just teaches nothing this time
            logger.warning("Could not read the traits of %d items: %s", len(items), error)
            return [Traits() for _ in items]
        return [Traits.model_validate(read[n].model_dump(exclude={"number"})) if n in read else Traits()
                for n in range(1, len(items) + 1)]


def describe(product) -> str:
    """A product in a few facts, for reading its traits (its title, kind, style and listed details)."""
    facts = [product.title]
    if product.product_type:
        facts.append(f"type: {product.product_type}")
    if product.style_tags:
        facts.append("style: " + ", ".join(product.style_tags[:6]))
    facts += [f"{a.name}: {a.value}" for a in product.attributes[:10]]
    summary = product.spoken_summary_short or (product.description or "")[:200]
    if summary:
        facts.append(summary)
    return "; ".join(facts)
