"""
Wardrobe: turns a photo of clothes the shopper owns into WardrobeItems (Gemini vision, one fast call).

A photo may show one piece or several (clothes laid out on a bed); each becomes its own item. The photo is
read once, when it is added, so planning later works from short text and never waits on images. The same call
says what each piece looks and feels like (its Traits), so Scout learns their taste without another call. The
category is one of the shop's own categories, so the Stylist can mix owned pieces with new ones. Only a
small thumbnail of the photo is kept (made by the browser before upload).
"""

import json
import uuid
from typing import Literal, Optional

from google.genai import types
from pydantic import BaseModel, Field

from scout.catalog import CatalogRepository
from scout.config import Settings
from scout.domain import Traits, WardrobeItem


MAX_ITEMS_PER_PHOTO = 8


class _Piece(Traits):
    category: str = Field(description="One of the shop's categories, spelled exactly")
    description: str = Field(description="What it is in plain words: colour, kind, fit, material, how formal")
    made_for: Literal["men", "women", "any"] = Field(
        default="any", description="The cut of the piece itself, when it clearly is a men's or a women's one; "
                                   "any when it could be either")


class OwnedPhotoPiece(BaseModel):
    """A piece seen in a photo: the wardrobe item, and what it says about their taste."""
    item: WardrobeItem
    traits: Traits
    made_for: str = "any"


class _Seen(BaseModel):
    pieces: list[_Piece] = Field(default_factory=list, max_length=MAX_ITEMS_PER_PHOTO)


PROMPT = """This photo shows clothes, shoes, bags or accessories the shopper owns: one piece or several.
The shop's categories are: {categories}.
For each piece you can clearly see, return its category (one of those, spelled exactly) and a short
description a stylist would use, e.g. "navy slim chinos, cotton, smart casual". Also give its main colours,
materials or textures, patterns (none if plain) and a few style words, one or two plain words each, and
whether the piece itself is cut as a men's or a women's one (any when it could be either). Describe only what
you can see. If the photo shows no clothing, or it is too unclear to tell, return no pieces."""


class WardrobeTagger:
    def __init__(self, settings: Settings, catalog: CatalogRepository):
        self._client = settings.genai_client()
        self._model = settings.model_for("wardrobe")
        tuning = settings.tuning("wardrobe")
        self._config = types.GenerateContentConfig(
            temperature=tuning.temperature,
            response_mime_type="application/json", response_schema=_Seen,
            thinking_config=types.ThinkingConfig(thinking_budget=tuning.thinking_budget))
        self._catalog = catalog

    def from_photo(self, photo: bytes, mime_type: str, thumbnail: Optional[str]) -> list[OwnedPhotoPiece]:
        """Every piece in the photo; empty when nothing could be made out."""
        prompt = PROMPT.format(categories=", ".join(self._catalog.categories()))
        response = self._client.models.generate_content(
            model=self._model, config=self._config,
            contents=[types.Part.from_text(text=prompt), types.Part.from_bytes(data=photo, mime_type=mime_type)])
        seen = _Seen.model_validate(json.loads(response.text))
        spelled = {c.lower(): c for c in self._catalog.categories()}
        return [OwnedPhotoPiece(item=new_item(spelled.get(p.category.lower(), p.category), p.description, thumbnail),
                                traits=Traits.model_validate(p.model_dump(include=set(Traits.model_fields))),
                                made_for=p.made_for)
                for p in seen.pieces if p.description.strip()]


def new_item(category: str, description: str, thumbnail: Optional[str] = None) -> WardrobeItem:
    return WardrobeItem(id=uuid.uuid4().hex[:8], category=category, description=description, thumbnail=thumbnail)
