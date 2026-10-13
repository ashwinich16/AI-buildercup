"""
Try-on: the shopper's own photo, wearing the products they chose and, for a look from their outfit plan,
the pieces they already own. Two interchangeable studios behind one interface (SCOUT_TRY_ON_ENGINE):

  GeminiTryOnStudio  (default, "gemini") Gemini's image model: the whole outfit in one call, with an
                     instruction to keep the person exactly as they are, plus any styling the shopper asks
                     for ("tuck the shirt in")
  VertexTryOnStudio  ("vto") Vertex AI Virtual Try-On: built only for garments, very faithful to them, but it
                     takes no instructions and dresses one piece at a time, each on top of the last result

The shopper's photo is used only for this request: it is never saved, logged or kept in memory afterwards.
"""

import base64
import binascii
import logging
from abc import ABC, abstractmethod
from typing import Optional

from google.genai import types

from scout.config import Settings
from scout.domain import Product, WardrobeItem
from scout.tools.image_fetcher import ImageData, ImageFetcher

logger = logging.getLogger(__name__)


class TryOnError(RuntimeError):
    """Nothing could be tried on (no photo of the pieces, or the model could not dress them)."""


class TryOnStudio(ABC):
    follows_wishes = False      # can it follow how the shopper would like it worn ("tuck the shirt in")?

    def __init__(self, images: ImageFetcher):
        self._images = images

    @abstractmethod
    def dress(self, photo: bytes, mime_type: str, products: list[Product],
              owned: Optional[list[WardrobeItem]] = None, wish: str = "") -> ImageData:
        """The photo with every piece put on (their own pieces first, then the products). `wish`: how they
        would like it worn, if the studio can follow it."""

    def _garments(self, products: list[Product], owned: Optional[list[WardrobeItem]]) -> list[tuple[str, ImageData]]:
        """(what it is, its picture) for every piece that has a picture."""
        pieces = [(item.description, _from_data_url(item.thumbnail)) for item in owned or []]
        pieces += [(" ".join(p.title.split()[:10]), next(iter(self._images.fetch_many(p.image_urls, limit=1)), None))
                   for p in products]
        for name, picture in pieces:
            if picture is None:
                logger.warning("No picture to try on for %s", name)
        found = [(name, picture) for name, picture in pieces if picture is not None]
        if not found:
            raise TryOnError("I couldn't find pictures of those pieces to try on.")
        return found


GEMINI_PROMPT = """The first image is a photo of a person. Each following image shows one clothing item:
{items}

Create a photo of this same person wearing all of these items together as one outfit.
- Keep the person exactly as they are: the same face and facial features, skin tone, hair, body shape, pose,
  expression and background. Do not change their identity in any way.
- Show each item faithfully, as in its image: the same colour, pattern, material, length and details.
- Make it look like a natural, realistic photo, with the clothes fitting the person's body naturally.
{wish}"""


class GeminiTryOnStudio(TryOnStudio):
    follows_wishes = True
    def __init__(self, settings: Settings, images: ImageFetcher):
        super().__init__(images)
        self._client = settings.genai_client()
        self._model = settings.try_on_image_model
        self._config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])

    def dress(self, photo: bytes, mime_type: str, products: list[Product],
              owned: Optional[list[WardrobeItem]] = None, wish: str = "") -> ImageData:
        garments = self._garments(products, owned)
        items = "\n".join(f"{n}. {name}" for n, (name, _) in enumerate(garments, start=1))
        styling = f"- How they would like it worn: {wish.strip()}" if wish.strip() else ""
        contents = [types.Part.from_text(text=GEMINI_PROMPT.format(items=items, wish=styling)),
                    types.Part.from_bytes(data=photo, mime_type=mime_type),
                    *(types.Part.from_bytes(data=g.data, mime_type=g.mime_type) for _, g in garments)]
        try:
            response = self._client.models.generate_content(model=self._model, contents=contents,
                                                            config=self._config)
        except Exception as error:
            logger.warning("Gemini try-on failed: %s", error)
            raise TryOnError("I couldn't try that on your photo just now. Please try again.") from error
        for candidate in response.candidates or []:
            for part in (candidate.content.parts if candidate.content else None) or []:
                if part.inline_data and part.inline_data.data:
                    return ImageData(data=part.inline_data.data, mime_type=part.inline_data.mime_type or "image/png")
        raise TryOnError("I couldn't try that on your photo. A photo of you standing, facing the camera, works best.")


class VertexTryOnStudio(TryOnStudio):
    def __init__(self, settings: Settings, images: ImageFetcher):
        super().__init__(images)
        self._client = settings.genai_client(settings.try_on_location)
        self._model = settings.try_on_model

    def dress(self, photo: bytes, mime_type: str, products: list[Product],
              owned: Optional[list[WardrobeItem]] = None, wish: str = "") -> ImageData:
        """One piece at a time, each on top of the last result. `wish` is not supported by this model."""
        person = types.Image(image_bytes=photo, mime_type=mime_type)
        dressed = 0
        for name, garment in self._garments(products, owned):
            try:
                result = self._client.models.recontext_image(
                    model=self._model,
                    source=types.RecontextImageSource(
                        person_image=person,
                        product_images=[types.ProductImage(product_image=types.Image(
                            image_bytes=garment.data, mime_type=garment.mime_type))],
                    ),
                    config=types.RecontextImageConfig(number_of_images=1),
                )
            except Exception as error:      # e.g. a kind of piece the model cannot put on a person
                logger.warning("Try-on failed for %s: %s", name, error)
                continue
            if result.generated_images:
                person = result.generated_images[0].image
                dressed += 1
        if not dressed:
            raise TryOnError("I couldn't try that on your photo.")
        return ImageData(data=person.image_bytes, mime_type=person.mime_type or "image/png")


def build_try_on_studio(settings: Settings, images: ImageFetcher) -> TryOnStudio:
    if settings.try_on_engine == "vto":
        return VertexTryOnStudio(settings, images)
    return GeminiTryOnStudio(settings, images)


def _from_data_url(url: Optional[str]) -> Optional[ImageData]:
    """The picture in a data URL (how a wardrobe piece's photo is kept), or None."""
    if not url or not url.startswith("data:image/") or "," not in url:
        return None
    header, encoded = url.split(",", 1)
    try:
        return ImageData(data=base64.b64decode(encoded), mime_type=header[5:].split(";")[0])
    except (binascii.Error, ValueError):
        return None
