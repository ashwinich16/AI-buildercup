"""
Virtual try-on: the shopper's own photo, wearing the products they chose and, for a look from their outfit
plan, the pieces they already own (Vertex AI Virtual Try-On).

The model dresses one piece at a time, so a whole outfit is put on piece by
piece, each on top of the last result. The shopper's photo is used only for
this request: it is never saved, logged or kept in memory afterwards.
"""

import base64
import binascii
import logging
from typing import Optional

from google.genai import types

from scout.config import Settings
from scout.domain import Product, WardrobeItem
from scout.tools.image_fetcher import ImageData, ImageFetcher

logger = logging.getLogger(__name__)


class TryOnError(RuntimeError):
    """Nothing could be tried on (no product photo, or the model could not dress it)."""


class TryOnStudio:
    def __init__(self, settings: Settings, images: ImageFetcher):
        self._client = settings.genai_client(settings.try_on_location)
        self._model = settings.try_on_model
        self._images = images

    def dress(self, photo: bytes, mime_type: str, products: list[Product],
              owned: Optional[list[WardrobeItem]] = None) -> bytes:
        """The photo with every piece put on (their own pieces first, then the products), as PNG bytes."""
        garments = [(item.description, _from_data_url(item.thumbnail)) for item in owned or []]
        garments += [(product.id, next(iter(self._images.fetch_many(product.image_urls, limit=1)), None))
                     for product in products]
        person = types.Image(image_bytes=photo, mime_type=mime_type)
        dressed = 0
        for name, garment in garments:
            if garment is None:
                logger.warning("No photo to try on for %s", name)
                continue
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
        return person.image_bytes


def _from_data_url(url: Optional[str]) -> Optional[ImageData]:
    """The picture in a data URL (how a wardrobe piece's photo is kept), or None."""
    if not url or not url.startswith("data:image/") or "," not in url:
        return None
    header, encoded = url.split(",", 1)
    try:
        return ImageData(data=base64.b64decode(encoded), mime_type=header[5:].split(";")[0])
    except (binascii.Error, ValueError):
        return None
