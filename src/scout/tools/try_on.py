"""
Virtual try-on: the shopper's own photo, wearing the products they chose (Vertex AI Virtual Try-On).

The model dresses one product at a time, so a whole outfit is put on piece by
piece, each on top of the last result. The shopper's photo is used only for
this request: it is never saved, logged or kept in memory afterwards.
"""

import logging

from google import genai
from google.genai import types

from scout.config import Settings
from scout.domain import Product
from scout.tools.image_fetcher import ImageFetcher

logger = logging.getLogger(__name__)


class TryOnError(RuntimeError):
    """Nothing could be tried on (no product photo, or the model could not dress it)."""


class TryOnStudio:
    def __init__(self, settings: Settings, images: ImageFetcher):
        self._client = genai.Client(vertexai=True, project=settings.project_id, location=settings.vertex_location)
        self._model = settings.try_on_model
        self._images = images

    def dress(self, photo: bytes, mime_type: str, products: list[Product]) -> bytes:
        """The photo with every product put on, as PNG bytes."""
        person = types.Image(image_bytes=photo, mime_type=mime_type)
        dressed = 0
        for product in products:
            garment = self._images.fetch_many(product.image_urls, limit=1)
            if not garment:
                logger.warning("No photo to try on for %s", product.id)
                continue
            try:
                result = self._client.models.recontext_image(
                    model=self._model,
                    source=types.RecontextImageSource(
                        person_image=person,
                        product_images=[types.ProductImage(product_image=types.Image(
                            image_bytes=garment[0].data, mime_type=garment[0].mime_type))],
                    ),
                    config=types.RecontextImageConfig(number_of_images=1),
                )
            except Exception as error:      # e.g. a kind of product the model cannot put on a person
                logger.warning("Try-on failed for %s: %s", product.id, error)
                continue
            if result.generated_images:
                person = result.generated_images[0].image
                dressed += 1
        if not dressed:
            raise TryOnError("I couldn't try that on your photo.")
        return person.image_bytes
