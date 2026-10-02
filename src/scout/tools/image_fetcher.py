"""
ImageFetcher: downloads product photos so a vision model can look at them.

Keeps a small in-memory cache (nothing is written to disk), so follow-up
questions about the same product do not download its photos again.
"""

import logging
from collections import OrderedDict
from typing import Optional

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ImageData(BaseModel):
    data: bytes
    mime_type: str


class ImageFetcher:
    def __init__(self, timeout_seconds: float = 8.0, max_cached: int = 64):
        self._timeout = timeout_seconds
        self._max_cached = max_cached
        self._cache: OrderedDict[str, ImageData] = OrderedDict()

    def fetch(self, url: str) -> Optional[ImageData]:
        if url in self._cache:
            self._cache.move_to_end(url)
            return self._cache[url]
        try:
            response = requests.get(url, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as error:
            logger.warning("Image download failed (%s): %s", url, error)
            return None

        mime_type = response.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        if not mime_type.startswith("image/"):
            mime_type = "image/jpeg"
        image = ImageData(data=response.content, mime_type=mime_type)
        self._remember(url, image)
        return image

    def fetch_many(self, urls: list[str], limit: int) -> list[ImageData]:
        images = []
        for url in urls:
            if len(images) >= limit:
                break
            image = self.fetch(url)
            if image:
                images.append(image)
        return images

    def _remember(self, url: str, image: ImageData) -> None:
        self._cache[url] = image
        if len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
