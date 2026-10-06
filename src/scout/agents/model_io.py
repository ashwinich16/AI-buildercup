"""
What goes into and comes out of a model call, in one place, so every agent
reads and writes it the same way:

  - content_text:   the text of a message (the shopper's, or a model's), without thoughts
  - response_json:  a model's structured (JSON) answer, or None if it is broken or cut off
  - attach:         extra parts (e.g. product photos) added to the request the model is about to get
"""

import json
import logging
from typing import Optional, TypeVar

from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from pydantic import BaseModel, ValidationError

from scout.tools import ImageFetcher

logger = logging.getLogger(__name__)

Model = TypeVar("Model", bound=BaseModel)


def content_text(content: Optional[types.Content], separator: str = "") -> str:
    """All non-thought text in a message."""
    if not content or not content.parts:
        return ""
    return separator.join(part.text for part in content.parts if part.text and not part.thought)


def response_json(llm_response: LlmResponse, schema: type[Model]) -> Optional[Model]:
    """The model's answer parsed into `schema`, or None (and a warning) if it can't be read."""
    if llm_response.partial or not llm_response.content:
        return None
    try:
        return schema.model_validate(json.loads(content_text(llm_response.content)))
    except (json.JSONDecodeError, ValidationError) as error:
        logger.warning("%s output could not be parsed: %s", schema.__name__, error)
        return None


def photo_parts(images: ImageFetcher, urls: list[str], limit: int, label: str) -> list[types.Part]:
    """A label followed by up to `limit` product photos, or nothing if none could be loaded."""
    photos = images.fetch_many(urls, limit=limit)
    if not photos:
        return []
    return [types.Part(text=label), *(types.Part.from_bytes(data=p.data, mime_type=p.mime_type) for p in photos)]


def attach(llm_request: LlmRequest, parts: list[types.Part]) -> None:
    """Add parts to the shopper's message in the request, so the question and the photos stay together."""
    if not parts:
        return
    last = llm_request.contents[-1] if llm_request.contents else None
    if last is not None and last.role == "user":
        last.parts = list(last.parts or []) + parts
    else:
        llm_request.contents.append(types.Content(role="user", parts=parts))
