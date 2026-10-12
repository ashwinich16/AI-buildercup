"""
Product Expert agent: looks closely at ONE product and talks about it.

It sees the product's photos (attached before each model call) and all its
stored facts, then either describes it so the shopper can picture it, or
answers a specific question ("does it run small?"). It speaks in Scout's
shared voice, so no separate Narration step is needed for these answers.

Which product to talk about is decided before this agent runs: the
Orchestrator resolves "the second one" and stores the id in session state.
"""

import logging
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.planners import BuiltInPlanner
from google.genai import types

from scout.agents import state_keys
from scout.agents.model_io import attach, photo_parts
from scout.agents.voice import VOICE
from scout.config import Settings
from scout.tools import ImageFetcher

logger = logging.getLogger(__name__)

PRODUCT_EXPERT = """{voice}
RIGHT NOW you are looking closely at ONE product the shopper asked about: {position}.
You can see its photos (attached) and you have the facts below. You CAN see the photos, so never say you cannot see images.

THE PRODUCT
{product}

WHAT THE SHOPPER WANTS OVERALL
{need}

HOW TO ANSWER
- If they ask to describe it, what it looks like, or about the picture, describe it so they can picture it, in this order:
  overall impression; colour and pattern in everyday comparisons; material and texture, how it would feel to touch;
  shape and notable details; how it wears or is used (reviewers' experience); anything to watch out for.
  About 90 to 130 words.
- If they ask something specific about it, answer that directly in one to three sentences.
- Use the photos for anything visual. When unsure, make clear it is based on how the photo looks. If the photo and the listing disagree, say so kindly.
- If neither the photos nor the facts answer the question, say honestly that the listing doesn't say. Never guess.
- Connect what you say to what they want, when it helps them decide.
- End with one easy, useful next step (such as comparing it with another option or choosing it).
"""

_POSITIONS = ["the first option you mentioned", "the second option you mentioned", "the third option you mentioned"]


class ProductExpertAgentBuilder:
    """Builds the Product Expert LlmAgent with photos attached to every call."""

    def __init__(self, settings: Settings, image_fetcher: ImageFetcher, images_per_product: int = 2):
        self._settings = settings
        self._images = image_fetcher
        self._images_per_product = images_per_product

    def build(self) -> LlmAgent:
        tuning = self._settings.tuning("product_expert")
        return LlmAgent(
            name="product_expert",
            model=self._settings.llm("product_expert"),
            description="Looks at one product's photos and facts; describes it or answers questions about it.",
            instruction=self._instruction,
            include_contents="none",            # the product and the current question are all it needs
            generate_content_config=types.GenerateContentConfig(temperature=tuning.temperature),
            planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(
                thinking_budget=tuning.thinking_budget)),
            before_model_callback=self._attach_photos,
        )

    def _instruction(self, context: ReadonlyContext) -> str:
        state = dict(context.state)
        product = state_keys.focus(state)
        if product is None:
            return VOICE + "\nThe shopper asked about a product, but it is not clear which one. Ask kindly which option they mean."

        presented = state.get(state_keys.PRESENTED) or []
        position = _POSITIONS[presented.index(product.id)] if product.id in presented[:3] else "one of the options"
        return PRODUCT_EXPERT.format(
            voice=VOICE,
            position=position,
            product=product.to_prompt_text(detailed=True),
            need=state_keys.need(state).summary(),
        )

    # ADK calls this right before the model call: add the product photos to the request.
    def _attach_photos(self, callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        product = state_keys.focus(callback_context.state)
        if product is None or not product.image_urls:
            return None
        parts = photo_parts(self._images, product.image_urls, self._images_per_product, "Photos of the product:")
        if not parts:
            logger.warning("No photos could be loaded for %s; answering from facts only.", product.id)
        attach(llm_request, parts)
        return None
