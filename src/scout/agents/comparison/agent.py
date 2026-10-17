"""
Comparison agent: Scout's analyst.

It compares two or more presented products for THIS shopper's need and
returns a structured ComparisonResult : the differences
that would change their choice, each product's strengths and drawbacks with
the source of every fact and a recommendation only when one is clearly better.

It sees each product's full facts plus one photo per product, so visual
differences are judged from the pictures, not guessed.
"""

import logging
from typing import Literal, Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.planners import BuiltInPlanner
from google.genai import types
from pydantic import BaseModel, Field

from scout.agents import state_keys
from scout.agents.model_io import attach, photo_parts
from scout.config import Settings
from scout.domain import Product
from scout.tools import ImageFetcher

logger = logging.getLogger(__name__)

Source = Literal["image", "seller", "reviews", "interpretation"]


class Fact(BaseModel):
    text: str = Field(description="One short fact, in plain words")
    source: Source = Field(description="Where it comes from: photo, seller listing, reviews, or your interpretation")


class ProductAssessment(BaseModel):
    position: int = Field(description="The option number as presented to the shopper")
    spoken_name: str = Field(description="Short name from its type and most distinctive feature")
    fit_for_need: str = Field(description="One sentence: how well it suits this shopper's need, and why")
    strengths: list[Fact] = Field(default_factory=list)
    drawbacks: list[Fact] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    deciding_aspects: list[str] = Field(description="The 2 or 3 aspects that matter most for this shopper's choice")
    key_differences: list[str] = Field(description="The 2 or 3 differences between the products that would change the choice")
    assessments: list[ProductAssessment]
    recommended_position: Optional[int] = Field(
        default=None, description="Option number of the clearly better product for this need; null if it depends")
    recommendation_reason: Optional[str] = Field(default=None, description="Why it is better for this need")
    depends_on: Optional[str] = Field(
        default=None, description="If no clear winner: the one thing about the shopper that would decide it")


INSTRUCTION = """You are Scout's comparison expert, helping a blind or low-vision shopper decide between products. Think carefully: your analysis is what lets them choose without seeing the products.

WHAT THE SHOPPER WANTS
{need}

WHAT YOU KNOW ABOUT THEIR TASTE
{profile}

PRODUCTS TO COMPARE (photos attached, one per product, in the same order)
{products}

HOW TO COMPARE
- Judge everything against THIS shopper's need (purpose, must-haves, things to avoid, budget) and their known taste. Stated preferences matter more than learned ones.
- Focus on differences that would change their choice. Ignore what the products have in common.
- Use the photos for anything visual. Use review points for real-world experience such as fit, comfort and durability.
- Use only the facts and photos given. Tag every fact with its source. Never invent features, prices or opinions.
- A product that misses a must-have, includes something they want to avoid, or is over budget must have that as a drawback.
- Recommend one product only if it is clearly better for this need. If it genuinely depends on something you do not know about the shopper, leave the recommendation empty and say what it depends on.
"""


class ComparisonAgentBuilder:
    """Builds the Comparison LlmAgent; photos are attached before each model call."""

    def __init__(self, settings: Settings, image_fetcher: ImageFetcher, images_per_product: int = 1):
        self._settings = settings
        self._images = image_fetcher
        self._images_per_product = images_per_product

    def build(self) -> LlmAgent:
        tuning = self._settings.tuning("comparison")
        return LlmAgent(
            name="comparison",
            model=self._settings.llm("comparison"),
            description="Compares presented products for the shopper's need and returns a structured analysis.",
            instruction=self._instruction,
            include_contents="none",
            output_schema=ComparisonResult,
            output_key=state_keys.COMPARISON,
            generate_content_config=types.GenerateContentConfig(temperature=tuning.temperature),
            planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(
                thinking_budget=tuning.thinking_budget)),
            before_model_callback=self._attach_photos,
        )

    def _instruction(self, context: ReadonlyContext) -> str:
        state = context.state
        products_text = "\n\n".join(
            f"Option {position}:\n{product.to_prompt_text(detailed=True)}"
            for position, product in compared_products(state)
        )
        return INSTRUCTION.format(need=state_keys.need(state).summary(),
                                  profile=state_keys.profile(state).to_prompt_text(),
                                  products=products_text or "None.")

    def _attach_photos(self, callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        parts: list[types.Part] = []
        for position, product in compared_products(callback_context.state):
            parts += photo_parts(self._images, product.image_urls, self._images_per_product,
                                 f"Photo of option {position}:")
        attach(llm_request, parts)
        return None


def compared_products(state) -> list[tuple[int, Product]]:
    """(option number as presented, product) for the products chosen for comparison."""
    shortlist = state_keys.shortlist(state)
    presented = state.get(state_keys.PRESENTED) or []
    result = []
    for product_id in state.get(state_keys.COMPARE_IDS) or []:
        if product_id in shortlist:
            position = presented.index(product_id) + 1 if product_id in presented else len(result) + 1
            result.append((position, shortlist[product_id]))
    return result
