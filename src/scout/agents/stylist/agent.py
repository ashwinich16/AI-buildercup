"""
Stylist agent: plans an outfit for an occasion ("my first day at the office").

It returns an OutfitPlan: the pieces the shopper will choose, one at a time,
and up to two extras that would finish the look (offered once at the end,
never required). When the shopper named the pieces they want, those are the
pieces; anything else is only an extra. Every piece is one of the shop's own
categories, so the plan never asks for something the shop cannot show.

The Orchestrator then finds real products for each piece, matched to what
the shopper has already picked.
"""

import logging
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmResponse
from google.adk.planners import BuiltInPlanner
from google.genai import types

from scout.agents import state_keys
from scout.agents.elicitation import read_turn
from scout.agents.model_io import response_json
from scout.catalog import CatalogRepository
from scout.config import Settings
from scout.domain import OutfitPiece, OutfitPlan

logger = logging.getLogger(__name__)

INSTRUCTION = """You are Scout's stylist: a friend with a great eye who helps a shopper put together an outfit for an occasion.

THE SHOP'S CATEGORIES: {categories}
THE SHOP'S PRODUCT TYPES: {product_types}

THE SHOPPER
Occasion: {occasion}
Whose style: {gender}
Budget for the whole outfit: {budget}
Pieces they asked for: {asked_for}
What you know about their taste: {profile}

Plan the outfit:
- pieces: the pieces they will choose, one at a time, most important first. If they asked for particular pieces, use exactly those, in the order they asked. Otherwise pick the two or three pieces that make the outfit for this occasion.
- extras: up to two more pieces that would finish the look, which Scout will offer at the end. Never repeat a piece already in pieces.
- Every piece must be one of THE SHOP'S CATEGORIES, spelled exactly. Use product_types only from THE SHOP'S PRODUCT TYPES, and only when the occasion calls for particular kinds.
- looks_like: what to look for in that piece, in plain words a product description would use, so the pieces go together (colours, fit, material, how formal).
- why: one short reason it suits the occasion.
- Respect their taste and budget. Never invent categories the shop does not have.
"""


class StylistAgentBuilder:
    """Builds the Stylist LlmAgent. Dependencies are passed in, never created here."""

    def __init__(self, settings: Settings, catalog: CatalogRepository):
        self._settings = settings
        self._catalog = catalog

    def build(self) -> LlmAgent:
        tuning = self._settings.tuning("stylist")
        return LlmAgent(
            name="stylist",
            model=self._settings.model_for("stylist"),
            description="Plans an outfit for an occasion from the shop's own categories.",
            instruction=self._instruction,
            include_contents="none",
            output_schema=OutfitPlan,
            generate_content_config=types.GenerateContentConfig(temperature=tuning.temperature),
            planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=tuning.thinking_budget)),
            after_model_callback=self._keep_shop_pieces,
        )

    def _instruction(self, context: ReadonlyContext) -> str:
        state = context.state
        need = state_keys.need(state)
        turn = read_turn(state)
        return INSTRUCTION.format(
            categories=", ".join(self._catalog.categories()) or "unknown",
            product_types=", ".join(self._catalog.product_types()) or "unknown",
            occasion=need.purpose or need.looking_for or "not said",
            gender=need.gender or "not said",
            budget=f"${need.budget_max:.0f}" if need.budget_max is not None else "not given",
            asked_for=", ".join(turn.outfit_pieces) if turn and turn.outfit_pieces else "none in particular",
            profile=state_keys.profile(state).to_prompt_text(),
        )

    # ADK calls this after the model answers: keep only pieces the shop really has, in its spelling.
    def _keep_shop_pieces(self, callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
        plan = response_json(llm_response, OutfitPlan)
        if plan is None:
            return None
        plan.pieces = self._in_shop(plan.pieces)
        plan.extras = [p for p in self._in_shop(plan.extras) if p.category not in {q.category for q in plan.pieces}]
        callback_context.state[state_keys.STYLIST_PLAN] = plan.model_dump()
        logger.info("Outfit for %r: %s (+ extras %s)", plan.occasion, [p.category for p in plan.pieces],
                    [p.category for p in plan.extras])
        return None

    def _in_shop(self, pieces: list[OutfitPiece]) -> list[OutfitPiece]:
        categories = {c.lower(): c for c in self._catalog.categories()}
        kinds = {t.lower(): t for t in self._catalog.product_types()}
        kept = []
        for piece in pieces:
            if piece.category.lower() in categories:
                piece.category = categories[piece.category.lower()]
                piece.product_types = [kinds[t.lower()] for t in piece.product_types if t.lower() in kinds]
                kept.append(piece)
        return kept
