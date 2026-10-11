"""
Stylist agent: plans outfits for a trip, a week or an event, from the shopper's own wardrobe.

It returns a CapsulePlan in one go (one call, text only: the photos were read when they were added): one
look per day or occasion, made of pieces the shopper already owns wherever possible, the owned pieces that
do not suit the trip (with a kind reason), and only the few new pieces worth buying, each chosen to be worn in as many looks as
possible. Every new piece is one of the shop's own categories, so the plan never asks for something the
shop cannot show. The Orchestrator then finds a real product for each new piece.
"""

import logging
from collections import Counter
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
from scout.catalog import CatalogRepository, SearchQuery
from scout.config import Settings
from scout.domain import CapsulePlan

logger = logging.getLogger(__name__)

INSTRUCTION = """You are Scout's stylist: a friend with a great eye who plans outfits with the shopper, and who would rather they wear what they already own than buy more.

THE SHOP'S CATEGORIES: {categories}
THE SHOP'S PRODUCT TYPES: {product_types}

WHAT THE SHOP HAS IN THIS STYLE (category: kinds, with how many of each)
{stock}

THE SHOPPER
What it is for: {occasion}
What they will do there: {activities}
Their wishes for the looks: {wishes}
Whose style: {gender}
Budget for new pieces: {budget}
What you know about their taste: {profile}

THEIR WARDROBE (id: what it is)
{wardrobe}

Plan it:
- First think about the occasion itself: the likely weather, and what each place or activity expects (a workplace dress code, a religious place, a beach, long walks, how dressed up an evening out is).
- leave_at_home: the owned pieces that do not suit this occasion, each with one short, kind reason. Never use them in a look.
- looks: one look per day or occasion they mentioned (one look if they named a single event), each a complete outfit for what they do that day. Build every look from THEIR WARDROBE first (refer to an owned piece by its id in owned).
- new_pieces: only what their wardrobe is missing, and as few as possible. Prefer pieces that work in several looks, and say in why how many looks each one is in. Use a new piece in a look by its key in new. If their wardrobe already covers everything, new_pieces is empty.
- A new piece's category must be one of THE SHOP'S CATEGORIES, spelled exactly; product_types only from THE SHOP'S PRODUCT TYPES, and only when the moment calls for particular kinds.
- Plan only new pieces the shop can really provide for THIS occasion: check WHAT THE SHOP HAS IN THIS STYLE. Pick the category where the right kind lives (a collared shirt for an office is in "shirt", not "top"). If the shop has nothing that suits the occasion for a piece (no blazer for a business look), leave that piece out of new_pieces rather than plan something unsuitable, and say in the look's why what the shop could not provide.
- The occasion comes first. Their taste shapes the choice only within what the occasion needs: liking soft, comfortable things means a soft dress shirt for an office, never a thermal or gym top; avoiding a material means pieces without it. Their taste is about things for themselves: if the outfit is in a different style from what they usually wear, or for someone else, plan for the occasion and leave their personal taste out.
- looks_like: what to look for, in plain words a product description would use, so it goes with the owned pieces it is worn with (colour, fit, material, how formal).
- Respect their taste and budget. Never invent owned pieces or categories.
"""


class StylistAgentBuilder:
    """Builds the Stylist LlmAgent. Dependencies are passed in, never created here."""

    def __init__(self, settings: Settings, catalog: CatalogRepository):
        self._settings = settings
        self._catalog = catalog
        self._stock_cache: dict[Optional[str], str] = {}

    def build(self) -> LlmAgent:
        tuning = self._settings.tuning("stylist")
        return LlmAgent(
            name="stylist",
            model=self._settings.llm("stylist"),
            description="Plans outfits for any occasion from the shopper's wardrobe and the shop's categories.",
            instruction=self._instruction,
            include_contents="none",
            output_schema=CapsulePlan,
            generate_content_config=types.GenerateContentConfig(temperature=tuning.temperature),
            planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=tuning.thinking_budget)),
            after_model_callback=self._keep_real_pieces,
        )

    def _instruction(self, context: ReadonlyContext) -> str:
        state = context.state
        need = state_keys.need(state)
        wardrobe = state_keys.wardrobe(state)
        turn = read_turn(state)
        return INSTRUCTION.format(
            stock=self._stock(need.gender_filter()),
            categories=", ".join(self._catalog.categories()) or "unknown",
            product_types=", ".join(self._catalog.product_types()) or "unknown",
            occasion=need.purpose or "not said",
            activities=", ".join(turn.activities if turn else []) or "as above",
            wishes=need.looking_for or "none",
            gender=need.gender or "not said",
            budget=f"${need.budget_max:.0f}" if need.budget_max is not None else "not given",
            profile=state_keys.profile(state).to_prompt_text(),
            wardrobe="\n".join(f"{item.id}: {item.description} ({item.category})" for item in wardrobe)
                     or "Nothing added yet: every piece is new.",
        )

    def _stock(self, gender: Optional[str]) -> str:
        """Each category's kinds for this style, with counts, so the plan never relies on what is not there."""
        if gender not in self._stock_cache:
            lines = []
            for category in self._catalog.categories():
                try:
                    found = self._catalog.search(SearchQuery(category=category, gender=gender, limit=50))
                except Exception as error:
                    logger.warning("Could not list %s for the stylist: %s", category, error)
                    continue
                kinds = Counter(p.product_type or category for p in found).most_common()
                lines.append(f"{category}: " + (", ".join(f"{kind} {n}" for kind, n in kinds) or "nothing"))
            self._stock_cache[gender] = "\n".join(lines) or "not available"
        return self._stock_cache[gender]

    # ADK calls this after the model answers: keep only what the shop and the wardrobe really have.
    def _keep_real_pieces(self, callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
        plan = None if llm_response.partial else response_json(llm_response, CapsulePlan)
        if plan is None:
            return None
        categories = {c.lower(): c for c in self._catalog.categories()}
        kinds = {t.lower(): t for t in self._catalog.product_types()}
        plan.new_pieces = [p for p in plan.new_pieces if p.category.lower() in categories]
        for piece in plan.new_pieces:
            piece.category = categories[piece.category.lower()]
            piece.product_types = [kinds[t.lower()] for t in piece.product_types if t.lower() in kinds]
        owned = {item.id for item in state_keys.wardrobe(callback_context.state)}
        plan.leave_at_home = [left for left in plan.leave_at_home if left.owned in owned]
        packed = owned - {left.owned for left in plan.leave_at_home}
        new = {piece.key for piece in plan.new_pieces}
        for look in plan.looks:
            look.pieces = [p for p in look.pieces if p.owned in packed or p.new in new]
        plan.looks = [look for look in plan.looks if look.pieces]
        worn = {p.new for look in plan.looks for p in look.pieces}
        plan.new_pieces = [p for p in plan.new_pieces if p.key in worn]      # never buy a piece no look uses
        callback_context.state[state_keys.STYLIST_PLAN] = plan.model_dump()
        logger.info("Capsule for %r: %d looks, %d left at home, new pieces %s", plan.occasion, len(plan.looks),
                    len(plan.leave_at_home), [p.category for p in plan.new_pieces])
        return None
