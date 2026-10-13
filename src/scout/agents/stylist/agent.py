"""
Stylist agent: plans outfits for an occasion from the shopper's own wardrobe and the shop's categories.

It returns an OutfitPlan in one go (one call, text only: photos were read when they were added): the looks
they asked for, each a complete outfit with its own pieces, every piece either one they own (by its id) or a
new one described in words, plus their own pieces that do not suit the occasion. Every new piece is one of the
shop's categories, so the plan never asks for something the shop cannot show. The Orchestrator then finds a
real product for each new piece.
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
from scout.agents.model_io import response_json
from scout.catalog import CatalogRepository, SearchQuery
from scout.config import Settings
from scout.domain import OutfitPlan

logger = logging.getLogger(__name__)

INSTRUCTION = """You are Scout's stylist: a friend with a great eye who plans outfits with the shopper, and who would rather they wear what they already own than buy more.

THE SHOP'S CATEGORIES: {categories}
THE SHOP'S PRODUCT TYPES: {product_types}

WHAT THE SHOP HAS IN THIS STYLE (category: kinds, with how many of each)
{stock}

THE SHOPPER
The occasion: {occasion}
How many looks: {look_count}
Their wishes for the looks: {wishes}
Whose style: {gender}
Budget for new pieces: {budget}
What you know about their taste: {profile}
Their taste, learned from what they own and buy (a leaning, never a rule): {taste}

THEIR WARDROBE (id: what it is)
{wardrobe}

Plan it:
- First think about the occasion itself: the likely weather, and what it expects (a workplace dress code, a religious place, a beach, long walks, how dressed up an evening out is).
- looks: exactly the number in How many looks when it gives one; otherwise one look per day or occasion they mentioned. Each look is a complete outfit for that moment.
- Every piece is different: never put the same piece in two looks, whether it is one of theirs or a new one (two looks for the same office are two different outfits, each with its own pieces).
- Each piece of a look is either one of THEIR WARDROBE (put its id in owned) or a new piece (owned empty; category, product_types and looks_like filled). Use an owned piece only when it really suits both the occasion and whose style the outfit is (a casual polo is not part of an office look; a piece made for the other style does not belong); otherwise plan a new piece.
- leave_at_home: their own pieces that do not suit this occasion or this style, each with one short, kind reason. Never use them in a look.
- A new piece's category must be one of THE SHOP'S CATEGORIES, spelled exactly; product_types only from THE SHOP'S PRODUCT TYPES, and only when the moment calls for particular kinds. Pick the category where the right kind lives in WHAT THE SHOP HAS IN THIS STYLE. If the shop has nothing that suits the occasion for a piece, leave that piece out and say in the look's why what the shop could not provide.
- looks_like: what to look for, in plain words a product description would use (colour, fit, material, how formal), so it goes with the rest of that look.
- The occasion comes first. Their taste shapes the choice only within what the occasion needs (liking soft things means a soft dress shirt for an office, never a gym top; avoiding a material means pieces without it). If the outfit is for a different style from what they usually wear, or for someone else, leave their personal taste out.
- Respect their budget. Never invent owned pieces or categories.
"""


class StylistAgentBuilder:
    """Builds the Stylist LlmAgent. Dependencies are passed in, never created here."""

    def __init__(self, settings: Settings, catalog: CatalogRepository):
        self._settings = settings
        self._catalog = catalog
        self._stock_cache: dict[Optional[str], str] = {}
        self._kinds_in_stock: dict[Optional[str], dict[str, set[str]]] = {}   # style -> category -> kinds

    def build(self) -> LlmAgent:
        tuning = self._settings.tuning("stylist")
        return LlmAgent(
            name="stylist",
            model=self._settings.llm("stylist"),
            description="Plans outfits for an occasion from the shopper's wardrobe and the shop's categories.",
            instruction=self._instruction,
            include_contents="none",
            output_schema=OutfitPlan,
            generate_content_config=types.GenerateContentConfig(temperature=tuning.temperature),
            planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=tuning.thinking_budget)),
            after_model_callback=self._keep_real_pieces,
        )

    def _instruction(self, context: ReadonlyContext) -> str:
        state = context.state
        need = state_keys.need(state)
        wardrobe = state_keys.usable_wardrobe(state)     # without the pieces they set aside for now
        return INSTRUCTION.format(
            stock=self._stock(need.gender_filter()),
            categories=", ".join(self._catalog.categories()) or "unknown",
            product_types=", ".join(self._catalog.product_types()) or "unknown",
            occasion=need.purpose or "not said",
            look_count=(f"exactly {count}" if (count := state.get(state_keys.LOOK_COUNT))
                        else "not said: one per day or occasion"),
            wishes=need.looking_for or "none",
            gender=need.gender or "not said",
            budget=f"${need.budget_max:.0f}" if need.budget_max is not None else "not given",
            profile=state_keys.profile(state).to_prompt_text(),
            taste=state_keys.taste_text(state),
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
                self._kinds_in_stock.setdefault(gender, {})[category.lower()] = {k.lower() for k, _ in kinds}
                lines.append(f"{category}: " + (", ".join(f"{kind} {n}" for kind, n in kinds) or "nothing"))
            self._stock_cache[gender] = "\n".join(lines) or "not available"
        return self._stock_cache[gender]

    # ADK calls this after the model answers: keep only what the shop and the wardrobe really have.
    def _keep_real_pieces(self, callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
        plan = None if llm_response.partial else response_json(llm_response, OutfitPlan)
        if plan is None:
            return None
        state = callback_context.state
        categories = {c.lower(): c for c in self._catalog.categories()}
        kinds = {t.lower(): t for t in self._catalog.product_types()}
        gender = state_keys.need(state).gender_filter()
        self._stock(gender)
        in_stock = self._kinds_in_stock.get(gender, {})
        owned = {item.id for item in state_keys.usable_wardrobe(state)}
        left = {item.owned for item in plan.leave_at_home}
        for look in plan.looks:
            kept = []
            for piece in look.pieces:
                if piece.owned:
                    if piece.owned in owned and piece.owned not in left:
                        kept.append(piece)
                    continue
                if piece.category.lower() not in categories:
                    continue
                piece.category = categories[piece.category.lower()]
                piece.product_types = [kinds[t.lower()] for t in piece.product_types if t.lower() in kinds]
                # A kind the shop does not have in this style (a men's blazer) is left out, never filled with
                # something unsuitable: the look is fine without it.
                stocked = in_stock.get(piece.category.lower(), set())
                if piece.product_types and in_stock and not {t.lower() for t in piece.product_types} & stocked:
                    logger.info("No %s in this style; leaving the %s out", piece.product_types, piece.category)
                    continue
                kept.append(piece)
            look.pieces = kept
        planned = len(plan.looks)
        plan.looks = [look for look in plan.looks if look.pieces]
        wanted = state.get(state_keys.LOOK_COUNT)
        if (wanted and len(plan.looks) != wanted) or len(plan.looks) < planned:
            logger.warning("Looks: %s asked, %d planned, %d kept", wanted or "not said", planned, len(plan.looks))
        plan.leave_at_home = [item for item in plan.leave_at_home if item.owned in owned]
        state[state_keys.STYLIST_PLAN] = plan.model_dump()
        logger.info("Plan for %r: %d looks, pieces per look %s", plan.occasion, len(plan.looks),
                    [len(look.pieces) for look in plan.looks])
        return None
