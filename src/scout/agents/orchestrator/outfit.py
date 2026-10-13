"""
OutfitPlanner: planning an outfit with the shopper, one of the Orchestrator's steps (kept in its own class so
the Orchestrator stays a router).

The Stylist plans the looks they asked for, each with its own pieces (their own clothes first, new ones from
the shop); the best product is found for every new piece and the board is shown. Then a piece can be changed,
added or removed in the looks they name (a few options to choose from), the new pieces put in the cart, or
the plan left. Planning is a mode: once started, every request is for the outfit until it is put in the cart
or the shopper asks for regular shopping.

Every event goes out through the Orchestrator that owns this planner (its name is on them), and the agents and
tools are the Orchestrator's own, so nothing is created here.
"""

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from scout.agents import state_keys
from scout.agents.elicitation import OutfitStep, TurnUnderstanding, read_turn
from scout.agents.narration import NarrationTask
from scout.domain import Change, Outfit, OutfitPlan, Piece, Product, ShoppingNeed
from scout.domain.sizes import SizeScale, read_size, size_scale
from scout.agents.model_io import content_text
from scout.tools import RetrievalResult
from scout.tools.taste_reader import describe

if TYPE_CHECKING:
    from scout.agents.orchestrator.agent import ScoutOrchestrator

logger = logging.getLogger(__name__)

STYLIST_TIMEOUT_S = 45    # a plan takes 6-20 s; waiting longer beats throwing a slow plan away
STYLIST_ATTEMPTS = 2      # a plan that fails is tried once more before Scout apologises


class OutfitPlanner:
    def __init__(self, scout: "ScoutOrchestrator", options_shown: int):
        self._scout = scout                    # the Orchestrator: its agents, tools and events
        self._options_shown = options_shown    # how many options are presented at a time


    def as_step(self, ctx: InvocationContext, turn: TurnUnderstanding) -> TurnUnderstanding:
        """In outfit mode what they ask for is for the outfit, until they finish or leave it. Elicitation says
        which step; this only covers what reads like regular shopping: clothes they own (plan again with them),
        other options for the piece being chosen, or a kind of product (a different one in the looks, or one
        added to them). Which piece is decided by its category, never by wording."""
        outfit = state_keys.outfit(ctx.session.state)
        if outfit is None or turn.intent == "outfit" or turn.question or turn.about_suggestion:
            return turn
        owned = {item.id for item in state_keys.wardrobe(ctx.session.state)}
        uses_removed = any(p.owned and p.owned not in owned for look in outfit.looks for p in look.pieces)
        if turn.owned_items or (turn.wardrobe_remove and uses_removed):   # their wardrobe changed: plan again
            return turn.model_copy(update={"intent": "outfit", "outfit": OutfitStep(action="plan")})
        changing = outfit.changing if outfit.changing and outfit.changing.looks else None
        wish = ", ".join(turn.need.must_have) or None
        if turn.intent in ("more_options", "reject") and changing:
            step = OutfitStep(action="change", category=changing.category, looks=[n + 1 for n in changing.looks],
                              wish=wish)
        elif turn.intent == "shop" and turn.need.category:
            same = changing is not None and changing.category.lower() == turn.need.category.lower()
            step = OutfitStep(action="change", category=turn.need.category, wish=wish,
                              looks=[n + 1 for n in changing.looks] if same else [])
        elif turn.intent in ("shop", "more_options", "reject"):
            step = OutfitStep(action="talk")         # nothing to search outside the outfit: talk about it
        else:
            return turn
        return turn.model_copy(update={"intent": "outfit", "outfit": step})

    def asks_where(self, ctx: InvocationContext, turn: TurnUnderstanding) -> bool:
        """While planning, a kind of product the outfit does not have (jeans, when it has trousers) may be for
        the outfit or something they want on its own: never guess, ask."""
        outfit = state_keys.outfit(ctx.session.state)
        if outfit is None or turn.intent != "shop" or not turn.need.category or turn.question:
            return False
        category = turn.need.category.lower()
        changing = outfit.changing.category.lower() if outfit.changing else None
        return category not in {c.lower() for c in outfit.categories()} and category != changing

    async def ask_where(self, ctx: InvocationContext, turn: TurnUnderstanding):
        outfit = state_keys.outfit(ctx.session.state)
        need = turn.need
        in_outfit = {c.lower(): c for c in outfit.categories()}
        replaces = next((in_outfit[c.lower()] for c in self._scout._neighbour_categories(need)
                         if c.lower() in in_outfit), None)
        kind = need.product_types[0] if need.product_types else need.category
        them = "them" if kind.endswith("s") else "it"
        place = f", in place of the {replaces}," if replaces else ""
        yield self._scout._update_state(ctx, {state_keys.PENDING_WHERE: {"need": need.model_dump(),
                                                                         "replaces": replaces}})
        async for item in self._ask(ctx, f"Should the {kind} go in your outfit{place} or are you shopping for "
                                         f"{them} separately?", ["In my outfit", "Shop separately"]):
            yield item

    @staticmethod
    def where_choice(ctx: InvocationContext, turn: TurnUnderstanding) -> Optional[str]:
        """Their answer: "outfit", "separate", or None when they moved on to something else."""
        text = content_text(ctx.user_content, separator=" ").lower()
        if re.search(r"separate|on (its|their) own|regular|by itself|by themselves|just (the|for)|not (in|for) the (outfit|look)", text):
            return "separate"
        if re.search(r"\b(outfit|look|in place|swap|replace|instead)\b", text) or turn.intent == "outfit":
            return "outfit"
        return None

    async def where_answered(self, ctx: InvocationContext, where: dict, choice: str):
        need = ShoppingNeed.model_validate(where["need"])
        if choice == "separate":                 # regular shopping (to the cart); the board stays on screen
            yield self._scout._update_state(ctx, {state_keys.MODE: None, state_keys.NEED: need.model_dump()})
            async for item in self._scout._search_and_narrate(ctx):
                yield item
            return
        outfit = state_keys.outfit(ctx.session.state)
        replaces = where.get("replaces")
        looks = (outfit.looks_with(replaces) if replaces else []) or list(range(len(outfit.looks)))
        outfit.changing = Change(category=need.category, looks=looks, wish=need.looking_for or "", replaces=replaces)
        place = f", in place of the {replaces}" if replaces else ""
        async for item in self._offer_options(ctx, outfit, f"They want {need.category} in the outfit{place}."):
            yield item

    async def sizes_confirmed(self, ctx: InvocationContext, turn: TurnUnderstanding):
        """Their answer to "add them in those sizes?": add them, change one size they named, or ask each again."""
        outfit = state_keys.outfit(ctx.session.state)
        yield self._scout._update_state(ctx, {state_keys.CONFIRM_SIZES: None})
        if outfit is None:
            return
        text = content_text(ctx.user_content, separator=" ").lower()
        known = dict(ctx.session.state.get(state_keys.KNOWN_SIZES) or {})
        scales = {s.key: s for s in map(size_scale, outfit.products_to_buy()) if s is not None}
        said = turn.size or text
        named = [(key, read_size(said, scale)) for key, scale in scales.items() if read_size(said, scale)]
        if len(named) == 1 and not re.search(r"\b(yes|add them|go ahead|sure|ok|okay|fine|perfect)\b", text):
            key, size = named[0]                 # "make the shoes a 10": that size, then check again
            yield self._scout._update_state(ctx, {state_keys.KNOWN_SIZES: {**known, key: size}})
            async for item in self._add_outfit_to_cart(ctx, outfit):
                yield item
            return
        if re.search(r"\b(change|different|other|no)\b", text):   # ask every size again
            yield self._scout._update_state(ctx, {state_keys.KNOWN_SIZES: {k: v for k, v in known.items()
                                                                           if k not in scales}})
            async for item in self._add_outfit_to_cart(ctx, outfit):
                yield item
            return
        async for item in self._add_outfit_to_cart(ctx, outfit, confirmed=True):
            yield item

    async def run(self, ctx: InvocationContext, turn: TurnUnderstanding):
        """Plan (or plan again), change, add or remove a piece in some looks, put the new pieces in the cart (the
        outfit is finished), leave for regular shopping, or talk about the plan."""
        outfit = state_keys.outfit(ctx.session.state)
        step = turn.outfit or OutfitStep(action="talk")
        if step.action == "plan":
            steps = self._plan_outfit(ctx)
        elif step.action == "leave":
            steps = self._leave_outfit(ctx)
        elif outfit is None:
            steps = self._scout._just_say(ctx, "We haven't planned an outfit yet. Tell me the occasion, and I'll plan it "
                                        "with you.")
        elif step.action in ("change", "remove"):
            steps = self._change_piece(ctx, outfit, step)
        elif step.action == "to_cart":
            steps = self._add_outfit_to_cart(ctx, outfit)
        else:
            steps = self._narrate_outfit(ctx, outfit, turn.understood or "They asked about the plan.")
        async for item in steps:
            yield item

    async def _plan_outfit(self, ctx: InvocationContext):
        """The Stylist plans the looks, then a product is found for every new piece. Planning takes a few
        seconds, so Scout says straight away that it is on it."""
        state = ctx.session.state
        yield self._scout._update_state(ctx, {state_keys.STYLIST_PLAN: None})   # never reuse an older plan
        own = " from your wardrobe" if state_keys.usable_wardrobe(state) else ""
        yield self._scout._say(ctx, f"Lovely! Give me a few seconds while I plan your looks{own}.")
        started = time.monotonic()
        for attempt in range(1, STYLIST_ATTEMPTS + 1):
            try:
                events = await asyncio.wait_for(self._scout._collect(self._scout._silently(self._scout.stylist, ctx)),
                                                STYLIST_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning("The Stylist took longer than %ss (attempt %d)", STYLIST_TIMEOUT_S, attempt)
                events = []
            except Exception:                    # a failed call: try again rather than give up at once
                logger.exception("The Stylist failed (attempt %d)", attempt)
                events = []
            for event in events:
                yield event
            raw = state.get(state_keys.STYLIST_PLAN)
            if raw and OutfitPlan.model_validate(raw).looks:
                break
        raw = state.get(state_keys.STYLIST_PLAN)
        plan = OutfitPlan.model_validate(raw) if raw else None
        need = state_keys.need(state)
        outfit = Outfit.from_plan(plan, state_keys.usable_wardrobe(state), need.budget_max) if plan else None
        if outfit is None or not outfit.looks:
            logger.warning("No outfit plan: %s", "the Stylist gave none" if plan is None else
                           f"{len(plan.looks)} looks planned, none usable")
            yield self._scout._say(ctx, "Sorry, I couldn't finish the plan just now. Shall I try again? You can also "
                                 "tell me a little more about the occasion.")
            return
        planned = time.monotonic()
        # Finding the pieces takes a while too: say what is planned now, so the shopper is never left in silence.
        count = len(outfit.looks)
        looks = "the look" if count == 1 else f"{count} looks"
        yield self._scout._say(ctx, f"I've planned {looks}. Now I'm finding each piece in the shop.")
        await asyncio.to_thread(self._find_products, outfit, need)
        logger.info("Outfit timing: stylist %.1fs, finding products %.1fs", planned - started,
                    time.monotonic() - planned)
        yield self._scout._update_state(ctx, {state_keys.PRESENTED: [], state_keys.FOCUS_PRODUCT: None})   # old cards
        async for item in self._narrate_outfit(ctx, outfit, state_keys.OUTFIT_PLANNED):
            yield item

    def _find_products(self, outfit: Outfit, need: ShoppingNeed) -> None:
        """A product for every new piece, all searched at the same time. Each piece is its own: a product already
        used in another look is not used again while the shop has another that fits. Within budget when
        possible; otherwise the nearest the shop has, which the total makes plain."""
        pieces = outfit.unfound()
        if not pieces:
            return
        share = round(outfit.budget / len(pieces), 2) if outfit.budget is not None else None

        def ranked(piece: Piece) -> tuple[ShoppingNeed, list[Product]]:
            piece_need = outfit.need_for_piece(piece, need, share)
            found = self._for_occasion(outfit, piece_need, self._search_piece(piece_need, set()).products, set())
            return piece_need, sorted(found, key=lambda p: not self._scout.retriever.fits(piece_need, p))   # stable

        with ThreadPoolExecutor(max_workers=min(len(pieces), 6)) as pool:
            results = list(pool.map(ranked, pieces))
        used: set[str] = set()
        for piece, (_, found) in zip(pieces, results):
            piece.product = next((p for p in found if p.id not in used), found[0] if found else None)
            if piece.product is not None:
                used.add(piece.product.id)

    async def _change_piece(self, ctx: InvocationContext, outfit: Outfit, step: OutfitStep):
        """Change, add or remove a piece in the looks they mean. When several looks could be meant and they did
        not say which, Scout asks first (with answers to tap)."""
        category = step.category or (outfit.changing.category if outfit.changing else None)
        if not category:
            names = " or the ".join(outfit.categories())
            yield self._scout._say(ctx, f"Which piece do you mean: the {names}?" if names else "Which piece do you mean?")
            return
        if step.action == "remove" and not outfit.looks_with(category):
            yield self._scout._say(ctx, f"None of the looks has a {category}. What would you like to change?")
            return
        looks = self._looks_meant(outfit, step, category)
        if looks is None:
            outfit.changing = Change(action=step.action, category=category, wish=step.wish or "")
            yield self._scout._update_state(ctx, {state_keys.OUTFIT: outfit.model_dump()})
            candidates = outfit.looks_with(category) or list(range(len(outfit.looks)))
            verb = ("take out the" if step.action == "remove" else
                    "change the" if outfit.looks_with(category) else "add a")
            async for item in self._ask(ctx, f"Should I {verb} {category} in every look, or just one?",
                                        ["All looks", *(f"Look {n + 1}" for n in candidates)]):
                yield item
            return
        if step.action == "remove":
            looks = [n for n in looks if n in outfit.looks_with(category)]
            if not looks:
                outfit.changing = None
                yield self._scout._update_state(ctx, {state_keys.OUTFIT: outfit.model_dump()})
                yield self._scout._say(ctx, f"That look doesn't have a {category}. What would you like to change?")
                return
            outfit.remove(category, looks)
            outfit.changing = None
            yield self._scout._update_state(ctx, {state_keys.PRESENTED: [], state_keys.FOCUS_PRODUCT: None})
            async for item in self._narrate_outfit(ctx, outfit, f"They took the {category} out of "
                                                              f"{Change(category=category, looks=looks).looks_text()}."):
                yield item
            return
        earlier = outfit.changing
        same = earlier is not None and earlier.looks == looks and earlier.category.lower() == category.lower()
        outfit.changing = Change(category=category, looks=looks,
                                 wish=step.wish or (earlier.wish if same else ""),
                                 price=self._price_given(ctx, outfit) or (earlier.price if same else (None, None)),
                                 shown=earlier.shown if same else [])
        adding = not set(looks) & set(outfit.looks_with(category))
        wish = f" ({outfit.changing.wish})" if outfit.changing.wish else ""
        event = (f"They asked to add a {category}{wish} to {outfit.changing.looks_text()}." if adding else
                 f"They asked for a different {category}{wish} in {outfit.changing.looks_text()}.")
        async for item in self._offer_options(ctx, outfit, event):
            yield item

    @staticmethod
    def _looks_meant(outfit: Outfit, step: OutfitStep, category: str) -> Optional[list[int]]:
        """Indexes of the looks they mean, or None when several could be meant and they did not say which."""
        everywhere = outfit.looks_with(category) or list(range(len(outfit.looks)))
        if step.all_looks:
            return everywhere
        named = [n - 1 for n in step.looks if 1 <= n <= len(outfit.looks)]
        if named:
            return list(dict.fromkeys(named))
        return everywhere if len(everywhere) == 1 else None

    @staticmethod
    def _price_given(ctx: InvocationContext, outfit: Outfit) -> Optional[tuple[Optional[float], Optional[float]]]:
        """A price they just gave while changing a piece is for that piece alone, not the whole outfit."""
        need = state_keys.need(ctx.session.state)
        if need.budget_min is not None or (need.budget_max is not None and need.budget_max != outfit.budget):
            return need.budget_min, need.budget_max
        return None

    async def _offer_options(self, ctx: InvocationContext, outfit: Outfit, event: str):
        """A few products for the piece being changed, never ones already offered for it. The shopping need
        becomes that piece, so the next "cheaper" or "in black" is about it."""
        change = outfit.changing
        base = state_keys.need(ctx.session.state)
        need = outfit.need_for(change, base)
        exclude = set(change.shown)
        result = await asyncio.to_thread(self._search_piece, need, exclude)
        ranked = await asyncio.to_thread(self._for_occasion, outfit, need, result.products, exclude)
        options = ranked[:self._options_shown]
        change.shown = list(dict.fromkeys([*change.shown, *(p.id for p in options)]))
        logger.info("Outfit %s for %s: searched %r, found %d", change.category, change.looks_text(),
                    need.search_text(), len(ranked))
        piece_need = base.model_copy(update={"category": change.category, "product_types": need.product_types,
                                             "looking_for": need.looking_for, "must_have": [],
                                             "budget_min": None, "budget_max": outfit.budget})
        yield self._scout._update_state(ctx, {
            state_keys.NEED: piece_need.model_dump(),
            state_keys.SHORTLIST: [p.model_dump() for p in ranked],
            state_keys.EXACT_MATCHES: result.exact_ids,
            state_keys.RELAXED: result.relaxed,
            state_keys.OTHER_TYPES: result.other_types,
            state_keys.PRESENTED: [p.id for p in options],
            state_keys.SEEN: [p.id for p in options],
            state_keys.FOCUS_PRODUCT: None,
        })
        async for item in self._narrate_outfit(ctx, outfit, event):
            yield item

    async def put_in(self, ctx: InvocationContext, outfit: Outfit, product: Product):
        """The product they chose goes into the looks being changed, and nowhere else."""
        change = outfit.changing
        outfit.put(product, change)
        outfit.changing = None
        yield self._scout._update_state(ctx, {state_keys.PRESENTED: [], state_keys.FOCUS_PRODUCT: None})
        event = f"They chose {' '.join(product.title.split()[:8])} for the {change.category} in {change.looks_text()}."
        async for item in self._narrate_outfit(ctx, outfit, event):
            yield item

    def _for_occasion(self, outfit: Outfit, need: ShoppingNeed, products: list[Product],
                      exclude: set[str]) -> list[Product]:
        """The products for a piece that suit the occasion, best first (search matches the piece, not the
        occasion). Ones that do not suit it are never offered: none at all is better than a ski jacket for the
        office. Why each suits it is kept for Scout to say."""
        if self._scout.occasion_ranker is None or not products:
            return products
        ranked, notes = self._scout._suited(outfit.occasion, need.category or "item", need.looking_for, need, products,
                                     exclude)
        if notes is None:                         # could not judge: the search order, as before
            return ranked
        outfit.fit_notes.update({pid: why for pid, why in notes.items() if why})
        return [p for p in ranked if p.id in notes]

    def _search_piece(self, need: ShoppingNeed, exclude: set[str]) -> RetrievalResult:
        """Search for an outfit piece. The Stylist may name kinds the shop has few of; if none fit, any
        product of that category is better than nothing."""
        result = self._scout.retriever.find(need, exclude)
        if not result.found and need.product_types:
            result = self._scout.retriever.find(need.model_copy(update={"product_types": []}), exclude)
        return result

    def stop_changing(self, ctx: InvocationContext) -> Optional[Event]:
        """They moved on to something else while a piece was being changed: the outfit stays as it is, but new
        options and choices are no longer for it."""
        outfit = state_keys.outfit(ctx.session.state)
        if outfit is None or outfit.changing is None:
            return None
        outfit.changing = None
        return self._scout._update_state(ctx, {state_keys.OUTFIT: outfit.model_dump()})

    async def _narrate_outfit(self, ctx: InvocationContext, outfit: Outfit, event: str):
        yield self._scout._update_state(ctx, {state_keys.OUTFIT: outfit.model_dump(), state_keys.OUTFIT_EVENT: event,
                                       state_keys.SUGGESTION: None,
                                       state_keys.NARRATION_TASK: NarrationTask.OUTFIT.value})
        async for item in self._scout.narration.run_async(ctx):
            yield item

    async def _ask(self, ctx: InvocationContext, question: str, choices: list[str]):
        """Ask a question from code, with answers to tap, so the next reply is read as its answer."""
        turn = read_turn(ctx.session.state)
        if turn is not None:
            asked = turn.model_copy(update={"question": question, "choices": choices, "ready_to_search": False})
            yield self._scout._update_state(ctx, {state_keys.TURN: asked.model_dump()})
        yield self._scout._update_state(ctx, {state_keys.LAST_QUESTION: question})
        yield self._scout._say(ctx, question)

    async def _add_outfit_to_cart(self, ctx: InvocationContext, outfit: Outfit, follow_up: bool = False,
                                  confirmed: bool = False):
        """The new pieces go in the cart and the outfit is finished: back to regular shopping. The board stays,
        so they can still try the looks on. Sizes are asked here, once per kind of size."""
        products = outfit.products_to_buy()
        if not products:
            yield self._scout._say(ctx, "Good news: these looks don't need anything new. You already own every piece!")
            return
        cart = state_keys.cart(ctx.session.state)
        known = ctx.session.state.get(state_keys.KNOWN_SIZES) or {}
        new = [p for p in products if all(item.product_id != p.id for item in cart.items)]
        missing = next((scale for scale in map(size_scale, new)
                        if scale is not None and known.get(scale.key) not in scale.options), None)
        if missing is not None:
            kinds = list(dict.fromkeys(p.kind or "item" for p in new
                                       if (size_scale(p) or missing).key == missing.key and size_scale(p)))
            names = " and the ".join([", the ".join(kinds[:-1]), kinds[-1]]) if len(kinds) > 1 else kinds[0]
            yield self._scout._update_state(ctx, {
                state_keys.PENDING_SIZE: {"outfit": True, "scale": missing.model_dump(), "kinds": names},
                state_keys.SIZE_CHOICES: missing.options,
            })
            opening = "Got it. And" if follow_up else "Lovely! Before I add the outfit to your cart:"
            yield self._scout._say(ctx, f"{opening} what size would you like for the {names}? "
                                 f"They come in {missing.spoken}.")
            return
        sized = [(p, size_scale(p)) for p in new if size_scale(p) is not None]
        if sized and not confirmed:                      # read every size back once before adding
            kinds = list(dict.fromkeys(f"the {p.kind or 'item'} in {scale.label(known[scale.key])}" for p, scale in sized))
            summary = ", ".join(kinds[:-1]) + f" and {kinds[-1]}" if len(kinds) > 1 else kinds[0]
            yield self._scout._update_state(ctx, {state_keys.CONFIRM_SIZES: True})
            async for item in self._ask(ctx, f"Before I add them: {summary}. Shall I add them in those sizes, or "
                                             f"would you like to change a size?", ["Add them", "Change sizes"]):
                yield item
            return
        added = []
        for product in products:
            scale = size_scale(product)
            label = scale.label(known[scale.key]) if scale is not None else None
            if cart.add(product, label):
                added.append(f"{product.title}{' (size ' + label + ')' if label else ''}")
        outfit.changing = None
        wishlist = state_keys.wishlist(ctx.session.state)
        wishlist.remove([product.id for product in products])
        yield self._scout._update_state(ctx, {
            **self._scout._learn_from(ctx, "bought", [(describe(p), p.gender) for p in new]),
            state_keys.CART: cart.model_dump(),
            state_keys.WISHLIST: wishlist.model_dump(),
            state_keys.CART_EVENT: {"action": "added" if added else "already_in_cart", "items": added,
                                    "outfit_finished": True},
            state_keys.NARRATION_TASK: NarrationTask.CART_UPDATE.value,
            state_keys.SUGGESTION: None,
            state_keys.OUTFIT: outfit.model_dump(),
            state_keys.PRESENTED: [],
            state_keys.MODE: None,
            state_keys.SET_ASIDE: [],          # set aside for that outfit only
            state_keys.LOOK_COUNT: None,
            state_keys.PENDING_SIZE: None,
            state_keys.SIZE_CHOICES: None,
        })
        async for event in self._scout.narration.run_async(ctx):
            yield event

    async def size_given(self, ctx: InvocationContext, turn: TurnUnderstanding, pending: dict):
        """They gave a size while the outfit waits to go in the cart: remember it, then carry on adding (which
        asks for the next kind of size, if one is still missing)."""
        scale = SizeScale.model_validate(pending["scale"])
        size = read_size(turn.size or "", scale)
        if size is None:
            yield self._scout._say(ctx, f"Sorry, I didn't catch a size I know for the {pending.get('kinds', 'outfit')}. "
                                 f"They come in {scale.spoken}.")
            return
        known = {**(ctx.session.state.get(state_keys.KNOWN_SIZES) or {}), scale.key: size}
        yield self._scout._update_state(ctx, {state_keys.KNOWN_SIZES: known, state_keys.PENDING_SIZE: None})
        outfit = state_keys.outfit(ctx.session.state)
        if outfit is None:
            yield self._scout._say(ctx, "Sorry, I lost track of the outfit. Shall we plan it again?")
            return
        async for item in self._add_outfit_to_cart(ctx, outfit, follow_up=True):
            yield item

    async def _leave_outfit(self, ctx: InvocationContext):
        yield self._scout._update_state(ctx, {state_keys.OUTFIT: None, state_keys.PRESENTED: [], state_keys.MODE: None,
                                       state_keys.SET_ASIDE: [], state_keys.LOOK_COUNT: None})
        yield self._scout._say(ctx, "Sure, we'll leave the plan there. What would you like to shop for?")
