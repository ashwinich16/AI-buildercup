"""
Orchestrator: Scout's traffic controller. A custom ADK agent with NO LLM call.

Each turn:
  0. while still working out the need, summarise what the catalog has that is
     close to the shopper's words (one search, no LLM) so questions stay realistic
  1. run the Elicitation agent (it updates the need and says what the shopper wants)
  2. route in plain Python:
       need not clear yet    -> say Elicitation's friendly question
       need clear            -> Retrieval, then Narration presents the options
       reject                -> remember why, search again, Narration presents
       more options          -> next unheard products (searching again if needed), Narration
       about a product       -> resolve which one, Product Expert looks and answers
       compare               -> Comparison analyses, Narration speaks it
       choose                -> add to cart, suggest one item that goes with it, Narration confirms
       cart                  -> view / remove / checkout (with one last "you might have forgotten"), Narration
       see suggestion        -> present the suggested item with a few like it
       outfit                -> Stylist plans one look per day or occasion from the clothes they own, and the
                                few new pieces worth buying, shown as a board. If they own pieces for it, the
                                new ones are all found at once; if everything is new, they choose each piece
                                from a few options, one at a time (guided). Then swap or skip a piece, add the
                                new pieces to the cart, or leave. Planning is a mode: once started, every
                                request is for the outfit (another kind of product is added to it, a price
                                or "others" is for the piece being chosen) until the outfit is put in the
                                cart or the shopper asks for regular shopping
       pause                 -> say nothing; the UI goes quiet and keeps listening
       end_conversation      -> a short goodbye; the UI stops listening
  3. AFTER the reply: the Preference agent learns lasting taste when there is
     something to learn (a choice, a rejection, or a stated preference)

Routing in code instead of a model is faster (no extra Gemini call per turn)
and fully predictable, which makes it easy to test.

Agents and tools are passed in (Dependency Inversion); this class never
creates them.
"""

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types
from pydantic import ConfigDict

from scout.agents import state_keys
from scout.agents.elicitation import TurnUnderstanding, read_turn
from scout.agents.model_io import content_text
from scout.agents.narration import NarrationTask
from scout.domain import Capsule, CapsulePlan, NewPiece, Product, ShoppingNeed
from scout.tools import CatalogOverviewBuilder, ProductReferenceResolver, ProductRetriever, RetrievalResult
from scout.tools.recommendations import Recommender, Suggestion, SuggestionRules
from scout.tools.wardrobe import new_item

logger = logging.getLogger(__name__)

PRODUCTS_TO_PRESENT = 3   # listeners can keep about three options in mind
STYLIST_TIMEOUT_S = 15    # a plan normally takes a few seconds; never leave the shopper waiting longer
SCREEN_CUES = re.compile(r"\[\[\d+\]\]")   # [[2]] tags that light up an option on screen; not words
TURNS_KEPT = 50             # exchanges kept word for word, to show the conversation again (agents see the last few)
NOTHING_SHOWN_YET = "I haven't shown you anything yet. Tell me what you're looking for and I'll find a few options."


class ScoutOrchestrator(BaseAgent):
    # ADK agents are pydantic models: dependencies are declared as fields.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    elicitation: BaseAgent
    narration: BaseAgent
    product_expert: BaseAgent
    comparison: BaseAgent
    preference: BaseAgent
    stylist: BaseAgent
    retriever: ProductRetriever
    overview_builder: CatalogOverviewBuilder
    resolver: ProductReferenceResolver
    recommender: Recommender

    def __init__(self, elicitation: BaseAgent, narration: BaseAgent, product_expert: BaseAgent,
                 comparison: BaseAgent, preference: BaseAgent, stylist: BaseAgent, retriever: ProductRetriever,
                 overview_builder: CatalogOverviewBuilder, resolver: ProductReferenceResolver,
                 recommender: Recommender, name: str = "scout"):
        super().__init__(
            name=name,
            description="Scout, a voice shopping companion: routes each turn to the right step.",
            elicitation=elicitation,
            narration=narration,
            product_expert=product_expert,
            comparison=comparison,
            preference=preference,
            stylist=stylist,
            retriever=retriever,
            overview_builder=overview_builder,
            resolver=resolver,
            recommender=recommender,
            sub_agents=[elicitation, narration, product_expert, comparison, preference, stylist],
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # 0. Ground the questions in what the shop really has.
        if self._still_working_out_need(ctx):
            overview = await self._catalog_overview(ctx)
            if overview:
                yield self._update_state(ctx, {state_keys.CATALOG_OVERVIEW: overview})

        # 1. Understand the message (its JSON is data, never spoken).
        async for event in self._silently(self.elicitation, ctx):
            yield event

        turn = read_turn(ctx.session.state)
        if ctx.session.state.get(state_keys.OFFER_OPEN):   # the offer was answered (or passed over) just now
            yield self._update_state(ctx, {state_keys.OFFER_OPEN: False})
        if ctx.session.state.get(state_keys.WARDROBE_JUST_ADDED):   # Elicitation has seen the new photos now
            yield self._update_state(ctx, {state_keys.WARDROBE_JUST_ADDED: []})
        if turn is not None and turn.intent == "outfit" and turn.outfit_action == "start" \
                and not state_keys.planning_outfit(ctx.session.state):
            yield self._update_state(ctx, {state_keys.MODE: state_keys.OUTFIT_MODE})
        if turn is not None and turn.owned_items:          # clothes they told Scout they own
            wardrobe = state_keys.wardrobe(ctx.session.state)
            wardrobe += [new_item(piece.category, piece.description) for piece in turn.owned_items]
            yield self._update_state(ctx, {state_keys.WARDROBE: [item.model_dump() for item in wardrobe]})
        if turn is None:
            yield self._say(ctx, "Sorry, I didn't quite catch that. Could you say it another way?")
            return

        # 2. Reply.
        evidence: Optional[str] = None      # what the Preference agent should learn from, if anything
        spoken: list[str] = []
        async for event in self._route(ctx, turn):
            if isinstance(event, str):      # a route may hand back evidence for learning
                evidence = event
            else:
                if event.content and not event.partial:
                    spoken.append(content_text(event.content))
                yield event
        yield self._update_state(ctx, {state_keys.CONVERSATION: self._conversation(ctx, "\n\n".join(spoken))})

        # 3. Learn, after the shopper already has their answer.
        if evidence or turn.lasting_preference:
            yield self._update_state(ctx, {state_keys.PREFERENCE_EVIDENCE: evidence})
            async for event in self._silently(self.preference, ctx):
                yield event

    async def _route(self, ctx: InvocationContext, turn: TurnUnderstanding):
        if state_keys.planning_outfit(ctx.session.state):
            turn = self._as_outfit_step(ctx, turn)
        suggestion = state_keys.suggestion(ctx.session.state) if turn.about_suggestion else None
        if suggestion is not None and turn.intent in ("choose", "ask_about_product", "reject"):
            # They mean the item Scout suggested, not one on screen.
            if turn.intent == "choose":
                steps = self._choose(ctx, turn, suggestion.product)
            elif turn.intent == "ask_about_product":
                steps = self._answer_about_product(ctx, turn, suggestion.product)
            else:
                steps = self._decline_suggestion(ctx, turn, suggestion.product)
            async for item in steps:
                yield item
            return
        if turn.intent == "reject":
            yield self._evidence("rejected", self._presented_product(ctx, turn.product_position), turn.reason)
        if turn.intent in ("shop", "reject") and turn.ready_to_search:
            # After a rejection, only options they have not heard yet: repeating them is not "better options".
            steps = self._search_and_narrate(ctx, skip_seen=turn.intent == "reject")
        elif turn.question:
            steps = self._just_say(ctx, turn.question)
        elif turn.intent == "more_options":
            steps = self._more_options(ctx, turn.keep_positions)
        elif turn.intent == "ask_about_product":
            steps = self._answer_about_product(ctx, turn)
        elif turn.intent == "compare":
            steps = self._compare(ctx, turn)
        elif turn.intent == "choose":
            steps = self._choose(ctx, turn)
        elif turn.intent == "cart":
            steps = self._cart(ctx, turn)
        elif turn.intent == "see_suggestion":
            steps = self._show_suggestion(ctx)
        elif turn.intent == "outfit":
            steps = self._outfit(ctx, turn)
        elif turn.intent == "pause":
            steps = self._nothing()
        elif turn.intent == "end_conversation":
            steps = self._just_say(ctx, turn.reply or "Okay, talk soon! I'll be here whenever you need me.")
        else:
            steps = self._just_say(ctx, turn.reply or "I'm here whenever you're ready. What are you shopping for today?")
        async for item in steps:
            yield item

    # -----------------------------------------------------------------
    # Steps
    # -----------------------------------------------------------------

    async def _search_and_narrate(self, ctx: InvocationContext, skip_seen: bool = False):
        """Search for the need and present the results."""
        state = ctx.session.state
        stopped = self._stop_choosing_piece(ctx)        # shopping for something else now
        if stopped:
            yield stopped
        need = state_keys.need(state)
        seen = set(state.get(state_keys.SEEN) or []) if skip_seen else set()
        result = await asyncio.to_thread(self.retriever.find, need, seen)
        logger.info("Retrieval for %r found %d (%d exact, relaxed: %s)", result.query_text,
                    len(result.products), len(result.exact_ids), result.relaxed)

        # They rejected everything that really fits (e.g. every black pair): say so honestly, keep what they heard,
        # rather than quietly offering things that miss what they asked for.
        nothing_new_fits = bool(need.must_have) and result.found and not result.exact_ids
        if seen and (not result.found or nothing_new_fits):
            yield self._update_state(ctx, {state_keys.OTHER_TYPES: result.other_types,
                                           state_keys.NARRATION_TASK: NarrationTask.NO_MORE_OPTIONS.value})
            async for event in self.narration.run_async(ctx):
                yield event
            return

        first = [p.id for p in result.products[:PRODUCTS_TO_PRESENT]]
        yield self._update_state(ctx, {         # save results first, so Narration reads them
            state_keys.SHORTLIST: [p.model_dump() for p in result.products],
            state_keys.EXACT_MATCHES: result.exact_ids,
            state_keys.RELAXED: result.relaxed,
            state_keys.OTHER_TYPES: result.other_types,
            state_keys.PRESENTED: first,
            state_keys.SEEN: sorted(seen) + first,
            state_keys.NARRATION_TASK: NarrationTask.PRESENT_OPTIONS.value,
            state_keys.FOCUS_PRODUCT: None,
        })
        async for event in self.narration.run_async(ctx):
            yield event

    async def _more_options(self, ctx: InvocationContext, keep_positions: list[int]):
        """Options they have not heard yet. Only ones that still fit the need: if this message tightened it
        ("under $70"), earlier finds over the new limit are dropped and the search runs again.
        Options they asked to keep stay in their places; only the other places get new options."""
        state = ctx.session.state
        if not state.get(state_keys.SHORTLIST):
            yield self._say(ctx, NOTHING_SHOWN_YET)
            return

        need = state_keys.need(state)
        shortlist = state_keys.shortlist(state)
        presented = list(state.get(state_keys.PRESENTED) or [])
        kept = [presented[n - 1] for n in keep_positions if 1 <= n <= len(presented)]
        wanted = PRODUCTS_TO_PRESENT - len(kept) if kept else PRODUCTS_TO_PRESENT
        seen = list(dict.fromkeys([*(state.get(state_keys.SEEN) or []), *presented]))
        unseen = [p for p in shortlist.values() if p.id not in seen and self.retriever.fits(need, p)]
        delta: dict = {}

        if len(unseen) < wanted:        # not enough left: search again, skipping what we have
            extra = await asyncio.to_thread(self.retriever.find, need, set(seen) | set(shortlist))
            new = [p for p in extra.products if self.retriever.fits(need, p)]   # never over budget to top up
            if new:
                unseen += new
                delta[state_keys.SHORTLIST] = [p.model_dump() for p in [*shortlist.values(), *new]]
                delta[state_keys.EXACT_MATCHES] = list(state.get(state_keys.EXACT_MATCHES) or []) + extra.exact_ids
                delta[state_keys.RELAXED] = [r for r in extra.relaxed if r != "budget"]
            delta[state_keys.OTHER_TYPES] = extra.other_types

        fresh = [p.id for p in unseen[:wanted]]
        if kept:     # the kept ones stay in their places; the new ones fill the others, in order
            queue = iter(fresh)
            next_ids = [i for i in (pid if pid in kept else next(queue, None) for pid in presented) if i]
            next_ids += list(queue)
        else:
            next_ids = fresh
        if fresh:
            delta.update({
                state_keys.PRESENTED: next_ids,
                state_keys.KEPT: kept,
                state_keys.SEEN: seen + fresh,
                state_keys.NARRATION_TASK: NarrationTask.MORE_OPTIONS.value,
                state_keys.FOCUS_PRODUCT: None,
            })
        else:   # nothing new: keep what was presented, so references still work
            delta[state_keys.NARRATION_TASK] = NarrationTask.NO_MORE_OPTIONS.value
        yield self._update_state(ctx, delta)
        async for event in self.narration.run_async(ctx):
            yield event

    async def _answer_about_product(self, ctx: InvocationContext, turn: TurnUnderstanding,
                                    product: Optional[Product] = None):
        """`product` is given when they ask about the suggested item; otherwise it is one on screen."""
        state = ctx.session.state
        if product is None:
            if not state.get(state_keys.SHORTLIST):
                yield self._say(ctx, NOTHING_SHOWN_YET)
                return
            product = self._presented_product(ctx, turn.product_position)
        if product is None:
            yield self._say(ctx, "Happy to! Which one would you like to hear about: the first, second or third?")
            return
        delta = {state_keys.FOCUS_PRODUCT: product.id}
        if product.id not in state_keys.shortlist(state):   # the Product Expert reads the focus from the shortlist,
            delta[state_keys.SHORTLIST] = list(state.get(state_keys.SHORTLIST) or []) + [product.model_dump()]
            delta[state_keys.SEEN] = list(state.get(state_keys.SEEN) or []) + [product.id]   # and it is not "new" later
        yield self._update_state(ctx, delta)
        async for event in self.product_expert.run_async(ctx):
            yield event

    async def _compare(self, ctx: InvocationContext, turn: TurnUnderstanding):
        presented = [p.id for p in state_keys.presented(ctx.session.state)]
        if len(presented) < 2:
            yield self._say(ctx, "I need at least two options to compare. Shall I find a few more for you?")
            return

        chosen = [presented[n - 1] for n in turn.product_positions if 1 <= n <= len(presented)]
        compare_ids = list(dict.fromkeys(chosen)) if len(set(chosen)) >= 2 else presented

        yield self._update_state(ctx, {state_keys.COMPARE_IDS: compare_ids})
        async for event in self._silently(self.comparison, ctx):     # analysis is data, not speech
            yield event
        yield self._update_state(ctx, {state_keys.NARRATION_TASK: NarrationTask.SPEAK_COMPARISON.value})
        async for event in self.narration.run_async(ctx):
            yield event

    async def _choose(self, ctx: InvocationContext, turn: TurnUnderstanding, suggested: Optional[Product] = None):
        """Add a presented product (or the `suggested` item) to the cart, with one item that goes with it."""
        if suggested is None and not ctx.session.state.get(state_keys.SHORTLIST):
            yield self._say(ctx, NOTHING_SHOWN_YET)
            return
        product = suggested or self._presented_product(ctx, turn.product_position)
        if product is None:
            yield self._say(ctx, "Lovely! Which one would you like: the first, second or third?")
            return

        capsule = state_keys.capsule(ctx.session.state)
        if capsule is not None and suggested is None and state_keys.planning_outfit(ctx.session.state):
            # Planning an outfit: what they choose goes into it (the piece being chosen, else the piece of
            # that kind, else a new piece of that kind), never into the cart.
            piece = (capsule.new_piece(capsule.current or "") or capsule.new_piece(product.category or "")
                     or capsule.add_piece(product.category or product.kind or "item", [], product.kind or ""))
            capsule.choose(piece.key, product)
            capsule.skipped = [key for key in capsule.skipped if key != piece.key]
            following = capsule.move_on()
            event = (f"They chose {' '.join(product.title.split()[:8])} for the {piece.category}. "
                     + (f"Next: the {following.category}." if following else "Every piece is now settled."))
            async for item in self._offer_options(ctx, capsule, event):
                yield item
            yield self._evidence("chose", product, turn.reason)
            return

        cart = state_keys.cart(ctx.session.state)
        added = cart.add(product)
        delta = {
            state_keys.CART: cart.model_dump(),
            state_keys.CART_EVENT: {"action": "added" if added else "already_in_cart", "item": product.title},
            state_keys.FOCUS_PRODUCT: product.id,
            state_keys.NARRATION_TASK: NarrationTask.CART_UPDATE.value,
            state_keys.SUGGESTION: None,
        }
        suggestion = None
        if added and suggested is None:   # one thing that goes with it; never a chain of suggestions
            suggestion = await asyncio.to_thread(self.recommender.for_new_item, product, cart,
                                                 self._suggestion_rules(ctx))
            delta.update(self._offer(ctx, suggestion, at_checkout=False))
        yield self._update_state(ctx, delta)
        async for event in self._narrate_offer(ctx, suggestion):
            yield event
        yield self._evidence("chose", product, turn.reason)

    async def _cart(self, ctx: InvocationContext, turn: TurnUnderstanding):
        cart = state_keys.cart(ctx.session.state)
        action = turn.cart_action or "view"
        event: dict = {"action": action}

        if not cart.items and action in ("view", "checkout"):
            # Said in code, never by a model: with nothing to read back, a model can invent a cart,
            # and a blind shopper cannot check the screen.
            yield self._update_state(ctx, {state_keys.SUGGESTION: None})
            yield self._say(ctx, "Your cart is empty right now. What would you like to find?")
            return

        if action in ("remove", "clear"):
            removed = cart.clear() if action == "clear" else cart.remove(turn.cart_positions)
            if removed:
                event = {"action": "removed", "items": [item.name for item in removed]}
            else:
                event = {"action": "cart_already_empty" if action == "clear" else "unclear_which_item"}

        delta = {
            state_keys.CART: cart.model_dump(),
            state_keys.CART_EVENT: event,
            state_keys.NARRATION_TASK: NarrationTask.CART_UPDATE.value,
        }
        if action in ("checkout", "remove", "clear"):   # what an open suggestion went with may be gone
            delta[state_keys.SUGGESTION] = None
        reminder = None
        if action == "checkout" and not ctx.session.state.get(state_keys.CHECKOUT_REMINDED):
            # One last "you might have forgotten..." BEFORE paying; if they go ahead, they are not asked again.
            reminder = await asyncio.to_thread(self.recommender.for_cart, cart, self._suggestion_rules(ctx))
            delta[state_keys.CHECKOUT_REMINDED] = True
            if reminder:
                delta.update(self._offer(ctx, reminder, at_checkout=True))
                delta[state_keys.NARRATION_TASK] = NarrationTask.CHECKOUT_REMINDER.value
        yield self._update_state(ctx, delta)
        async for item in self._narrate_offer(ctx, reminder):
            yield item

    def _as_outfit_step(self, ctx: InvocationContext, turn: TurnUnderstanding) -> TurnUnderstanding:
        """In outfit mode every request is a step of the outfit, decided here by kind of product, never by
        wording: the same kind as a piece (or none named) is other options for that piece, with any new price
        or wish; another kind is a piece to add; pointing at clothes they own is planning again with them."""
        capsule = state_keys.capsule(ctx.session.state)
        if capsule is None or turn.intent == "outfit" or turn.question or turn.about_suggestion:
            return turn
        if turn.owned_items:
            return turn.model_copy(update={"intent": "outfit", "outfit_action": "start"})
        if turn.intent not in ("shop", "more_options", "reject"):
            return turn
        category = turn.need.category
        piece = capsule.new_piece(category) if category else capsule.new_piece(capsule.current or "")
        wish = turn.swap_wish or turn.reason or ", ".join(turn.need.must_have) or None
        if piece is None and category:
            return turn.model_copy(update={"intent": "outfit", "outfit_action": "add",
                                           "swap_piece": category, "swap_wish": wish})
        return turn.model_copy(update={"intent": "outfit", "outfit_action": "swap",
                                       "swap_piece": piece.category if piece else None, "swap_wish": wish})

    async def _outfit(self, ctx: InvocationContext, turn: TurnUnderstanding):
        """Planning an outfit: plan (or plan again), other options for a piece, skip or add a piece, put the new
        pieces in the cart (the outfit is finished), or leave for regular shopping."""
        capsule = state_keys.capsule(ctx.session.state)
        action = turn.outfit_action
        if action == "start":
            steps = self._plan_capsule(ctx, capsule)
        elif action == "leave":
            steps = self._leave_capsule(ctx)
        elif capsule is None:
            steps = self._just_say(ctx, "We haven't planned an outfit yet. Tell me the occasion, and I'll plan "
                                        "it with you.")
        elif action == "swap":
            steps = self._swap_piece(ctx, capsule, turn)
        elif action == "skip":
            steps = self._skip_piece(ctx, capsule, turn)
        elif action == "add":
            steps = self._add_piece(ctx, capsule, turn)
        elif action == "add_to_cart":
            steps = self._add_capsule_to_cart(ctx, capsule)
        else:
            steps = self._narrate_capsule(ctx, capsule, turn.understood or "They asked about the plan.")
        async for item in steps:
            yield item

    async def _plan_capsule(self, ctx: InvocationContext, earlier: Optional[Capsule]):
        """The Stylist plans the looks; then the new pieces are found in the shop. Planning again keeps the
        pieces already chosen. Planning takes a few seconds, so Scout says straight away that it is on it."""
        yield self._update_state(ctx, {state_keys.STYLIST_PLAN: None})   # never reuse an older plan
        own = " from your wardrobe" if state_keys.wardrobe(ctx.session.state) else ""
        yield self._say(ctx, f"Lovely! Give me a few seconds while I plan your looks{own}.")
        try:
            events = await asyncio.wait_for(self._collect(self._silently(self.stylist, ctx)), STYLIST_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning("The Stylist took longer than %ss", STYLIST_TIMEOUT_S)
            events = []
        for event in events:
            yield event
        raw = ctx.session.state.get(state_keys.STYLIST_PLAN)
        plan = CapsulePlan.model_validate(raw) if raw else None
        if plan is None or not plan.looks:
            yield self._say(ctx, "Sorry, I couldn't finish the plan just now. Shall I try again? You can also "
                                 "tell me a little more about the occasion.")
            return
        need = state_keys.need(ctx.session.state)
        owns_some = any(piece.owned for look in plan.looks for piece in look.pieces)
        # The plan keeps the words for their own pieces, not the pictures (those stay in their wardrobe).
        wardrobe = [item.model_copy(update={"thumbnail": None}) for item in state_keys.wardrobe(ctx.session.state)]
        capsule = Capsule(plan=plan, wardrobe=wardrobe, budget=need.budget_max, guided=not owns_some)
        if earlier is not None:
            capsule.keep_choices_from(earlier)
        if capsule.guided:                           # all new: they choose each piece, starting with the first
            capsule.move_on()
            steps = self._offer_options(ctx, capsule, state_keys.CAPSULE_PLANNED)
        else:                                        # the best match for every new piece, all at once
            still_open = [p for p in plan.new_pieces if p.key not in capsule.picks]
            await asyncio.to_thread(self._find_pieces, capsule, need, still_open)
            yield self._update_state(ctx, {state_keys.PRESENTED: [], state_keys.FOCUS_PRODUCT: None})  # old cards
            steps = self._narrate_capsule(ctx, capsule, state_keys.CAPSULE_PLANNED)
        async for item in steps:
            yield item

    async def _swap_piece(self, ctx: InvocationContext, capsule: Capsule, turn: TurnUnderstanding):
        """Other options for one piece, to choose from (what was chosen for it stays until they choose again)."""
        piece = self._piece_meant(capsule, turn)
        if piece is None:
            yield self._which_piece(ctx, capsule)
            return
        self._limit_price(ctx, capsule, piece)
        capsule.skipped = [key for key in capsule.skipped if key != piece.key]
        capsule.current = piece.key
        wish = f" ({turn.swap_wish})" if turn.swap_wish else ""
        async for item in self._offer_options(ctx, capsule, f"They asked for other {piece.category} options{wish}.",
                                              turn.swap_wish or ""):
            yield item

    async def _add_piece(self, ctx: InvocationContext, capsule: Capsule, turn: TurnUnderstanding):
        """Another kind of piece in the outfit (shoes, a bag...), with options to choose from."""
        need = state_keys.need(ctx.session.state)
        category = turn.swap_piece or need.category
        if not category:
            yield self._say(ctx, "Sure! What would you like to add to the outfit?")
            return
        piece = capsule.new_piece(category)
        if piece is None:
            kinds = need.product_types if (need.category or "").lower() == category.lower() else []
            piece = capsule.add_piece(category, kinds, turn.swap_wish or "")
        capsule.skipped = [key for key in capsule.skipped if key != piece.key]
        capsule.current = piece.key
        self._limit_price(ctx, capsule, piece)
        async for item in self._offer_options(ctx, capsule, f"They asked to add a {piece.category} to the outfit.",
                                              turn.swap_wish or ""):
            yield item

    async def _skip_piece(self, ctx: InvocationContext, capsule: Capsule, turn: TurnUnderstanding):
        piece = self._piece_meant(capsule, turn)
        if piece is None:
            yield self._which_piece(ctx, capsule)
            return
        capsule.picks.pop(piece.key, None)
        capsule.skip(piece.key)
        event = f"They do not want the {piece.category}, so it is left out."
        if capsule.guided or capsule.current == piece.key:
            following = capsule.move_on()
            event += f" Next: the {following.category}." if following else " Every piece is now settled."
            steps = self._offer_options(ctx, capsule, event)
        else:
            steps = self._narrate_capsule(ctx, capsule, event)
        async for item in steps:
            yield item

    @staticmethod
    def _limit_price(ctx: InvocationContext, capsule: Capsule, piece: NewPiece) -> None:
        """A price they gave while choosing a piece is for that piece alone, not the whole outfit."""
        need = state_keys.need(ctx.session.state)
        if need.budget_min is not None or (need.budget_max is not None and need.budget_max != capsule.budget):
            capsule.price_limits[piece.key] = (need.budget_min, need.budget_max)

    def _stop_choosing_piece(self, ctx: InvocationContext) -> Optional[Event]:
        """The shopper moved on to regular shopping while choosing an outfit piece: the plan stays on the board
        (the piece can be chosen later), but new options and picks are no longer for the outfit."""
        capsule = state_keys.capsule(ctx.session.state)
        if capsule is None or capsule.current is None:
            return None
        capsule.current = None
        return self._update_state(ctx, {state_keys.CAPSULE: capsule.model_dump()})

    @staticmethod
    def _piece_meant(capsule: Capsule, turn: TurnUnderstanding) -> Optional[NewPiece]:
        """The new piece they mean: the one they named, else the one being chosen, else the only one."""
        pieces = capsule.plan.new_pieces
        return (capsule.new_piece(turn.swap_piece or "") or capsule.new_piece(capsule.current or "")
                or (pieces[0] if len(pieces) == 1 else None))

    def _which_piece(self, ctx: InvocationContext, capsule: Capsule) -> Event:
        names = " or the ".join(p.category for p in capsule.plan.new_pieces)
        return self._say(ctx, f"Which piece do you mean: the {names}?" if names else
                         "This plan doesn't need anything new: you already own every piece!")

    async def _offer_options(self, ctx: InvocationContext, capsule: Capsule, event: str, wish: str = ""):
        """A few options for the piece being chosen now (none once every piece is settled). The shopping need
        becomes that piece, so the next "cheaper" or "in black" is about it."""
        piece = capsule.new_piece(capsule.current or "")
        base = state_keys.need(ctx.session.state)
        current = base.model_copy(update={"category": piece.category if piece else None,
                                          "product_types": piece.product_types if piece else [],
                                          "looking_for": piece.looks_like if piece else "", "must_have": [],
                                          "budget_min": None, "budget_max": capsule.budget})
        delta: dict = {state_keys.PRESENTED: [], state_keys.FOCUS_PRODUCT: None, state_keys.NEED: current.model_dump()}
        if piece is not None:
            need = capsule.need_for(piece, base, wish)
            result = await asyncio.to_thread(self._search_piece, need, set(capsule.shown.get(piece.key, [])))
            options = result.products[:PRODUCTS_TO_PRESENT]
            capsule.offer(piece.key, [p.id for p in options])
            delta.update({
                state_keys.SHORTLIST: [p.model_dump() for p in result.products],
                state_keys.EXACT_MATCHES: result.exact_ids,
                state_keys.RELAXED: result.relaxed,
                state_keys.OTHER_TYPES: result.other_types,
                state_keys.PRESENTED: [p.id for p in options],
                state_keys.SEEN: [p.id for p in options],
            })
        yield self._update_state(ctx, delta)
        async for item in self._narrate_capsule(ctx, capsule, event):
            yield item

    def _find_pieces(self, capsule: Capsule, need: ShoppingNeed, pieces: list[NewPiece], wish: str = "") -> None:
        """Find a product for each piece (all at once), never one already offered for it. Within budget when
        possible; otherwise the nearest the shop has, which the plan's total makes plain."""
        def best(piece: NewPiece) -> Optional[Product]:
            piece_need = capsule.need_for(piece, need, wish)
            found = self._search_piece(piece_need, set(capsule.shown.get(piece.key, []))).products
            return next((p for p in found if self.retriever.fits(piece_need, p)), found[0] if found else None)

        if not pieces:
            return
        with ThreadPoolExecutor(max_workers=len(pieces)) as pool:
            for piece, product in zip(pieces, pool.map(best, pieces)):
                if product is not None:
                    capsule.choose(piece.key, product)

    def _search_piece(self, need: ShoppingNeed, exclude: set[str]) -> RetrievalResult:
        """Search for an outfit piece. The Stylist may name kinds the shop has few of; if none fit, any
        product of that category is better than nothing."""
        result = self.retriever.find(need, exclude)
        if not result.found and need.product_types:
            result = self.retriever.find(need.model_copy(update={"product_types": []}), exclude)
        return result

    async def _narrate_capsule(self, ctx: InvocationContext, capsule: Capsule, event: str):
        yield self._update_state(ctx, {state_keys.CAPSULE: capsule.model_dump(), state_keys.CAPSULE_EVENT: event,
                                       state_keys.SUGGESTION: None,
                                       state_keys.NARRATION_TASK: NarrationTask.CAPSULE.value})
        async for item in self.narration.run_async(ctx):
            yield item

    async def _add_capsule_to_cart(self, ctx: InvocationContext, capsule: Capsule):
        """The new pieces go in the cart and the outfit is finished: back to regular shopping. The board stays,
        so they can still try the looks on."""
        products = capsule.products_to_buy()
        if not products:
            yield self._say(ctx, "Good news: this plan doesn't need anything new. You already own every piece!")
            return
        cart = state_keys.cart(ctx.session.state)
        added = [product.title for product in products if cart.add(product)]
        capsule.current = None
        yield self._update_state(ctx, {
            state_keys.CART: cart.model_dump(),
            state_keys.CART_EVENT: {"action": "added" if added else "already_in_cart", "items": added,
                                    "outfit_finished": True},
            state_keys.NARRATION_TASK: NarrationTask.CART_UPDATE.value,
            state_keys.SUGGESTION: None,
            state_keys.CAPSULE: capsule.model_dump(),
            state_keys.PRESENTED: [],
            state_keys.MODE: None,
        })
        async for event in self.narration.run_async(ctx):
            yield event

    async def _leave_capsule(self, ctx: InvocationContext):
        yield self._update_state(ctx, {state_keys.CAPSULE: None, state_keys.PRESENTED: [], state_keys.MODE: None})
        yield self._say(ctx, "Sure, we'll leave the plan there. What would you like to shop for?")

    async def _decline_suggestion(self, ctx: InvocationContext, turn: TurnUnderstanding, product: Product):
        """No thanks: no new search, no pressure. Declining the reminder at checkout means "check me out"."""
        at_checkout = bool(ctx.session.state.get(state_keys.OFFER_AT_CHECKOUT))
        yield self._update_state(ctx, {state_keys.SUGGESTION: None, state_keys.OFFER_AT_CHECKOUT: False})
        if at_checkout:
            checkout = turn.model_copy(update={"intent": "cart", "cart_action": "checkout", "about_suggestion": False})
            async for item in self._cart(ctx, checkout):     # already reminded once, so this checks out
                yield item
        else:
            yield self._say(ctx, turn.reply or "No problem at all. Is there anything else I can help you find?")
        yield self._evidence("turned down a suggested", product, turn.reason)

    async def _show_suggestion(self, ctx: InvocationContext):
        """The shopper wants to see the suggested item: present it first, with a few like it."""
        state = ctx.session.state
        suggestion = state_keys.suggestion(state)
        if suggestion is None:
            yield self._say(ctx, "I don't have anything else to suggest right now. What else can I find for you?")
            return
        product = suggestion.product
        previous = state_keys.need(state)
        stopped = self._stop_choosing_piece(ctx)
        if stopped:
            yield stopped
        need = ShoppingNeed(
            category=product.category,
            product_types=[product.product_type] if product.product_type else [],
            gender=product.gender if product.gender in ("men", "women") else previous.gender,
            looking_for=product.kind,
            budget_max=previous.budget_max,
            avoid=previous.avoid,
        )
        similar = await asyncio.to_thread(self.retriever.find, need, {product.id})
        shortlist = [product] + similar.products
        first = [p.id for p in shortlist[:PRODUCTS_TO_PRESENT]]
        yield self._update_state(ctx, {
            state_keys.NEED: need.model_dump(),
            state_keys.SHORTLIST: [p.model_dump() for p in shortlist],
            state_keys.EXACT_MATCHES: [product.id] + similar.exact_ids,
            state_keys.RELAXED: similar.relaxed,
            state_keys.OTHER_TYPES: [],
            state_keys.PRESENTED: first,
            state_keys.SEEN: first,
            state_keys.FOCUS_PRODUCT: product.id,
            state_keys.SUGGESTION: None,             # it is on screen now
            state_keys.QUESTIONS_ASKED: 0,
            state_keys.LAST_QUESTION: None,
            state_keys.NARRATION_TASK: NarrationTask.PRESENT_OPTIONS.value,
        })
        async for event in self.narration.run_async(ctx):
            yield event

    async def _narrate_offer(self, ctx: InvocationContext, suggestion: Optional[Suggestion]):
        """Run Narration. If it was asked to offer a suggestion, note whether it really did, so a plain "yes"
        next turn only counts as accepting an offer the shopper actually heard."""
        spoken = []
        async for event in self.narration.run_async(ctx):
            if not event.partial:
                spoken.append(content_text(event.content).lower())
            yield event
        if suggestion is not None:
            kind = suggestion.product.kind.lower()
            yield self._update_state(ctx, {state_keys.OFFER_OPEN: bool(kind) and kind in " ".join(spoken)})

    @staticmethod
    def _conversation(ctx: InvocationContext, reply: str) -> list[dict]:
        """The conversation so far with this exchange added, word for word (screen cues removed)."""
        turns = state_keys.conversation(ctx.session.state)
        turns.append({"shopper": content_text(ctx.user_content, separator=" "),
                      "scout": SCREEN_CUES.sub("", reply).strip()})
        return turns[-TURNS_KEPT:]

    @staticmethod
    def _suggestion_rules(ctx: InvocationContext) -> SuggestionRules:
        """What a suggestion must respect: the current need (budget, gender), things to avoid, and
        everything already suggested (never twice)."""
        state = ctx.session.state
        need = state_keys.need(state)
        return SuggestionRules(
            need=need,
            avoid=need.avoid + [p.value for p in state_keys.profile(state).preferences if p.polarity == "avoid"],
            suggested_ids=set(state.get(state_keys.SUGGESTED_IDS) or []),
            suggested_kinds=set(state.get(state_keys.SUGGESTED_KINDS) or []),
        )

    @staticmethod
    def _offer(ctx: InvocationContext, suggestion: Optional[Suggestion], at_checkout: bool) -> dict:
        """State for a suggestion Scout is about to offer (nothing when there is none)."""
        if suggestion is None:
            return {}
        state = ctx.session.state
        return {
            state_keys.SUGGESTION: suggestion.model_dump(),
            state_keys.SUGGESTED_IDS: list(state.get(state_keys.SUGGESTED_IDS) or []) + [suggestion.product.id],
            state_keys.SUGGESTED_KINDS: list(state.get(state_keys.SUGGESTED_KINDS) or []) + [suggestion.product.kind],
            state_keys.OFFER_AT_CHECKOUT: at_checkout,
        }

    async def _just_say(self, ctx: InvocationContext, text: str):
        yield self._say(ctx, text)

    async def _nothing(self):
        return
        yield   # an empty async generator: Scout stays quiet

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _presented_product(self, ctx: InvocationContext, position: Optional[int]) -> Optional[Product]:
        return self.resolver.resolve(position, state_keys.presented(ctx.session.state))

    @staticmethod
    def _evidence(event: str, product: Optional[Product], reason: Optional[str]) -> str:
        """Plain-text description of what happened, for the Preference agent."""
        lines = [f"The shopper {event} a product." + (f" Their reason: {reason}" if reason else "")]
        if product:
            lines.append(product.to_prompt_text())
        return "\n".join(lines)

    def _still_working_out_need(self, ctx: InvocationContext) -> bool:
        """True before any products were shown, or while Scout is asking questions."""
        state = ctx.session.state
        return not state.get(state_keys.PRESENTED) or bool(state.get(state_keys.LAST_QUESTION))

    async def _catalog_overview(self, ctx: InvocationContext) -> Optional[str]:
        need = state_keys.need(ctx.session.state)
        text = f"{need.search_text()} {content_text(ctx.user_content, separator=' ')}".strip()
        if not text:
            return None
        try:   # helpful, not essential: a failed search must never break the conversation
            # No category filter: if the shopper switches category mid-way, their words lead the search.
            # Gender is kept once said, so the kinds Scout offers are ones it has for that person.
            overview = await asyncio.to_thread(self.overview_builder.build, text, None, need.gender_filter())
            return overview.to_prompt_text()
        except Exception as error:
            logger.warning("Catalog overview skipped: %s", error)
            return None

    @staticmethod
    async def _collect(events: AsyncGenerator[Event, None]) -> list[Event]:
        return [event async for event in events]

    @staticmethod
    async def _silently(agent: BaseAgent, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Run an agent whose output is data (JSON): keep its state updates, hide its text from the shopper."""
        async for event in agent.run_async(ctx):
            if event.content is not None:
                event = event.model_copy(update={"content": None})
            yield event

    def _say(self, ctx: InvocationContext, text: str) -> Event:
        return Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(text=text)]),
        )

    def _update_state(self, ctx: InvocationContext, state_delta: dict) -> Event:
        return Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(state_delta=state_delta),
        )

