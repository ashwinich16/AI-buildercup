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
       outfit                -> Stylist plans the pieces for the occasion; options for one piece at a time,
                                each matched to what was already picked; extras offered once; add it all
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
from scout.domain import Outfit, OutfitPlan, Product, ShoppingNeed
from scout.tools import CatalogOverviewBuilder, ProductReferenceResolver, ProductRetriever
from scout.tools.recommendations import Recommender, Suggestion, SuggestionRules

logger = logging.getLogger(__name__)

PRODUCTS_TO_PRESENT = 3   # listeners can keep about three options in mind
SCREEN_CUES = re.compile(r"\[\[\d+\]\]")   # [[2]] tags that light up an option on screen; not words
RECENT_TURNS_KEPT = 6       # exchanges Elicitation sees word for word (the need, cart and profile hold the rest)
SCOUT_WORDS_KEPT = 400      # characters of each reply: enough to know what was said, small enough to stay fast
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
        outfit = state_keys.outfit(ctx.session.state)
        if turn is not None and outfit is not None and turn.outfit_drop:   # pieces they don't want in it
            outfit.drop(turn.outfit_drop)
            yield self._update_state(ctx, {state_keys.OUTFIT: outfit.model_dump()})
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
        yield self._update_state(ctx, {state_keys.RECENT_TURNS: self._recent_turns(ctx, " ".join(spoken))})

        # 3. Learn, after the shopper already has their answer.
        if evidence or turn.lasting_preference:
            yield self._update_state(ctx, {state_keys.PREFERENCE_EVIDENCE: evidence})
            async for event in self._silently(self.preference, ctx):
                yield event

    async def _route(self, ctx: InvocationContext, turn: TurnUnderstanding):
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
            steps = self._more_options(ctx)
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

    async def _search_and_narrate(self, ctx: InvocationContext, skip_seen: bool = False,
                                  task: Optional[NarrationTask] = None):
        """Search for the need and present the results (as part of the outfit, while one is being planned)."""
        state = ctx.session.state
        task = task or (NarrationTask.OUTFIT_STEP if state.get(state_keys.OUTFIT) else NarrationTask.PRESENT_OPTIONS)
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
            state_keys.NARRATION_TASK: task.value,
            state_keys.FOCUS_PRODUCT: None,
        })
        async for event in self.narration.run_async(ctx):
            yield event

    async def _more_options(self, ctx: InvocationContext):
        """Options they have not heard yet. Only ones that still fit the need: if this message tightened it
        ("under $70"), earlier finds over the new limit are dropped and the search runs again."""
        state = ctx.session.state
        if not state.get(state_keys.SHORTLIST):
            yield self._say(ctx, NOTHING_SHOWN_YET)
            return

        need = state_keys.need(state)
        shortlist = state_keys.shortlist(state)
        seen = list(state.get(state_keys.SEEN) or state.get(state_keys.PRESENTED) or [])
        unseen = [p for p in shortlist.values() if p.id not in seen and self.retriever.fits(need, p)]
        delta: dict = {}

        if len(unseen) < PRODUCTS_TO_PRESENT:        # not enough left: search again, skipping what we have
            extra = await asyncio.to_thread(self.retriever.find, need, set(seen) | set(shortlist))
            new = [p for p in extra.products if self.retriever.fits(need, p)]   # never over budget to top up
            if new:
                unseen += new
                delta[state_keys.SHORTLIST] = [p.model_dump() for p in [*shortlist.values(), *new]]
                delta[state_keys.EXACT_MATCHES] = list(state.get(state_keys.EXACT_MATCHES) or []) + extra.exact_ids
                delta[state_keys.RELAXED] = [r for r in extra.relaxed if r != "budget"]
            delta[state_keys.OTHER_TYPES] = extra.other_types

        next_ids = [p.id for p in unseen[:PRODUCTS_TO_PRESENT]]
        if next_ids:
            delta.update({
                state_keys.PRESENTED: next_ids,
                state_keys.SEEN: seen + next_ids,
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

        outfit = state_keys.outfit(ctx.session.state)
        if suggested is None and outfit is not None:   # planning an outfit: picks go into it, not the cart
            outfit.pick(product)
            yield self._evidence("chose", product, turn.reason)
            async for item in self._outfit_step(ctx, outfit):
                yield item
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

    async def _outfit(self, ctx: InvocationContext, turn: TurnUnderstanding):
        """Styling an outfit: start one, or steer the one in progress."""
        outfit = state_keys.outfit(ctx.session.state)
        action = turn.outfit_action
        if action == "start":
            yield self._update_state(ctx, {state_keys.STYLIST_PLAN: None})   # never reuse an older plan
            async for event in self._silently(self.stylist, ctx):
                yield event
            raw = ctx.session.state.get(state_keys.STYLIST_PLAN)
            plan = OutfitPlan.model_validate(raw) if raw else None
            if plan is None or not plan.pieces:
                yield self._say(ctx, "I couldn't put an outfit together for that from what the shop has. "
                                     "Could you tell me a bit more about the occasion?")
                return
            outfit = Outfit.start(plan, state_keys.need(ctx.session.state).budget_max)
        elif outfit is None:
            yield self._say(ctx, "We don't have an outfit going right now. Tell me the occasion and I'll help "
                                 "you put one together.")
            return
        elif action == "add_to_cart":
            async for item in self._add_outfit_to_cart(ctx, outfit):
                yield item
            return
        elif action == "leave":
            yield self._update_state(ctx, {state_keys.OUTFIT: None})
            yield self._say(ctx, "Sure, we'll leave the outfit there. What would you like to shop for?")
            return
        elif action == "skip" and outfit.current() is not None:
            outfit.skip_current()
        elif action == "add_extra":
            outfit.accept_extra()
        elif action in ("skip", "finish"):   # "no" to the extra on offer
            outfit.decline_extra()
        async for item in self._outfit_step(ctx, outfit):
            yield item

    async def _outfit_step(self, ctx: InvocationContext, outfit: Outfit):
        """Options for the next piece to choose; once every piece is picked, the whole outfit and one offer."""
        need = outfit.need_for_current(state_keys.need(ctx.session.state))
        if need is None and not outfit.chosen():   # every piece skipped
            yield self._update_state(ctx, {state_keys.OUTFIT: None})
            yield self._say(ctx, "No problem, we've skipped every piece. Shall we try a different kind of outfit?")
            return
        if need is None:
            outfit.offer_extra()
            yield self._update_state(ctx, {state_keys.OUTFIT: outfit.model_dump(),
                                           state_keys.NARRATION_TASK: NarrationTask.OUTFIT_DONE.value})
            async for event in self.narration.run_async(ctx):
                yield event
            return
        yield self._update_state(ctx, {state_keys.OUTFIT: outfit.model_dump(), state_keys.NEED: need.model_dump(),
                                       state_keys.SUGGESTION: None})
        async for event in self._search_and_narrate(ctx, task=NarrationTask.OUTFIT_STEP):
            yield event
        if not outfit.introduced:            # the plan is shared once, with the first piece
            outfit.introduced = True
            yield self._update_state(ctx, {state_keys.OUTFIT: outfit.model_dump()})

    async def _add_outfit_to_cart(self, ctx: InvocationContext, outfit: Outfit):
        chosen = outfit.chosen()
        if not chosen:
            yield self._say(ctx, "We haven't picked any pieces yet. Shall we start with the first one?")
            return
        cart = state_keys.cart(ctx.session.state)
        added = [product.title for product in chosen if cart.add(product)]
        yield self._update_state(ctx, {
            state_keys.CART: cart.model_dump(),
            state_keys.CART_EVENT: {"action": "added" if added else "already_in_cart", "items": added},
            state_keys.NARRATION_TASK: NarrationTask.CART_UPDATE.value,
            state_keys.OUTFIT: None,             # done: it is in the cart now
        })
        async for event in self.narration.run_async(ctx):
            yield event

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
    def _recent_turns(ctx: InvocationContext, reply: str) -> list[dict]:
        """The conversation so far, word for word but only the last few exchanges."""
        turns = list(ctx.session.state.get(state_keys.RECENT_TURNS) or [])
        turns.append({"shopper": content_text(ctx.user_content, separator=" "),
                      "scout": " ".join(SCREEN_CUES.sub(" ", reply).split())[:SCOUT_WORDS_KEPT]})
        return turns[-RECENT_TURNS_KEPT:]

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

