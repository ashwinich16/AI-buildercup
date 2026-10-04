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
       choose                -> add to cart, Narration confirms
       cart                  -> view / remove / checkout, Narration speaks it
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
import json
import logging
from typing import AsyncGenerator, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types
from pydantic import ConfigDict

from scout.agents import state_keys
from scout.agents.elicitation import TurnUnderstanding
from scout.agents.narration import NarrationTask
from scout.domain import Product, ShoppingNeed
from scout.tools import Cart, CatalogOverviewBuilder, ProductReferenceResolver, ProductRetriever

logger = logging.getLogger(__name__)

PRODUCTS_TO_PRESENT = 3   # listeners can keep about three options in mind
NOTHING_SHOWN_YET = "I haven't shown you anything yet. Tell me what you're looking for and I'll find a few options."


class ScoutOrchestrator(BaseAgent):
    # ADK agents are pydantic models: dependencies are declared as fields.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    elicitation: BaseAgent
    narration: BaseAgent
    product_expert: BaseAgent
    comparison: BaseAgent
    preference: BaseAgent
    retriever: ProductRetriever
    overview_builder: CatalogOverviewBuilder
    resolver: ProductReferenceResolver

    def __init__(self, elicitation: BaseAgent, narration: BaseAgent, product_expert: BaseAgent,
                 comparison: BaseAgent, preference: BaseAgent, retriever: ProductRetriever,
                 overview_builder: CatalogOverviewBuilder, resolver: ProductReferenceResolver,
                 name: str = "scout"):
        super().__init__(
            name=name,
            description="Scout, a voice shopping companion: routes each turn to the right step.",
            elicitation=elicitation,
            narration=narration,
            product_expert=product_expert,
            comparison=comparison,
            preference=preference,
            retriever=retriever,
            overview_builder=overview_builder,
            resolver=resolver,
            sub_agents=[elicitation, narration, product_expert, comparison, preference],
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

        turn = self._read_turn(ctx)
        if turn is None:
            yield self._say(ctx, "Sorry, I didn't quite catch that. Could you say it another way?")
            return

        # 2. Reply.
        evidence: Optional[str] = None      # what the Preference agent should learn from, if anything
        async for event in self._route(ctx, turn):
            if isinstance(event, str):      # a route may hand back evidence for learning
                evidence = event
            else:
                yield event

        # 3. Learn, after the shopper already has their answer.
        if evidence or turn.lasting_preference:
            yield self._update_state(ctx, {state_keys.PREFERENCE_EVIDENCE: evidence})
            async for event in self._silently(self.preference, ctx):
                yield event

    async def _route(self, ctx: InvocationContext, turn: TurnUnderstanding):
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
        state = ctx.session.state
        need = ShoppingNeed.model_validate(state.get(state_keys.NEED) or {})
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

    async def _more_options(self, ctx: InvocationContext):
        state = ctx.session.state
        shortlist = list(state.get(state_keys.SHORTLIST) or [])
        if not shortlist:
            yield self._say(ctx, NOTHING_SHOWN_YET)
            return

        seen = list(state.get(state_keys.SEEN) or state.get(state_keys.PRESENTED) or [])
        unseen = [p for p in shortlist if p.get("id") not in seen]
        delta: dict = {}

        if len(unseen) < PRODUCTS_TO_PRESENT:        # not enough left: search again, skipping what we have
            need = ShoppingNeed.model_validate(state.get(state_keys.NEED) or {})
            known = set(seen) | {p.get("id") for p in shortlist}
            extra = await asyncio.to_thread(self.retriever.find, need, known)
            if extra.found:
                new = [p.model_dump() for p in extra.products]
                shortlist += new
                unseen += new
                delta[state_keys.SHORTLIST] = shortlist
                delta[state_keys.EXACT_MATCHES] = list(state.get(state_keys.EXACT_MATCHES) or []) + extra.exact_ids
                if extra.relaxed:
                    delta[state_keys.RELAXED] = extra.relaxed
            delta[state_keys.OTHER_TYPES] = extra.other_types

        next_ids = [p.get("id") for p in unseen[:PRODUCTS_TO_PRESENT]]
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

    async def _answer_about_product(self, ctx: InvocationContext, turn: TurnUnderstanding):
        if not ctx.session.state.get(state_keys.SHORTLIST):
            yield self._say(ctx, NOTHING_SHOWN_YET)
            return
        product = self._presented_product(ctx, turn.product_position)
        if product is None:
            yield self._say(ctx, "Happy to! Which one would you like to hear about: the first, second or third?")
            return
        yield self._update_state(ctx, {state_keys.FOCUS_PRODUCT: product.id})
        async for event in self.product_expert.run_async(ctx):
            yield event

    async def _compare(self, ctx: InvocationContext, turn: TurnUnderstanding):
        presented = [p.id for p in self._presented(ctx)]
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

    async def _choose(self, ctx: InvocationContext, turn: TurnUnderstanding):
        if not ctx.session.state.get(state_keys.SHORTLIST):
            yield self._say(ctx, NOTHING_SHOWN_YET)
            return
        product = self._presented_product(ctx, turn.product_position)
        if product is None:
            yield self._say(ctx, "Lovely! Which one would you like: the first, second or third?")
            return

        cart = Cart.model_validate(ctx.session.state.get(state_keys.CART) or {})
        added = cart.add(product)
        yield self._update_state(ctx, {
            state_keys.CART: cart.model_dump(),
            state_keys.CART_EVENT: {"action": "added" if added else "already_in_cart", "item": product.title},
            state_keys.FOCUS_PRODUCT: product.id,
            state_keys.NARRATION_TASK: NarrationTask.CART_UPDATE.value,
        })
        async for event in self.narration.run_async(ctx):
            yield event
        yield self._evidence("chose", product, turn.reason)

    async def _cart(self, ctx: InvocationContext, turn: TurnUnderstanding):
        cart = Cart.model_validate(ctx.session.state.get(state_keys.CART) or {})
        action = turn.cart_action or "view"
        event: dict = {"action": action}

        if action in ("remove", "clear"):
            removed = cart.clear() if action == "clear" else cart.remove(turn.cart_positions)
            if removed:
                event = {"action": "removed", "items": [item.name for item in removed]}
            else:
                event = {"action": "cart_already_empty" if action == "clear" else "unclear_which_item"}

        yield self._update_state(ctx, {
            state_keys.CART: cart.model_dump(),
            state_keys.CART_EVENT: event,
            state_keys.NARRATION_TASK: NarrationTask.CART_UPDATE.value,
        })
        async for item in self.narration.run_async(ctx):
            yield item

    async def _just_say(self, ctx: InvocationContext, text: str):
        yield self._say(ctx, text)

    async def _nothing(self):
        return
        yield   # an empty async generator: Scout stays quiet

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _presented(self, ctx: InvocationContext) -> list[Product]:
        state = ctx.session.state
        by_id = {p.get("id"): p for p in state.get(state_keys.SHORTLIST) or []}
        return [Product.model_validate(by_id[i]) for i in state.get(state_keys.PRESENTED) or [] if i in by_id]

    def _presented_product(self, ctx: InvocationContext, position: Optional[int]) -> Optional[Product]:
        return self.resolver.resolve(position, self._presented(ctx))

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
        need = ShoppingNeed.model_validate(ctx.session.state.get(state_keys.NEED) or {})
        text = f"{need.search_text()} {_user_text(ctx)}".strip()
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
    def _read_turn(ctx: InvocationContext) -> Optional[TurnUnderstanding]:
        raw = ctx.session.state.get(state_keys.TURN)
        if isinstance(raw, str):
            raw = json.loads(raw)
        try:
            return TurnUnderstanding.model_validate(raw) if raw else None
        except ValueError as error:
            logger.warning("Could not read the turn understanding: %s", error)
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


def _user_text(ctx: InvocationContext) -> str:
    content = ctx.user_content
    if not content or not content.parts:
        return ""
    return " ".join(part.text for part in content.parts if part.text)
