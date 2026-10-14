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
       wishlist              -> save a product for later; show the saved ones (they become the presented
                                products, so describing, comparing or choosing them works as usual); compare
                                or remove some of them
       see suggestion        -> present the suggested item with a few like it
       outfit                -> (OutfitPlanner) the Stylist plans the looks they asked for (each with its own pieces: their
                                own clothes first, new ones from the shop), the best product is found for
                                every new piece, and the board is shown. Then change, add or remove a piece
                                in the looks they name (a few options to choose from), put the new pieces in
                                the cart, or leave. Planning is a mode: once started, every request is for the
                                outfit until it is put in the cart or the shopper asks for regular shopping
       pause                 -> say nothing; the UI goes quiet and keeps listening
       end_conversation      -> a short goodbye; the UI stops listening
  3. what there is to learn from (a choice, a rejection, a stated preference) is noted; the Preference agent
     learns from it after the turn has ended (ScoutChatService runs it), so the shopper never waits for it

Routing in code instead of a model is faster (no extra Gemini call per turn)
and fully predictable, which makes it easy to test.

Agents and tools are passed in (Dependency Inversion); this class never
creates them.
"""

import asyncio
import logging
import re
import time
from typing import AsyncGenerator, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types
from pydantic import ConfigDict, PrivateAttr

from scout.agents import state_keys
from scout.agents.elicitation import OutfitStep, TurnUnderstanding, read_turn
from scout.agents.model_io import content_text
from scout.agents.narration import NarrationTask
from scout.agents.orchestrator.outfit import OutfitPlanner
from scout.domain import Outfit, Product, ShoppingNeed
from scout.tools import CatalogOverviewBuilder, ProductReferenceResolver, ProductRetriever, RetrievalResult
from scout.tools.cart import Cart
from scout.tools.taste_reader import describe
from scout.tools.wishlist import Wishlist
from scout.domain.outfit import MAX_LOOKS
from scout.domain.sizes import SizeScale, read_size, scale_of_label, size_scale
from scout.tools.occasion_fit import OccasionRanker
from scout.tools.retrieval import same_listing_key, word_stems
from scout.tools.recommendations import Recommender, Suggestion, SuggestionRules
from scout.tools.wardrobe import new_item

logger = logging.getLogger(__name__)


class _TurnTimer:
    """Seconds since the turn started at each step, logged as one line: "Turn timing: overview 0.4s, ..."."""

    def __init__(self):
        self._start = time.perf_counter()
        self._marks: list[tuple[str, float]] = []

    def mark(self, step: str) -> None:
        self._marks.append((step, time.perf_counter() - self._start))

    def mark_once(self, step: str) -> None:
        if all(name != step for name, _ in self._marks):
            self.mark(step)

    def log(self) -> None:
        logger.info("Turn timing: %s", ", ".join(f"{step} {seconds:.1f}s" for step, seconds in self._marks))

PRODUCTS_TO_PRESENT = 3   # listeners can keep about three options in mind
NEIGHBOUR_CATEGORIES = 2  # other categories searched too when few products suit the occasion
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
    stylist: BaseAgent
    retriever: ProductRetriever
    overview_builder: CatalogOverviewBuilder
    resolver: ProductReferenceResolver
    recommender: Recommender
    occasion_ranker: Optional[OccasionRanker] = None   # outfit pieces best for the occasion first
    _outfits: OutfitPlanner = PrivateAttr()            # planning an outfit (its own class)

    def __init__(self, elicitation: BaseAgent, narration: BaseAgent, product_expert: BaseAgent,
                 comparison: BaseAgent, stylist: BaseAgent, retriever: ProductRetriever,
                 overview_builder: CatalogOverviewBuilder, resolver: ProductReferenceResolver,
                 recommender: Recommender, occasion_ranker: Optional[OccasionRanker] = None, name: str = "scout"):
        super().__init__(
            name=name,
            description="Scout, a voice shopping companion: routes each turn to the right step.",
            elicitation=elicitation,
            narration=narration,
            product_expert=product_expert,
            comparison=comparison,
            stylist=stylist,
            retriever=retriever,
            overview_builder=overview_builder,
            resolver=resolver,
            recommender=recommender,
            occasion_ranker=occasion_ranker,
            sub_agents=[elicitation, narration, product_expert, comparison, stylist],
        )
        self._outfits = OutfitPlanner(self, options_shown=PRODUCTS_TO_PRESENT)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        timer = _TurnTimer()          # where the seconds go, one log line per turn
        # 0. Ground the questions in what the shop really has (not needed for a command understood in code).
        if self._still_working_out_need(ctx) and self._fast_turn(ctx) is None:
            overview = await self._catalog_overview(ctx)
            if overview:
                yield self._update_state(ctx, {state_keys.CATALOG_OVERVIEW: overview})
        timer.mark("overview")

        # 1. Understand the message (its JSON is data, never spoken). Button presses and short, clear commands
        #    ("add the first one") are understood in code: the model step takes seconds and adds nothing there.
        fast = self._fast_turn(ctx)
        if fast is not None:
            yield self._update_state(ctx, {state_keys.TURN: fast.model_dump()})
        else:
            async for event in self._silently(self.elicitation, ctx):
                yield event
        timer.mark("understood" if fast is None else "understood in code")

        turn = read_turn(ctx.session.state)
        if ctx.session.state.get(state_keys.OFFER_OPEN):   # the offer was answered (or passed over) just now
            yield self._update_state(ctx, {state_keys.OFFER_OPEN: False})
        if ctx.session.state.get(state_keys.WARDROBE_JUST_ADDED):   # Elicitation has seen the new photos now
            yield self._update_state(ctx, {state_keys.WARDROBE_JUST_ADDED: []})
        step = turn.outfit if turn is not None and turn.intent == "outfit" else None
        if step is not None and step.action == "plan" and not state_keys.planning_outfit(ctx.session.state):
            # A new outfit: planning is a mode, and how many looks is this outfit's own.
            yield self._update_state(ctx, {state_keys.MODE: state_keys.OUTFIT_MODE, state_keys.LOOK_COUNT: None})
        if step is not None and step.look_count:           # how many looks: kept for plans made again
            yield self._update_state(ctx, {state_keys.LOOK_COUNT: min(max(step.look_count, 1), MAX_LOOKS)})
        if step is not None and step.set_aside:            # their own pieces not to use in this plan
            owned = state_keys.wardrobe(ctx.session.state)
            ids = [owned[n - 1].id for n in step.set_aside if 1 <= n <= len(owned)]
            aside = list(dict.fromkeys([*(ctx.session.state.get(state_keys.SET_ASIDE) or []), *ids]))
            yield self._update_state(ctx, {state_keys.SET_ASIDE: aside})
        if turn is not None and turn.wardrobe_remove:      # pieces they no longer own: out of the wardrobe for good
            wardrobe = state_keys.wardrobe(ctx.session.state)
            gone = [wardrobe[n - 1] for n in dict.fromkeys(turn.wardrobe_remove) if 1 <= n <= len(wardrobe)]
            kept = [item for item in wardrobe if item not in gone]
            aside = [i for i in ctx.session.state.get(state_keys.SET_ASIDE) or [] if i not in {g.id for g in gone}]
            yield self._update_state(ctx, {state_keys.WARDROBE: [item.model_dump() for item in kept],
                                           state_keys.SET_ASIDE: aside})
            if gone:                                      # said in code, so they always hear it was done
                done = f"Done, I've taken the {' and the '.join(item.description for item in gone)} out of your wardrobe."
                outfit = state_keys.outfit(ctx.session.state) if state_keys.planning_outfit(ctx.session.state) else None
                replan = outfit is not None and any(p.owned in {g.id for g in gone}
                                                    for look in outfit.looks for p in look.pieces)
                if turn.intent == "other" and not replan:   # nothing else to do: that is the whole reply
                    turn = turn.model_copy(update={"reply": done})
                else:                                     # then the looks are planned again, or the request answered
                    yield self._say(ctx, done)
        if turn is not None and turn.owned_items:          # clothes they told Scout they own
            wardrobe = state_keys.wardrobe(ctx.session.state)
            wardrobe += [new_item(piece.category, piece.description) for piece in turn.owned_items]
            yield self._update_state(ctx, {state_keys.WARDROBE: [item.model_dump() for item in wardrobe],
                                           **self._learn_from(ctx, "owned", [(piece.description, None)
                                                                            for piece in turn.owned_items])})
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
                if event.content:
                    timer.mark_once("first words")
                if event.content and not event.partial:
                    spoken.append(content_text(event.content))
                yield event
        timer.mark(f"reply ({turn.intent})")
        yield self._update_state(ctx, {state_keys.CONVERSATION: self._conversation(ctx, "\n\n".join(spoken))})

        # 3. Note what to learn from; the Preference agent learns it after the turn, so no one waits for it.
        if evidence or turn.lasting_preference:
            pending = list(ctx.session.state.get(state_keys.PREFERENCE_PENDING) or [])
            pending.append({"evidence": evidence, "message": content_text(ctx.user_content, separator=" ")})
            yield self._update_state(ctx, {state_keys.PREFERENCE_PENDING: pending})
        timer.log()

    async def _route(self, ctx: InvocationContext, turn: TurnUnderstanding):
        pending = ctx.session.state.get(state_keys.PENDING_SIZE)
        if pending and pending.get("outfit") and turn.size and turn.intent in ("choose", "other"):
            async for item in self._outfits.size_given(ctx, turn, pending):   # a size for the outfit's pieces
                yield item
            return
        if ctx.session.state.get(state_keys.CONFIRM_SIZES):            # "add the outfit in those sizes?"
            async for item in self._outfits.sizes_confirmed(ctx, turn):
                yield item
            return
        where = ctx.session.state.get(state_keys.PENDING_WHERE)
        if where:                                                       # "in your outfit, or separately?"
            yield self._update_state(ctx, {state_keys.PENDING_WHERE: None})
            choice = self._outfits.where_choice(ctx, turn)
            if choice:
                async for item in self._outfits.where_answered(ctx, where, choice):
                    yield item
                return
        if state_keys.planning_outfit(ctx.session.state) and self._outfits.asks_where(ctx, turn):
            async for item in self._outfits.ask_where(ctx, turn):
                yield item
            return
        if state_keys.planning_outfit(ctx.session.state):
            turn = self._outfits.as_step(ctx, turn)
        if pending and turn.size and turn.intent in ("choose", "other") and not turn.about_suggestion \
                and not pending.get("outfit"):
            product = Product.model_validate(pending["product"])      # the size for the product being added
            turn = turn.model_copy(update={"intent": "choose", "reason": turn.reason or pending.get("reason")})
            async for item in self._choose(ctx, turn, product if pending.get("suggested") else None, chosen=product):
                yield item
            return
        if pending and turn.intent != "other":                          # they moved on without a size
            yield self._update_state(ctx, {state_keys.PENDING_SIZE: None, state_keys.SIZE_CHOICES: None})
        suggestion = (state_keys.suggestion(ctx.session.state, turn.suggestion_position)
                      if turn.about_suggestion else None)
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
        if turn.intent == "shop" and turn.ready_to_search and self._same_search(ctx):
            steps = self._more_options(ctx, [])      # "show me some other ones": what they have not heard yet
        elif turn.intent in ("shop", "reject") and turn.ready_to_search:
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
        elif turn.intent == "wishlist":
            steps = self._wishlist(ctx, turn)
        elif turn.intent == "see_suggestion":
            steps = self._show_suggestion(ctx)
        elif turn.intent == "outfit":
            steps = self._outfits.run(ctx, turn)
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
        stopped = self._outfits.stop_changing(ctx)        # shopping for something else now
        if stopped:
            yield stopped
        need = state_keys.need(state)
        seen = set(state.get(state_keys.SEEN) or []) if skip_seen else set()
        result = await asyncio.to_thread(self.retriever.find, need, seen)
        logger.info("Retrieval for %r found %d (%d exact, relaxed: %s)", result.query_text,
                    len(result.products), len(result.exact_ids), result.relaxed)
        result, fit_notes = await asyncio.to_thread(self._rank_for_occasion, need, result, seen)
        result = result.model_copy(update={"products": self._by_taste(state, need, result, fit_notes)})
        heard = set(state.get(state_keys.SEEN) or [])
        if heard and not seen:     # a changed request ("cheaper"): ones they have not heard yet come first
            heard_names = {same_listing_key(p) for p in state_keys.shortlist(state).values() if p.id in heard}
            result = result.model_copy(update={"products": sorted(
                result.products, key=lambda p: p.id in heard or same_listing_key(p) in heard_names)})

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
            state_keys.FIT_NOTES: {**(state.get(state_keys.FIT_NOTES) or {}), **fit_notes},
            state_keys.SEARCHED_NEED: need.model_dump(),
        })
        async for event in self.narration.run_async(ctx):
            yield event

    @staticmethod
    def _same_search(ctx: InvocationContext) -> bool:
        """Is the need exactly what the options on screen were found for (they asked again, or for others)?"""
        state = ctx.session.state
        searched = state.get(state_keys.SEARCHED_NEED)
        return bool(searched and state.get(state_keys.SHORTLIST)) and \
            ShoppingNeed.model_validate(searched) == state_keys.need(state)

    def _rank_for_occasion(self, need: ShoppingNeed, result: RetrievalResult,
                           exclude: set[str]) -> tuple[RetrievalResult, dict]:
        """When they named an occasion (a wedding, the gym), the products best for it first; search matches
        the words, not the occasion. Exact matches that suit it stay ahead of related ones."""
        if self.occasion_ranker is None or not need.purpose or not result.products:
            return result, {}
        kind = " or ".join(need.product_types) or need.category or "item"
        ranked, notes = self._suited(need.purpose, kind, need.looking_for, need, result.products, exclude)
        if notes is None:
            return result, {}
        ranked.sort(key=lambda p: (p.id not in notes, p.id not in result.exact_ids))   # stable within each
        return result.model_copy(update={"products": ranked}), notes

    @staticmethod
    def _by_taste(state, need: ShoppingNeed, result: RetrievalResult, fit_notes: dict) -> list[Product]:
        """Options that match their taste (colours, materials, patterns, styles they keep choosing) first, but only
        within each group: exact matches stay ahead of related ones, good fits for the occasion ahead of the rest.
        Not for something they are buying for someone else."""
        if need.for_someone_else:
            return result.products
        avoid = {p.value.strip().lower() for p in state_keys.profile(state).preferences if p.polarity == "avoid"}
        marks = [m for found in state_keys.taste(state).patterns(need.gender_filter(), avoid).values() for m in found]
        if not marks:
            return result.products
        wanted = [(word_stems(m.value), m.weight) for m in marks]

        def score(product: Product) -> float:
            stems = word_stems(describe(product))
            return sum(weight for value, weight in wanted if value and value <= stems)

        exact = set(result.exact_ids)
        return sorted(result.products, key=lambda p: (bool(fit_notes) and p.id not in fit_notes, p.id not in exact,
                                                      -score(p)))

    def _neighbour_categories(self, need: ShoppingNeed) -> list[str]:
        """Other categories the shop files this kind of thing under: the ones its closest matches for the same
        words come from (in the same style), strongest first. Read from the catalog, never written into the code,
        so it works for any shop. A category seen only once among the matches is a stray, not a neighbour."""
        text = " ".join(filter(None, [need.looking_for, *need.product_types])) or need.category or ""
        try:
            overview = self.overview_builder.build(text, None, need.gender_filter())
        except Exception as error:          # widening is a bonus: without it, the category alone is searched
            logger.warning("Could not find neighbouring categories: %s", error)
            return []
        own = (need.category or "").lower()
        return [c for c, n in overview.categories.items() if c.lower() != own and n > 1][:NEIGHBOUR_CATEGORIES]

    def _suited(self, occasion: str, kind: str, looks_like: str, need: ShoppingNeed, products: list[Product],
                exclude: set[str]) -> tuple[list[Product], dict]:
        """The products best for the occasion first, and why each that suits it does. When fewer than three
        suit it (every soft long-sleeve top found was gym wear, for an office), the whole category is
        searched too: a blouse or a shirt may suit the occasion where that exact kind does not."""
        ranked, notes = self.occasion_ranker.rank(occasion, kind, looks_like, products)
        if notes is not None and len(notes) < PRODUCTS_TO_PRESENT and need.category:
            wider = self.wider_candidates(need, exclude | {p.id for p in products})
            if wider:
                logger.info("Only %d of %d suit %r; widened", len(notes), len(products), occasion)
                wider_ranked, wider_notes = self.occasion_ranker.rank(occasion, kind, looks_like, [*products, *wider])
                if wider_notes is not None:          # could not judge the wider set: keep the first judgement
                    ranked, notes = wider_ranked, self.own_close_only(need, wider_ranked, wider_notes)
        return self.one_per_listing(ranked), notes

    def wider_candidates(self, need: ShoppingNeed, exclude: set[str]) -> list[Product]:
        """More of what this piece could be: its whole category and the ones the shop files the same kind of
        thing under (a collared shirt may be a "shirt" or a "top"), never one already found."""
        categories = [need.category, *self._neighbour_categories(need)]
        taken, wider = set(exclude), []
        for category in categories:
            found = self.retriever.find(need.model_copy(update={"category": category, "product_types": [],
                                                                "looking_for": ""}), taken).products
            wider += found
            taken |= {p.id for p in found}
        return wider

    @staticmethod
    def own_close_only(need: ShoppingNeed, ranked: list[Product], notes: dict) -> dict:
        """From another category only real fits (a dress shirt for an office "top"), never a near miss (jeans
        when formal trousers run out)."""
        own = (need.category or "").lower()
        category = {p.id: (p.category or "").lower() for p in ranked}
        return {pid: why for pid, why in notes.items()
                if not (why.startswith("a close match") and category.get(pid, own) != own)}

    @staticmethod
    def one_per_listing(products: list[Product]) -> list[Product]:
        """The same product listed twice is one option, not two."""
        names: set[str] = set()
        return [p for p in products if not (same_listing_key(p) in names or names.add(same_listing_key(p)))]

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
        heard = {same_listing_key(p) for p in shortlist.values() if p.id in seen}   # a second listing is no news
        unseen = [p for p in shortlist.values()
                  if p.id not in seen and same_listing_key(p) not in heard and self.retriever.fits(need, p)]
        delta: dict = {}

        if len(unseen) < wanted:        # not enough left: search again, skipping what we have
            extra = await asyncio.to_thread(self.retriever.find, need, set(seen) | set(shortlist))
            new = [p for p in extra.products                 # never over budget to top up
                   if self.retriever.fits(need, p) and same_listing_key(p) not in heard]
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
        async for event in self._compare_products(ctx, list(dict.fromkeys(chosen)) if len(set(chosen)) >= 2
                                                  else presented):
            yield event

    async def _compare_products(self, ctx: InvocationContext, compare_ids: list[str]):
        """The Comparison agent analyses these products (ones on screen); Narration speaks it."""
        yield self._update_state(ctx, {state_keys.COMPARE_IDS: compare_ids})
        async for event in self._silently(self.comparison, ctx):     # analysis is data, not speech
            yield event
        yield self._update_state(ctx, {state_keys.NARRATION_TASK: NarrationTask.SPEAK_COMPARISON.value})
        async for event in self.narration.run_async(ctx):
            yield event

    async def _choose(self, ctx: InvocationContext, turn: TurnUnderstanding, suggested: Optional[Product] = None,
                      chosen: Optional[Product] = None):
        """Add a presented product (or the `suggested` item) to the cart, with one item that goes with it."""
        if suggested is None and not ctx.session.state.get(state_keys.SHORTLIST):
            yield self._say(ctx, NOTHING_SHOWN_YET)
            return
        product = suggested or chosen or self._presented_product(ctx, turn.product_position)
        if product is None:
            yield self._say(ctx, "Lovely! Which one would you like: the first, second or third?")
            return

        outfit = state_keys.outfit(ctx.session.state)
        if outfit is not None and outfit.changing and outfit.changing.looks and suggested is None \
                and state_keys.planning_outfit(ctx.session.state):
            # Planning an outfit: what they choose goes into the looks being changed, never into the cart
            # (sizes are asked once, when the outfit goes in the cart).
            async for item in self._outfits.put_in(ctx, outfit, product):
                yield item
            yield self._evidence("chose", product, turn.reason)
            return

        cart = state_keys.cart(ctx.session.state)
        scale = size_scale(product)
        size, remembered = self._size_for(ctx, scale, turn)
        in_cart = any(item.product_id == product.id for item in cart.items)
        if scale is not None and size is None and not in_cart:
            async for item in self._ask_size(ctx, product, scale, suggested is not None, turn.reason):
                yield item
            return
        label = scale.label(size) if scale is not None and size else None
        added = cart.add(product, label)
        event = {"action": "added" if added else "already_in_cart", "item": product.title}
        if label and added:
            event["size"] = label
            if remembered:
                event["size_note"] = "the size they gave earlier for this kind; they can change it"
        elif label and turn.size and cart.set_size(product.id, label):
            event = {"action": "size_changed", "item": product.title, "size": label}
        known = dict(ctx.session.state.get(state_keys.KNOWN_SIZES) or {})
        if scale is not None and size:
            known[scale.key] = size
        wishlist = state_keys.wishlist(ctx.session.state)
        wishlist.remove([product.id])                 # in the cart now: saving it for later is done
        delta = {
            **(self._learn_from(ctx, "bought", [(describe(product), product.gender)]) if added else {}),
            state_keys.CART: cart.model_dump(),
            state_keys.WISHLIST: wishlist.model_dump(),
            state_keys.CART_EVENT: event,
            state_keys.PENDING_SIZE: None,
            state_keys.SIZE_CHOICES: None,
            state_keys.KNOWN_SIZES: known,
            state_keys.FOCUS_PRODUCT: product.id,
            state_keys.NARRATION_TASK: NarrationTask.CART_UPDATE.value,
            state_keys.SUGGESTION: None,
        }
        suggestion = None
        if added and suggested is None:   # one thing that goes with it; never a chain of suggestions
            options = await asyncio.to_thread(self.recommender.options_for_new_item, product, cart,
                                              self._suggestion_rules(ctx))
            suggestion = options[0] if options else None   # Scout names the best; the screen shows them all
            delta.update(self._offer(ctx, suggestion, at_checkout=False))
            if options:
                delta[state_keys.SUGGESTIONS] = [option.model_dump() for option in options]
                # every option shown counts as suggested, so none of them comes back later
                delta[state_keys.SUGGESTED_IDS] = list(dict.fromkeys(
                    list(delta.get(state_keys.SUGGESTED_IDS) or []) + [o.product.id for o in options]))
        yield self._update_state(ctx, delta)
        async for event in self._narrate_offer(ctx, suggestion):
            yield event
        yield self._evidence("chose", product, turn.reason)

    def _size_for(self, ctx: InvocationContext, scale, turn: TurnUnderstanding) -> tuple[Optional[str], bool]:
        """The size to add a product in: the one they just gave, else the one they gave before for this kind
        (shoes, waist, tops); and whether it was remembered rather than just said."""
        if scale is None:
            return None, False
        if turn.size:
            return read_size(turn.size, scale), False
        known = (ctx.session.state.get(state_keys.KNOWN_SIZES) or {}).get(scale.key)
        return (known, True) if known in scale.options else (None, False)

    async def _ask_size(self, ctx: InvocationContext, product: Product, scale, suggested: bool,
                        reason: Optional[str]):
        """Ask which size to add it in. The catalog lists no sizes, so every usual size is taken to be in stock."""
        kind = product.kind or "item"
        question = f"Great choice! What size would you like in the {kind}? They come in {scale.spoken}."
        yield self._update_state(ctx, {
            state_keys.PENDING_SIZE: {"product": product.model_dump(), "suggested": suggested, "reason": reason},
            state_keys.SIZE_CHOICES: scale.options,
            state_keys.FOCUS_PRODUCT: product.id,
        })
        yield self._say(ctx, question)

    @staticmethod
    def _resize(cart, turn: TurnUnderstanding) -> dict:
        """Change the size of an item in the cart (the only item, or the one they named)."""
        positions = turn.cart_positions or ([1] if len(cart.items) == 1 else [])
        if len(positions) != 1 or not 1 <= positions[0] <= len(cart.items):
            return {"action": "unclear_which_item"}
        item = cart.items[positions[0] - 1]
        if not item.size:
            return {"action": "needs_no_size", "item": item.name}
        scale = scale_of_label(item.size)
        size = read_size(turn.size or "", scale)
        if size is None:
            return {"action": "size_unclear", "item": item.name, "sizes": scale.spoken}
        cart.set_size(item.product_id, scale.label(size))
        return {"action": "size_changed", "item": item.name, "size": scale.label(size)}

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

        if action in ("remove", "clear", "view"):
            # Plain facts about the cart are said from code: instant, and never wrong about what is in it.
            reply = self._cart_reply(cart, action, turn.cart_positions)       # changes the cart first
            if action != "view":     # what an open suggestion went with may be gone
                yield self._update_state(ctx, {state_keys.CART: cart.model_dump(), state_keys.SUGGESTION: None})
            yield self._say(ctx, reply)
            return
        if action == "size":
            event = self._resize(cart, turn)

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
        if action == "checkout" and reminder is None:      # read back the order, from code
            yield self._say(ctx, f"Here's your order: {cart.spoken()}. Payment isn't connected in this prototype "
                                 "yet, but your cart is saved for next time.")
            return
        async for item in self._narrate_offer(ctx, reminder):
            yield item

    @staticmethod
    def _cart_reply(cart: Cart, action: str, positions: list[int]) -> str:
        """What happened to the cart, in plain words. Changes `cart` for remove and clear."""
        def short(item) -> str:
            return item.kind or " ".join(item.name.split()[:6])
        if action == "view":
            return f"You have {cart.spoken()}. Would you like to check out, or keep shopping?"
        if action == "remove" and not positions and len(cart.items) > 1:
            which = "; ".join(f"{n}, the {short(item)}" for n, item in enumerate(cart.items, start=1))
            return f"Which one should I take out? {which}."
        removed = cart.clear() if action == "clear" else cart.remove(positions or [1])
        if not removed:
            return "Your cart is already empty. What would you like to find?"
        left = f"You still have {cart.spoken()}." if cart.items else "Your cart is empty now."
        return f"Done, I've taken out the {' and the '.join(short(item) for item in removed)}. {left} What next?"

    # -----------------------------------------------------------------
    # Wishlist: products saved to come back to
    # -----------------------------------------------------------------

    async def _wishlist(self, ctx: InvocationContext, turn: TurnUnderstanding):
        action = turn.wishlist_action or "view"
        if action == "save":
            steps = self._save_for_later(ctx, turn)
        elif action == "remove":
            steps = self._remove_saved(ctx, turn)
        elif action == "compare":
            steps = self._compare_saved(ctx, turn)
        else:
            steps = self._show_saved(ctx)
        async for item in steps:
            yield item

    async def _save_for_later(self, ctx: InvocationContext, turn: TurnUnderstanding):
        suggestion = state_keys.suggestion(ctx.session.state, turn.suggestion_position) if turn.about_suggestion \
            else None
        product = suggestion.product if suggestion else self._presented_product(ctx, turn.product_position)
        if product is None:
            yield self._say(ctx, "Happy to save one for later! Which one: the first, second or third?")
            return
        wishlist = state_keys.wishlist(ctx.session.state)
        name = product.kind or "one"
        if not wishlist.add(product):
            yield self._say(ctx, "That one is already on your wishlist.")
            return
        count = len(wishlist.items)
        yield self._update_state(ctx, {state_keys.WISHLIST: wishlist.model_dump(),
                                       **self._learn_from(ctx, "saved", [(describe(product), product.gender)])})
        yield self._say(ctx, f"Saved the {name} to your wishlist. You have {count} saved "
                             f"{'item' if count == 1 else 'items'} to compare or buy whenever you like.")

    async def _show_saved(self, ctx: InvocationContext, lead: str = ""):
        """The wishlist on screen as the presented products, numbered as on the wishlist."""
        wishlist = state_keys.wishlist(ctx.session.state)
        if not wishlist.items:
            yield self._say(ctx, "Your wishlist is empty for now. When you hear something you like, just say "
                                 "save it, and I'll keep it for later.")
            return
        for event in await self._present_saved(ctx, wishlist):
            yield event
        if lead:
            yield self._say(ctx, lead)
        elif len(wishlist.items) > 1:
            yield self._say(ctx, f"Here's your wishlist: {wishlist.spoken()}. Would you like me to compare some of "
                                 "them, tell you more about one, or add one to your cart?")
        else:
            yield self._say(ctx, f"You have one thing saved: the {Wishlist.short_name(wishlist.items[0])}. Would you "
                                 "like to hear more about it, or add it to your cart?")

    async def _present_saved(self, ctx: InvocationContext, wishlist: Wishlist) -> list[Event]:
        """Puts the saved products on screen (the full products: from this conversation, else the catalog)."""
        state = ctx.session.state
        known = state_keys.shortlist(state)
        missing = [pid for pid in wishlist.ids() if pid not in known]
        fetched = await asyncio.to_thread(self.retriever.products, missing) if missing else []
        known.update({p.id: p for p in fetched})
        ids = [pid for pid in wishlist.ids() if pid in known]
        events = []
        stopped = self._outfits.stop_changing(ctx)        # what is on screen is no longer options for a piece
        if stopped:
            events.append(stopped)
        events.append(self._update_state(ctx, {
            state_keys.SHORTLIST: list(state.get(state_keys.SHORTLIST) or []) + [p.model_dump() for p in fetched],
            state_keys.PRESENTED: ids,
            state_keys.SEEN: list(dict.fromkeys([*(state.get(state_keys.SEEN) or []), *ids])),
            state_keys.FOCUS_PRODUCT: None,
        }))
        return events

    async def _compare_saved(self, ctx: InvocationContext, turn: TurnUnderstanding):
        """Compare the saved products they name; with only a few saved, all of them. With many, Scout asks which
        (comparing a long list out loud is hard to follow)."""
        wishlist = state_keys.wishlist(ctx.session.state)
        if len(wishlist.items) < 2:
            yield self._say(ctx, "You need at least two things on your wishlist to compare. Say save it when you "
                                 "hear one you like.")
            return
        chosen = wishlist.ids(turn.wishlist_positions) if turn.wishlist_positions else []
        if len(chosen) < 2 and len(wishlist.items) > PRODUCTS_TO_PRESENT:
            async for item in self._show_saved(ctx, lead=f"Sure! Here's your wishlist: {wishlist.spoken()}. "
                                                         "Which ones shall I compare?"):
                yield item
            return
        for event in await self._present_saved(ctx, wishlist):
            yield event
        on_screen = set(ctx.session.state.get(state_keys.PRESENTED) or [])
        ids = [pid for pid in (chosen if len(chosen) >= 2 else wishlist.ids()) if pid in on_screen]
        async for event in self._compare_products(ctx, ids):
            yield event

    async def _remove_saved(self, ctx: InvocationContext, turn: TurnUnderstanding):
        wishlist = state_keys.wishlist(ctx.session.state)
        ids = wishlist.ids(turn.wishlist_positions) if turn.wishlist_positions else \
            (wishlist.ids() if len(wishlist.items) == 1 else [])
        if not ids:
            yield self._say(ctx, "Your wishlist is already empty." if not wishlist.items else
                            f"Which one should I take off? {wishlist.spoken()}.")
            return
        on_screen = list(ctx.session.state.get(state_keys.PRESENTED) or [])
        showing_wishlist = bool(on_screen) and set(on_screen) <= set(wishlist.ids())
        removed = wishlist.remove(ids)
        delta: dict = {state_keys.WISHLIST: wishlist.model_dump()}
        if showing_wishlist:                          # the wishlist is on screen: it shows what is left
            delta[state_keys.PRESENTED] = [pid for pid in on_screen if pid not in ids]
        yield self._update_state(ctx, delta)
        names = " and the ".join(Wishlist.short_name(item) for item in removed)
        left = f"You have {len(wishlist.items)} saved." if wishlist.items else "Your wishlist is empty now."
        yield self._say(ctx, f"Done, I've taken the {names} off your wishlist. {left}")

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
        stopped = self._outfits.stop_changing(ctx)
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
            occasion=need.purpose or "",
            taste=state_keys.taste_text(state),
            wardrobe=[item.description for item in state_keys.wardrobe(state)],
        )

    @staticmethod
    def _offer(ctx: InvocationContext, suggestion: Optional[Suggestion], at_checkout: bool) -> dict:
        """State for a suggestion Scout is about to offer (nothing when there is none)."""
        if suggestion is None:
            return {}
        state = ctx.session.state
        return {
            state_keys.SUGGESTION: suggestion.model_dump(),
            state_keys.SUGGESTIONS: [suggestion.model_dump()],   # replaced by the full list when there is one
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

    @staticmethod
    def _learn_from(ctx: InvocationContext, signal: str, items: list[tuple[str, Optional[str]]]) -> dict:
        """Items to learn their taste from (each: what it is, and the style it is made for), read after the turn.
        Nothing is learned from something bought or saved for someone else (what they own is always theirs)."""
        state = ctx.session.state
        if not items or (signal != "owned" and state_keys.need(state).for_someone_else):
            return {}
        queued = [{"signal": signal, "text": text, "style": style} for text, style in items]
        return {state_keys.TASTE_EVIDENCE: list(state.get(state_keys.TASTE_EVIDENCE) or []) + queued}

    def _fast_turn(self, ctx: InvocationContext) -> Optional[TurnUnderstanding]:
        """The turn, when the message is a button press or a short, unambiguous command; else None.

        Only exact, simple forms: "Add option 2 to my cart", "add the first one", "describe the second one",
        the suggestion buttons, "Show me more options", "Compare these", and the outfit board's buttons
        ("Change the shoe in look 2", "All looks"). Anything else, or anything that
        does not fit what is on screen, goes to the Elicitation model as before.
        """
        state = ctx.session.state
        text = re.sub(r"[.!?\s]+$", "", content_text(ctx.user_content, separator=" ").strip().lower()).replace("’", "'")
        if not text:
            return None
        need = ShoppingNeed.model_validate(state.get(state_keys.NEED) or {})
        presented = len(state.get(state_keys.PRESENTED) or [])
        base = {"need": need, "ready_to_search": False, "understood": text}

        if (state.get(state_keys.PENDING_WHERE) and re.search(r"outfit|look|separate|on (its|their) own", text)) \
                or (state.get(state_keys.CONFIRM_SIZES) and re.fullmatch(r"(yes,? )?(add them|change sizes)", text)):
            return TurnUnderstanding(intent="other", **base)            # the answer is read where it is used

        pending = state.get(state_keys.PENDING_SIZE)
        if pending:                                   # the answer to "what size?" (or a size button)
            scale = (SizeScale.model_validate(pending["scale"]) if pending.get("outfit")
                     else size_scale(Product.model_validate(pending["product"])))
            size = read_size(text, scale) if scale else None
            if size and len(text.split()) <= 4:
                return TurnUnderstanding(intent="choose", size=size, **base)

        for pattern, intent in _FAST_POSITIONAL:
            match = pattern.match(text)
            if match:
                position = _position(match.group("n") or match.group("m"))
                if position and position <= presented:
                    return TurnUnderstanding(intent=intent, product_position=position, **base)
                return None
        if state.get(state_keys.SUGGESTION):
            numbered = _FAST_SUGGESTION_N.match(text)
            if numbered:
                n = int(numbered.group("n"))
                if 1 <= n <= len(state_keys.suggestions(state)):
                    intent = "choose" if numbered.group("verb") == "add" else "ask_about_product"
                    return TurnUnderstanding(intent=intent, about_suggestion=True, suggestion_position=n, **base)
            if _FAST_ADD_SUGGESTED.match(text):
                return TurnUnderstanding(intent="choose", about_suggestion=True, **base)
            if _FAST_SIMILAR_TO_SUGGESTED.match(text):
                return TurnUnderstanding(intent="see_suggestion", about_suggestion=True, **base)
        wishlist_turn = _fast_wishlist_turn(text, presented, len(state_keys.suggestions(state)))
        if wishlist_turn is not None:
            return TurnUnderstanding(intent="wishlist", **wishlist_turn, **base)
        if presented and _FAST_MORE.match(text):
            return TurnUnderstanding(intent="more_options", **base)
        if presented > 1 and _FAST_COMPARE.match(text):
            return TurnUnderstanding(intent="compare", **base)
        outfit = state_keys.outfit(state) if state_keys.planning_outfit(state) else None
        step = _fast_outfit_step(text, outfit) if outfit is not None else None
        if step is not None:
            return TurnUnderstanding(intent="outfit", outfit=step, **base)
        return None

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


# Commands understood in code (see ScoutOrchestrator._fast_turn). Strict on purpose: anything less clear
# goes to the Elicitation model.
def _ordinal(name: str) -> str:
    return rf"(?P<{name}>first|second|third|fourth|1st|2nd|3rd|4th|one|two|three|four|[1-9][0-9]?)"
_FAST_POSITIONAL = [
    (re.compile(rf"^(i'?ll |i will |let'?s )?(choose|pick|go with|take) option {_ordinal('n')}( for the outfit)?( please)?$"),
     "choose"),
    (re.compile(rf"^(please )?(add|put) (option {_ordinal('n')}|(the )?{_ordinal('m')}( one)?)( (to|in|into) (my|the) cart)?( please)?$"),
     "choose"),
    (re.compile(rf"^(describe option {_ordinal('n')} in detail|(please )?(tell me (more )?about|describe) (the )?{_ordinal('m')}"
                rf"( one| option)?( in detail)?( please)?)$"),
     "ask_about_product"),
]
_FAST_SUGGESTION_N = re.compile(r"^(?P<verb>add|tell me about) suggestion (?P<n>[1-9])( to my cart)?$")
_FAST_ADD_SUGGESTED = re.compile(r"^add the suggested [a-z' -]+ to my cart$")
_FAST_SIMILAR_TO_SUGGESTED = re.compile(r"^show me other options like the suggested [a-z' -]+$")

_FAST_MORE = re.compile(r"^(please )?((show|give|find) me |can you show me |let me see |any )?(some |a few )?"
                        r"(more|other|different|some other)( options| ones| products| items| choices)?( please)?$")
_FAST_COMPARE = re.compile(r"^compare (these|them)( for me)?( please)?$")
# The wishlist buttons
_FAST_SAVE = re.compile(rf"^save option {_ordinal('n')}( (to|for) (my wishlist|later))?$")
_FAST_SAVE_SUGGESTION = re.compile(r"^save suggestion (?P<n>[1-9])( (to|for) (my wishlist|later))?$")
_FAST_SHOW_WISHLIST = re.compile(r"^(show|open|read) (me )?my wishlist$")
_FAST_WISHLIST_ITEMS = re.compile(r"^(?P<verb>compare|remove) wishlist items? (?P<items>[0-9]+((, | and |, and )[0-9]+)*)$")
_ORDINALS = {"first": 1, "1st": 1, "one": 1, "second": 2, "2nd": 2, "two": 2,
             "third": 3, "3rd": 3, "three": 3, "fourth": 4, "4th": 4, "four": 4}


# The outfit board's buttons, and the answers to "which look?".
_FAST_OUTFIT_PIECE = re.compile(r"^(?P<verb>change|remove) the (?P<category>.+?) (in|from) look (?P<n>\d+)$")
_FAST_WHICH_LOOK = re.compile(r"^(?:(?P<all>all looks|all of them|every look|both looks|both)|(just |only )?look (?P<n>\d+))$")
_FAST_OUTFIT_TO_CART = re.compile(r"^add the new pieces from the plan to my cart$")


def _fast_outfit_step(text: str, outfit: Outfit) -> Optional[OutfitStep]:
    if _FAST_OUTFIT_TO_CART.match(text):
        return OutfitStep(action="to_cart")
    match = _FAST_OUTFIT_PIECE.match(text)
    if match and outfit.looks_with(match["category"]):
        return OutfitStep(action=match["verb"], category=match["category"], looks=[int(match["n"])])
    waiting = outfit.changing
    match = _FAST_WHICH_LOOK.match(text)
    if match and waiting is not None and not waiting.looks:
        return OutfitStep(action=waiting.action, category=waiting.category, wish=waiting.wish or None,
                          all_looks=bool(match["all"]), looks=[int(match["n"])] if match["n"] else [])
    return None


def _fast_wishlist_turn(text: str, presented: int, suggested: int) -> Optional[dict]:
    """The wishlist buttons: what to set on the turn, or None."""
    if _FAST_SHOW_WISHLIST.match(text):
        return {"wishlist_action": "view"}
    match = _FAST_SAVE.match(text)
    if match and (position := _position(match["n"])) and position <= presented:
        return {"wishlist_action": "save", "product_position": position}
    match = _FAST_SAVE_SUGGESTION.match(text)
    if match and 1 <= int(match["n"]) <= suggested:
        return {"wishlist_action": "save", "about_suggestion": True, "suggestion_position": int(match["n"])}
    match = _FAST_WISHLIST_ITEMS.match(text)
    if match:
        return {"wishlist_action": match["verb"], "wishlist_positions": [int(n) for n in re.findall(r"[0-9]+", match["items"])]}
    return None


def _position(word: Optional[str]) -> Optional[int]:
    if not word:
        return None
    return int(word) if word.isdigit() else _ORDINALS.get(word)
