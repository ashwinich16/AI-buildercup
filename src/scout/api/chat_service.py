"""
ScoutChatService: runs the Scout agent for one shopper and turns session
state into what the UI shows (spoken reply, product cards, cart).

Two kinds of memory:
  - the ADK session (in memory): only the current conversation, such as what is
    being searched and which options were just presented. After every turn it is also saved as one
    compact summary row (ConversationStore, Cloud SQL), so if this server restarts, or the shopper's
    next message reaches another Cloud Run instance, the conversation is rebuilt from it and carries on.
  - the ShopperRepository (Cloud SQL): what Scout remembers about the shopper
    (name, preferences, cart, the clothes they own). It is loaded into the session when a conversation
    starts, and saved after any turn that changed it.
"""

import asyncio
import logging
import re
import time
from typing import AsyncIterator, Literal, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from scout.agents import state_keys
from scout.agents.elicitation import read_turn
from scout.agents.model_io import content_text
from scout.api.shopper_repository import ConversationStore, ShopperMemory, ShopperRepository
from scout.domain.sizes import size_scale
from scout.domain import Outfit, Preference, PreferenceProfile, Product, Traits, WardrobeItem
from scout.tools import Cart, CartItem, EvidencePicker, WishlistItem
from scout.tools.taste_reader import TasteReader

logger = logging.getLogger(__name__)

APP_NAME = "scout"
LEARNING_APP = "scout-learning"      # the Preference agent's own runner, used after a turn has ended
_EVIDENCE = EvidencePicker()


class EvidenceView(BaseModel):
    quote: str            # word for word from a real review
    count: str            # e.g. "3 of 10 reviews"
    heads_up: bool = False


class ProductCard(BaseModel):
    id: str
    position: int
    name: str
    price: Optional[float] = None
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    image_url: Optional[str] = None
    summary: str = ""
    exact_match: bool = True
    evidence: list[EvidenceView] = Field(default_factory=list)  

class ProductFact(BaseModel):
    name: str                                              
    value: str                                             


class ReviewPointView(BaseModel):
    aspect: str                                           
    finding: str                                           
    count: str = ""                                        
    heads_up: bool = False


class ProductDetail(BaseModel):
    """Everything about one product, for the shopper to read for themselves."""
    id: str
    name: str                                              # short, for the heading
    title: str                                             # the full listing title
    kind: Optional[str] = None
    price: Optional[float] = None
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    images: list[str] = Field(default_factory=list)
    summary: str = ""
    facts: list[ProductFact] = Field(default_factory=list)
    evidence: list[EvidenceView] = Field(default_factory=list)          # quotes that back it, with counts
    review_points: list[ReviewPointView] = Field(default_factory=list)  # what reviewers mention most
    reviews_read: Optional[int] = None                                 # how many written reviews these come from
    occasion: Optional[str] = None                                     # the outfit's occasion, if it is in one
    occasion_note: Optional[str] = None                                # why it suits that occasion
    size: Optional[str] = None                                         # the size it will be added in
    sizes: list[str] = Field(default_factory=list)                     # the sizes it comes in
    sizes_spoken: Optional[str] = None                                 # e.g. "UK shoe sizes 6 to 12"


class CartView(BaseModel):
    items: list[CartItem] = Field(default_factory=list)
    total: float = 0.0


class WishlistView(BaseModel):
    items: list[WishlistItem] = Field(default_factory=list)


class SmallProductCard(BaseModel):
    """A product shown small: the suggested item, or a new piece on the outfit board."""
    id: str
    name: str
    price: Optional[float] = None
    image_url: Optional[str] = None
    goes_with: str = ""
    kind: str = ""
    reasons: list[str] = Field(default_factory=list)


class BoardTile(BaseModel):
    """One piece of a look: something they own, or a new piece from the shop."""
    category: str                                          # what the Change / Remove buttons name
    description: str
    image_url: Optional[str] = None
    price: Optional[float] = None                          # a new piece's price
    product_id: Optional[str] = None                       # a new piece's product, to show its details
    size: Optional[str] = None                             # the size it will be added in ("L", "32 waist")
    needs_size: bool = False                               # it comes in sizes and none is chosen yet
    owned: bool = False
    missing: bool = False                                  # a new piece nothing suitable was found for
    changing: bool = False                                 # the options on screen are for this piece


class LookView(BaseModel):
    title: str                                             # e.g. "Day 1 · client meeting"
    why: str = ""
    tiles: list[BoardTile] = Field(default_factory=list)
    product_ids: list[str] = Field(default_factory=list)   # the new pieces in it, to try the look on
    wardrobe_ids: list[str] = Field(default_factory=list)  # their own pieces in it, tried on with the new ones


class CapsuleView(BaseModel):
    """The outfit board: the looks, each with its own pieces, and the new products to buy."""
    occasion: str
    looks: list[LookView] = Field(default_factory=list)
    to_buy: list[SmallProductCard] = Field(default_factory=list)   # every new product once
    total: float = 0.0
    budget: Optional[float] = None
    changing: Optional[str] = None                         # e.g. "shoe for look 1": what the options are for


class ScreenState(BaseModel):
    products: list[ProductCard] = Field(default_factory=list)
    focus_product_id: Optional[str] = None
    cart: CartView = Field(default_factory=CartView)
    wishlist: WishlistView = Field(default_factory=WishlistView)
    suggestion: Optional[SmallProductCard] = None
    suggestions: list[SmallProductCard] = Field(default_factory=list)   # the few options shown in the pop-up
    # The product Scout is describing when it is not one of the cards above (e.g. the suggested item):
    # shown large at the top of the page, so the shopper sees what Scout is talking about.
    spotlight: Optional[ProductCard] = None
    capsule: Optional[CapsuleView] = None
    planning_outfit: bool = False     # outfit mode: everything is for the outfit until it is finished or left
    wardrobe: list[WardrobeItem] = Field(default_factory=list)


class ChatTurn(ScreenState):
    reply: str
    action: Optional[str] = None      # "pause" (go quiet, keep listening) or "end" (stop listening)
    choices: list[str] = Field(default_factory=list)   # short answers to Scout's question, to tap
    choose_one: bool = False                            # one tap answers it (a size), rather than tap several


class ChatMessage(BaseModel):
    role: Literal["shopper", "scout"]
    text: str


class StartedSession(ScreenState):
    session_id: str
    greeting: str
    resumed: bool = False                                     # carrying on their last conversation
    history: list[ChatMessage] = Field(default_factory=list)  # that conversation so far, to show again
    shopper_name: str = ""            # what Scout calls the shopper
    categories: list[str] = Field(default_factory=list)   # what the shop sells: first things to ask for
    try_on_adjustable: bool = False   # the try-on can follow "tuck the shirt in" (so the screen offers it)


class ScoutChatService:
    def __init__(self, agent: BaseAgent, shoppers: ShopperRepository, conversations: ConversationStore,
                 taste_reader: Optional[TasteReader] = None, learner: Optional[BaseAgent] = None):
        self._shoppers = shoppers
        self._conversations = conversations
        self._taste_reader = taste_reader        # learns their taste from what they buy, save and own
        # The Preference agent learns from a turn after it has ended, in a scratch session of its own, so the
        # shopper can carry on talking while it thinks.
        self._learning_sessions = InMemorySessionService()
        self._learning = Runner(app_name=LEARNING_APP, agent=learner,
                                session_service=self._learning_sessions) if learner else None
        self._learning_lock = asyncio.Lock()     # one learning at a time, so two never overwrite each other
        self._saving: set[asyncio.Task] = set()    # conversation saves running in the background
        self._sessions = InMemorySessionService()
        self._runner = Runner(app_name=APP_NAME, agent=agent, session_service=self._sessions)

    async def start_session(self, user_id: str, display_name: str = "",
                            seed_preferences: Optional[list[Preference]] = None,
                            preferred_name: str = "", fresh: bool = False) -> StartedSession:
        """Carry on the shopper's last conversation (as they left it, with what was said), or start a new one
        when there is none or they asked for a fresh one (`fresh`). With Scout's opening words.

        Name: the one the shopper just typed (remembered for next time), else the one
        they typed before, else the name from their account. Demo seed preferences
        are applied only the very first time a shopper visits.
        """
        memory = await asyncio.to_thread(self._shoppers.load, user_id)
        returning = memory.known
        renamed = bool(preferred_name) and preferred_name != memory.preferred_name
        if renamed:
            memory.preferred_name = preferred_name
        if seed_preferences and not returning:
            memory.preferences = PreferenceProfile().merge(seed_preferences, [])
        if renamed or not returning:        
            await asyncio.to_thread(self._shoppers.save, user_id, memory)

        name = memory.preferred_name or display_name
        session = None if fresh else await self._last_conversation(user_id, memory)
        if session is not None:
            return StartedSession(session_id=session.id, greeting=_welcome_back(name), shopper_name=name,
                                  resumed=True, history=_history(session.state),
                                  **_screen_from_state(session.state).model_dump())

        session = await self._sessions.create_session(app_name=APP_NAME, user_id=user_id,
                                                      state=_state_from_memory(memory))
        await self._keep_conversation(user_id, session)   
        greeting = _greeting(name, returning, memory.cart)
        return StartedSession(session_id=session.id, greeting=greeting, shopper_name=name,
                              **_screen_from_state(session.state).model_dump())

    async def _last_conversation(self, user_id: str, memory: ShopperMemory):
        """The shopper's most recent conversation, if there is one worth carrying on (something was said)."""
        try:
            session_id = await asyncio.to_thread(self._conversations.latest_conversation, user_id)
            session = await self._get(user_id, session_id, memory) if session_id else None
        except Exception:                   
            logger.exception("Could not load the last conversation for shopper %s", user_id)
            return None
        return session if session is not None and state_keys.conversation(session.state) else None

    async def send_stream(self, user_id: str, session_id: str, text: str) -> AsyncIterator[dict]:
        """One turn of the conversation. Scout's words arrive while they are being written, so the shopper
        hears the first sentence within a second or two instead of waiting for the whole reply.

        Yields, in order:
          {"type": "delta", "text": ...}       a piece of the reply (append it)
          {"type": "screen", products, ...}    what to show, as soon as a spoken message is complete
          {"type": "done", reply, action, ...} the whole turn, after any background learning finished
        """
        before = _memory_from_state((await self._get(user_id, session_id)).state)
        message = types.Content(role="user", parts=[types.Part(text=text)])
        streaming = RunConfig(streaming_mode=StreamingMode.SSE)
        replies: list[str] = []
        streamed = False          
        async for event in self._runner.run_async(user_id=user_id, session_id=session_id,
                                                  new_message=message, run_config=streaming):
            raw = content_text(event.content)
            if not raw:
                continue
            if getattr(event, "partial", False):
                piece = _FORMATTING.sub("", raw)
                if piece:
                    if not streamed and replies:
                        piece = "\n\n" + piece.lstrip()   
                    streamed = True
                    yield {"type": "delta", "text": piece}
                continue
            whole = _FORMATTING.sub("", raw).strip()
            if not streamed and whole:                     
                yield {"type": "delta", "text": ("\n\n" if replies else "") + whole}
            if whole:
                replies.append(whole)
            streamed = False
            state = (await self._get(user_id, session_id)).state
            # The wardrobe (with its thumbnails) is sent once, at the end of the turn, not with every update.
            yield {"type": "screen", **_screen_from_state(state).model_dump(mode="json", exclude={"wardrobe"})}

        session = await self._get(user_id, session_id)
        await self._remember(user_id, before, _memory_from_state(session.state))
        self._keep_conversation_later(user_id, session)  
        if session.state.get(state_keys.TASTE_EVIDENCE) or session.state.get(state_keys.PREFERENCE_PENDING):
            self._in_background(self._learn(user_id, session_id))  
        turn = read_turn(session.state)
        action = _ACTIONS.get(turn.intent) if turn else None
        reply = "\n\n".join(replies)
        if not reply and action != "pause":
            reply = "Sorry, I lost my train of thought. Could you say that again?"
            yield {"type": "delta", "text": reply}
        choices = turn.choices if turn and turn.question and not turn.ready_to_search else []
        choose_one = bool(session.state.get(state_keys.PENDING_SIZE) or session.state.get(state_keys.PENDING_WHERE)
                          or session.state.get(state_keys.CONFIRM_SIZES))
        if session.state.get(state_keys.PENDING_SIZE):               
            choices = list(session.state.get(state_keys.SIZE_CHOICES) or [])
        done = ChatTurn(reply=reply, action=action, choices=choices, choose_one=choose_one,
                        **_screen_from_state(session.state).model_dump())
        yield {"type": "done", **done.model_dump(mode="json")}

    async def end_session(self, user_id: str, session_id: str) -> None:
        """Forget this conversation (the shopper cleared it). What Scout remembers about them is kept."""
        await self._get(user_id, session_id)           
        await self._sessions.delete_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
        await asyncio.to_thread(self._conversations.delete_conversation, user_id, session_id)

    async def ensure_session(self, user_id: str, session_id: str) -> None:
        """Raises LookupError if this shopper has no such conversation."""
        await self._get(user_id, session_id)

    async def products(self, user_id: str, session_id: str, product_ids: list[str]) -> list[Product]:
        """Products this conversation has shown (options, the suggestion, the board's new pieces), in the
        order asked. Raises LookupError if this shopper has no such conversation."""
        state = (await self._get(user_id, session_id)).state
        known = dict(state_keys.shortlist(state))
        suggestion, outfit = state_keys.suggestion(state), state_keys.outfit(state)
        if suggestion:
            known[suggestion.product.id] = suggestion.product
        if outfit:
            known.update({product.id: product for product in outfit.products_to_buy()})
        return [known[i] for i in product_ids if i in known]

    async def product_details(self, user_id: str, session_id: str, product_id: str) -> ProductDetail:
        """One product this conversation has shown, in full. Raises LookupError if it is not one of them."""
        found = await self.products(user_id, session_id, [product_id])
        if not found:
            raise LookupError(product_id)
        product = found[0]
        state = (await self._get(user_id, session_id)).state
        outfit = state_keys.outfit(state)
        scale = size_scale(product)
        facts, seen = [], set()
        for attribute in product.attributes:
            key = (attribute.name.lower(), attribute.value.lower())
            if key not in seen and len(attribute.value) <= 40:
                seen.add(key)
                facts.append(ProductFact(name=attribute.name, value=attribute.value))
        points = [ReviewPointView(aspect=point.aspect, finding=point.finding,
                                  count=(f"{point.count()} of {product.reviews_analyzed} reviews"
                                         if point.count() and product.reviews_analyzed else ""),
                                  heads_up=point.sentiment == "negative")
                  for point in product.review_points[:5] if point.finding]
        return ProductDetail(
            id=product.id, name=_card_name(product), title=product.title, kind=product.kind or None,
            price=product.price, rating=product.rating, rating_count=product.rating_count,
            images=product.image_urls[:6],
            summary=product.spoken_summary_detailed or product.spoken_summary_short or (product.description or "")[:400],
            facts=facts[:8],
            evidence=[EvidenceView(quote=e.quote, count=e.count_text(), heads_up=e.sentiment == "negative")
                      for e in _EVIDENCE.pick(product, state_keys.need(state))],
            review_points=points, reviews_read=product.reviews_analyzed,
            occasion=outfit.occasion if outfit else None,
            occasion_note=outfit.fit_notes.get(product.id) if outfit else None,
            size=_size_label(product, state.get(state_keys.KNOWN_SIZES) or {}),
            sizes=scale.options if scale else [], sizes_spoken=scale.spoken if scale else None,
        )

    async def wardrobe_items(self, user_id: str, session_id: str, item_ids: list[str]) -> list[WardrobeItem]:
        """Pieces the shopper owns, in the order asked (e.g. to try a look on)."""
        owned = {item.id: item for item in state_keys.wardrobe((await self._get(user_id, session_id)).state)}
        return [owned[i] for i in item_ids if i in owned]

    async def add_to_wardrobe(self, user_id: str, session_id: str, added: list[WardrobeItem],
                              traits: Optional[list[tuple[Traits, str]]] = None) -> list[WardrobeItem]:
        """Remember pieces the shopper owns (from a photo), and what each says about their taste (its traits and
        the style it is cut for). Scout mentions them in its next reply. Returns the whole wardrobe."""
        return await self._change_wardrobe(user_id, session_id, lambda items: [*items, *added],
                                           just_added=[item.id for item in added], traits=traits)

    async def remove_from_wardrobe(self, user_id: str, session_id: str, item_id: str) -> list[WardrobeItem]:
        return await self._change_wardrobe(user_id, session_id,
                                           lambda items: [i for i in items if i.id != item_id])

    async def _change_wardrobe(self, user_id: str, session_id: str, change, just_added: Optional[list[str]] = None,
                               traits: Optional[list[tuple[Traits, str]]] = None) -> list[WardrobeItem]:
        """Change the wardrobe in this conversation (so Scout knows at once) and in the shopper's memory."""
        session = await self._get(user_id, session_id)
        before = _memory_from_state(session.state)
        wardrobe = change(state_keys.wardrobe(session.state))
        kept = {item.id for item in wardrobe}
        shown = [i for i in [*(session.state.get(state_keys.WARDROBE_JUST_ADDED) or []), *(just_added or [])]
                 if i in kept]
        taste = before.taste.model_copy(deep=True)
        for piece_traits, made_for in traits or []:          
            taste.learn(piece_traits, made_for, "owned")
        delta = {state_keys.WARDROBE: [item.model_dump() for item in wardrobe],
                 state_keys.WARDROBE_JUST_ADDED: shown, state_keys.TASTE: taste.model_dump()}
        await self._sessions.append_event(session, Event(author="user", actions=EventActions(state_delta=delta)))
        await self._remember(user_id, before, before.model_copy(update={"wardrobe": wardrobe, "taste": taste}))
        return wardrobe

    async def _learn(self, user_id: str, session_id: str) -> None:
        """Everything this turn taught, after it has ended: lasting preferences, then taste."""
        async with self._learning_lock:
            await self._learn_preferences(user_id, session_id)
            await self._learn_taste(user_id, session_id)

    async def _learn_preferences(self, user_id: str, session_id: str) -> None:
        """The Preference agent learns from each turn waiting for it, in order, starting from their profile as it
        is now. Then the result goes into the conversation and their memory. A failure teaches nothing."""
        try:
            session = await self._get(user_id, session_id)
            pending = list(session.state.get(state_keys.PREFERENCE_PENDING) or [])
            if not pending or self._learning is None:
                return
            started = time.perf_counter()
            known = {state_keys.PREFERENCES: session.state.get(state_keys.PREFERENCES) or {},
                     state_keys.TASTE: session.state.get(state_keys.TASTE) or {}}
            taste_before = {mark.value for mark in state_keys.taste(known).marks}
            for item in pending:
                known = await self._run_learner(user_id, known, item)
            session = await self._get(user_id, session_id)          
            before = _memory_from_state(session.state)
            taste = before.taste.model_copy(deep=True)
            taste.forget(sorted(taste_before - {mark.value for mark in state_keys.taste(known).marks}))  
            preferences = state_keys.profile(known)
            left = [p for p in session.state.get(state_keys.PREFERENCE_PENDING) or [] if p not in pending]
            delta = {state_keys.PREFERENCES: preferences.model_dump(), state_keys.TASTE: taste.model_dump(),
                     state_keys.PREFERENCE_PENDING: left}
            await self._sessions.append_event(session, Event(author="user", actions=EventActions(state_delta=delta)))
            await self._remember(user_id, before, before.model_copy(update={"preferences": preferences,
                                                                            "taste": taste}))
            logger.info("Learned from %d turns in %.1fs", len(pending), time.perf_counter() - started)
        except Exception:
            logger.exception("Could not learn preferences for shopper %s", user_id)

    async def _run_learner(self, user_id: str, known: dict, item: dict) -> dict:
        """One run of the Preference agent, on a scratch session holding only what it reads and writes."""
        scratch = await self._learning_sessions.create_session(
            app_name=LEARNING_APP, user_id=user_id, state={**known, state_keys.PREFERENCE_EVIDENCE: item.get("evidence")})
        message = types.Content(role="user", parts=[types.Part(text=item.get("message") or "")])
        try:
            async for _ in self._learning.run_async(user_id=user_id, session_id=scratch.id, new_message=message):
                pass
            scratch = await self._learning_sessions.get_session(app_name=LEARNING_APP, user_id=user_id,
                                                                session_id=scratch.id)
            return {key: scratch.state.get(key) or {} for key in known}
        finally:
            await self._learning_sessions.delete_session(app_name=LEARNING_APP, user_id=user_id, session_id=scratch.id)

    async def _learn_taste(self, user_id: str, session_id: str) -> None:
        """What was bought, saved or added this turn: each item's traits are read (one call for all of them) and
        counted into their taste. Runs after the turn, so it never holds up a reply; a failure teaches nothing."""
        try:
            session = await self._get(user_id, session_id)
            queued = list(session.state.get(state_keys.TASTE_EVIDENCE) or [])
            if not queued or self._taste_reader is None:
                return
            known = sorted({mark.value for mark in state_keys.taste(session.state).marks})
            traits = await asyncio.to_thread(self._taste_reader.read, [q["text"] for q in queued], known)
            session = await self._get(user_id, session_id)         
            before = _memory_from_state(session.state)
            taste = before.taste.model_copy(deep=True)
            for item, item_traits in zip(queued, traits):
                taste.learn(item_traits, item.get("style"), item["signal"])
            left = [q for q in session.state.get(state_keys.TASTE_EVIDENCE) or [] if q not in queued]
            delta = {state_keys.TASTE: taste.model_dump(), state_keys.TASTE_EVIDENCE: left}
            await self._sessions.append_event(session, Event(author="user", actions=EventActions(state_delta=delta)))
            await self._remember(user_id, before, before.model_copy(update={"taste": taste}))
        except Exception:
            logger.exception("Could not learn taste for shopper %s", user_id)

    def _in_background(self, work) -> None:
        task = asyncio.create_task(work)
        self._saving.add(task)                       
        task.add_done_callback(self._saving.discard)

    async def _get(self, user_id: str, session_id: str, memory: Optional[ShopperMemory] = None):
        """This shopper's conversation: from memory, or rebuilt from its saved summary (after a restart, or
        on another instance), with `memory` if it was just loaded. Raises LookupError if there is neither."""
        session = await self._sessions.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
        if session is not None:
            return session
        summary = await asyncio.to_thread(self._conversations.load_conversation, user_id, session_id)
        if summary is None:
            raise LookupError("Conversation not found")
        if memory is None:                 
            memory = await asyncio.to_thread(self._shoppers.load, user_id)
        try:
            return await self._sessions.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id,
                                                       state={**summary, **_state_from_memory(memory)})
        except Exception:      
            session = await self._sessions.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
            if session is None:
                raise
            return session

    def _keep_conversation_later(self, user_id: str, session) -> None:
        """Save the conversation in the background, so the turn ends without waiting for the database.
        The summary is taken now, so it is exactly this turn's."""
        task = asyncio.create_task(self._save_summary(user_id, session.id, _summary(session.state)))
        self._saving.add(task)                       
        task.add_done_callback(self._saving.discard)

    async def _keep_conversation(self, user_id: str, session) -> None:
        await self._save_summary(user_id, session.id, _summary(session.state))

    async def _save_summary(self, user_id: str, session_id: str, summary: dict) -> None:
        """A failed save never breaks the reply: the conversation is still in this server's memory."""
        try:
            await asyncio.to_thread(self._conversations.save_conversation, user_id, session_id, summary)
        except Exception:
            logger.exception("Could not save the conversation for shopper %s", user_id)

    async def _remember(self, user_id: str, before: ShopperMemory, after: ShopperMemory) -> None:
        """Save the shopper's memory if this turn changed it. A failed save never breaks the reply."""
        if after == before:
            return
        try:
            await asyncio.to_thread(self._shoppers.save, user_id, after)
        except Exception:
            logger.exception("Could not save what Scout learned about shopper %s", user_id)


def _summary(state: dict) -> dict:
    """What is saved of a conversation: its session state, without what the shopper's memory already keeps
    (name, preferences, taste, cart, wishlist, wardrobe) or ADK's per-user keys."""
    return {key: value for key, value in state.items()
            if key not in _MEMORY_KEYS and not key.startswith(("user:", "app:", "temp:"))}


# Session keys that come from the shopper's memory (ShopperRepository), so a conversation summary leaves them out.
_MEMORY_KEYS = {state_keys.SHOPPER_NAME, state_keys.PREFERENCES, state_keys.TASTE, state_keys.CART,
                state_keys.WISHLIST, state_keys.WARDROBE}


def _state_from_memory(memory: ShopperMemory) -> dict:
    return {
        state_keys.SHOPPER_NAME: memory.preferred_name,
        state_keys.PREFERENCES: memory.preferences.model_dump(),
        state_keys.TASTE: memory.taste.model_dump(),
        state_keys.CART: memory.cart.model_dump(),
        state_keys.WISHLIST: memory.wishlist.model_dump(),
        state_keys.WARDROBE: [item.model_dump() for item in memory.wardrobe],
    }


def _memory_from_state(state: dict) -> ShopperMemory:
    return ShopperMemory(
        known=True,
        preferred_name=state.get(state_keys.SHOPPER_NAME) or "",
        preferences=state_keys.profile(state),
        taste=state_keys.taste(state),
        cart=state_keys.cart(state),
        wishlist=state_keys.wishlist(state),
        wardrobe=state_keys.wardrobe(state),
    )


_ACTIONS = {"pause": "pause", "end_conversation": "end"}   # intents the UI acts on (go quiet / stop listening)


def _welcome_back(name: str) -> str:
    """Opening words when carrying on a conversation: short, since what was said is on screen."""
    return f"Welcome back{', ' + name if name else ''}! Let's carry on where we left off."


def _history(state: dict) -> list[ChatMessage]:
    """The conversation so far, as it was shown."""
    messages = []
    for turn in state_keys.conversation(state):
        messages.append(ChatMessage(role="shopper", text=turn.get("shopper", "")))
        reply = _FORMATTING.sub("", turn.get("scout", "")).strip()
        if reply:                              
            messages.append(ChatMessage(role="scout", text=reply))
    return [m for m in messages if m.text]


def _greeting(name: str, returning: bool, cart: Cart) -> str:
    """Scout opens the conversation, and explains once how talking works."""
    hello = f", {name}" if name else ""
    if returning:
        waiting = f" You still have {_spoken_cart(cart)} in your cart." if cart.items else ""
        return f"Welcome back{hello}!{waiting} {_TWO_WAYS}"
    return (f"Hi{' ' + name if name else ''}! I'm Scout, your shopping companion. Just talk to me naturally, and interrupt me "
            "whenever you like. When you hear a soft chime, it's your turn to speak. "
            f"{_TWO_WAYS}")


# The two ways to shop with Scout, offered at the start of every conversation.
_TWO_WAYS = ("Shall we plan an outfit together, using the clothes you already own, "
             "or are you shopping for something in particular?")


# Formatting a model may still add (bold/italic asterisks, headings, code marks, bullet dashes).
# Replies are spoken aloud so these would be read out as symbols.
_FORMATTING = re.compile(r"[*#`_>]+|^[ \t]*[-•][ \t]+", re.MULTILINE)


def _spoken_cart(cart: Cart, named: int = 2) -> str:
    """The cart in a few words: by kind ("the jeans and the shirt"), or how many items with a kind or two.
    Never the long catalogue titles ("526Jeanswear 'SENJO' Mens Designer Branded Stretch"), which are not
    worth hearing read aloud."""
    kinds = list(dict.fromkeys(item.kind for item in cart.items if item.kind))
    count = len(cart.items)
    if count == 1:
        return f"the {kinds[0]}" if kinds else "one item"
    if len(kinds) == count <= 3:
        return ", ".join(f"the {kind}" for kind in kinds[:-1]) + f" and the {kinds[-1]}"
    including = f", including the {' and the '.join(kinds[:named])}" if kinds else ""
    return f"{count} items{including}"


def _screen_from_state(state: dict) -> ScreenState:
    need = state_keys.need(state)

    def card(product: Product, position: int) -> ProductCard:
        return ProductCard(
            id=product.id,
            position=position,
            name=_card_name(product),
            price=product.price,
            rating=product.rating,
            rating_count=product.rating_count,
            image_url=product.image_urls[0] if product.image_urls else None,
            summary=product.short_description,
            exact_match=state_keys.is_exact(state, product.id),
            evidence=[EvidenceView(quote=e.quote, count=e.count_text(), heads_up=e.sentiment == "negative")
                      for e in _EVIDENCE.pick(product, need)],
        )

    presented = state_keys.presented(state)
    cards = [card(product, position) for position, product in enumerate(presented, start=1)]
    focus = state_keys.focus(state)
    # Only while they are asking about it: after, say, adding it to the cart, it should not linger on screen.
    turn = read_turn(state)
    describing = turn is not None and turn.intent == "ask_about_product"
    spotlight = card(focus, 0) if describing and focus and focus.id not in {p.id for p in presented} else None
    cart = state_keys.cart(state)
    return ScreenState(products=cards, focus_product_id=state.get(state_keys.FOCUS_PRODUCT),
                       cart=CartView(items=cart.items, total=cart.total),
                       wishlist=WishlistView(items=state_keys.wishlist(state).items),
                       suggestion=_suggestion_card(state),
                       capsule=_capsule_view(state_keys.outfit(state), state_keys.wardrobe(state),
                                             state.get(state_keys.KNOWN_SIZES) or {}),
                       planning_outfit=state_keys.planning_outfit(state),
                       wardrobe=state_keys.wardrobe(state), spotlight=spotlight,
                       suggestions=[_small_card(s.product).model_copy(update={"goes_with": s.goes_with,
                                                                               "reasons": s.reasons})
                                    for s in state_keys.suggestions(state)])


def _suggestion_card(state: dict) -> Optional[SmallProductCard]:
    suggestion = state_keys.suggestion(state)
    if suggestion is None:
        return None
    return _small_card(suggestion.product).model_copy(update={"goes_with": suggestion.goes_with,
                                                             "reasons": suggestion.reasons})


def _size_label(product: Product, known_sizes: dict) -> Optional[str]:
    """The size a piece will be added in, from the sizes they gave this conversation ("L", "32 waist")."""
    scale = size_scale(product)
    size = known_sizes.get(scale.key) if scale is not None else None
    return scale.label(size) if scale is not None and size in scale.options else None


def _capsule_view(outfit: Optional[Outfit], wardrobe: list[WardrobeItem],
                  known_sizes: Optional[dict] = None) -> Optional[CapsuleView]:
    """The board. Pictures of their own pieces come from their wardrobe (the outfit keeps only the words)."""
    if outfit is None:
        return None
    pictures = {item.id: item.thumbnail for item in wardrobe}
    change = outfit.changing if outfit.changing and outfit.changing.looks else None
    looks = []
    for number, look in enumerate(outfit.looks):
        tiles = []
        for piece in look.pieces:
            changing = change is not None and number in change.looks and \
                piece.category.lower() == change.category.lower()
            if piece.owned:
                item = outfit.owned_item(piece.owned)
                tiles.append(BoardTile(category=piece.category, description=item.description if item else piece.category,
                                       image_url=pictures.get(piece.owned), owned=True, changing=changing))
            elif piece.product is not None:
                card = _small_card(piece.product)
                size = _size_label(piece.product, known_sizes or {})
                tiles.append(BoardTile(category=piece.category, description=card.name, image_url=card.image_url,
                                       price=card.price, product_id=piece.product.id, changing=changing, size=size,
                                       needs_size=size is None and size_scale(piece.product) is not None))
            else:
                tiles.append(BoardTile(category=piece.category, description=f"No {piece.category} found yet",
                                       missing=True, changing=changing))
        looks.append(LookView(title=look.title, why=look.why, tiles=tiles,
                              product_ids=[p.id for p in outfit.look_products(number)],
                              wardrobe_ids=[p.owned for p in look.pieces if p.owned and pictures.get(p.owned)]))
    return CapsuleView(occasion=outfit.occasion, looks=looks, to_buy=[_small_card(p) for p in outfit.products_to_buy()],
                       total=outfit.total(), budget=outfit.budget,
                       changing=f"{change.category} for {change.looks_text()}" if change else None)


def _small_card(product: Product) -> SmallProductCard:
    return SmallProductCard(id=product.id, name=_card_name(product), price=product.price,
                          image_url=product.image_urls[0] if product.image_urls else None, kind=product.kind or "item")


def _card_name(product: Product) -> str:
    """A product's title, short enough for a card."""
    return " ".join(product.title.split()[:9])
