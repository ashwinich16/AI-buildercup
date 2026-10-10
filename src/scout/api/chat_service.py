"""
ScoutChatService: runs the Scout agent for one shopper and turns session
state into what the UI shows (spoken reply, product cards, cart).

Two kinds of memory:
  - the ADK session (in memory): only the current conversation, such as what is
    being searched and which options were just presented
  - the ShopperRepository (Cloud SQL): what Scout remembers about the shopper
    (name, preferences, cart). It is loaded into the session when a conversation
    starts, and saved after any turn that changed it.
"""

import asyncio
import logging
import re
from typing import AsyncIterator, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from scout.agents import state_keys
from scout.agents.elicitation import read_turn
from scout.agents.model_io import content_text
from scout.api.shopper_repository import ShopperMemory, ShopperRepository
from scout.domain import Preference, PreferenceProfile, Product
from scout.tools import Cart, CartItem, EvidencePicker

logger = logging.getLogger(__name__)

APP_NAME = "scout"
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
    evidence: list[EvidenceView] = Field(default_factory=list)   # review quotes that back this option


class CartView(BaseModel):
    items: list[CartItem] = Field(default_factory=list)
    total: float = 0.0


class SmallProductCard(BaseModel):
    """A product shown small: the suggested item, or a picked piece of an outfit."""
    id: str
    name: str
    price: Optional[float] = None
    image_url: Optional[str] = None
    goes_with: str = ""
    kind: str = ""
    reasons: list[str] = Field(default_factory=list)


class OutfitPieceView(BaseModel):
    kind: str                            # the catalog category, e.g. "trousers"
    looks_like: str = ""
    picked: Optional[SmallProductCard] = None   # the product chosen for it, once picked


class OutfitView(BaseModel):
    """The outfit being put together: every piece, picked or still to choose."""
    occasion: str
    pieces: list[OutfitPieceView] = Field(default_factory=list)
    total: float = 0.0


class ScreenState(BaseModel):
    products: list[ProductCard] = Field(default_factory=list)
    focus_product_id: Optional[str] = None
    cart: CartView = Field(default_factory=CartView)
    suggestion: Optional[SmallProductCard] = None
    suggestions: list[SmallProductCard] = Field(default_factory=list)   # the few options shown in the pop-up
    # The product Scout is describing when it is not one of the cards above (e.g. the suggested item):
    # shown large at the top of the page, so the shopper sees what Scout is talking about.
    spotlight: Optional[ProductCard] = None
    outfit: Optional[OutfitView] = None


class ChatTurn(ScreenState):
    reply: str
    action: Optional[str] = None      # "pause" (go quiet, keep listening) or "end" (stop listening)


class StartedSession(ScreenState):
    session_id: str
    greeting: str
    shopper_name: str = ""            # what Scout calls the shopper
    categories: list[str] = Field(default_factory=list)   # what the shop sells: first things to ask for


class ScoutChatService:
    def __init__(self, agent: BaseAgent, shoppers: ShopperRepository):
        self._shoppers = shoppers
        self._sessions = InMemorySessionService()
        self._runner = Runner(app_name=APP_NAME, agent=agent, session_service=self._sessions)

    async def start_session(self, user_id: str, display_name: str = "",
                            seed_preferences: Optional[list[Preference]] = None,
                            preferred_name: str = "") -> StartedSession:
        """New conversation, with Scout's opening words.

        Name: the one the shopper just typed (remembered for next time), else the one
        they typed before, else the name from their account. Demo seed preferences
        are applied only the very first time a shopper visits.
        """
        memory = await asyncio.to_thread(self._shoppers.load, user_id)
        returning = memory.known
        if preferred_name:
            memory.preferred_name = preferred_name
        if seed_preferences and not returning:
            memory.preferences = PreferenceProfile().merge(seed_preferences, [])
        await asyncio.to_thread(self._shoppers.save, user_id, memory)   # also records this visit

        session = await self._sessions.create_session(app_name=APP_NAME, user_id=user_id,
                                                      state=_state_from_memory(memory))
        name = memory.preferred_name or display_name
        greeting = _greeting(name, returning, memory.cart)
        return StartedSession(session_id=session.id, greeting=greeting, shopper_name=name,
                              **_screen_from_state(session.state).model_dump())

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
        streamed = False          # pieces of the current message were already sent
        async for event in self._runner.run_async(user_id=user_id, session_id=session_id,
                                                  new_message=message, run_config=streaming):
            raw = content_text(event.content)
            if not raw:
                continue
            if getattr(event, "partial", False):
                piece = _FORMATTING.sub("", raw)
                if piece:
                    if not streamed and replies:
                        piece = "\n\n" + piece.lstrip()    # a second message in the same turn
                    streamed = True
                    yield {"type": "delta", "text": piece}
                continue
            whole = _FORMATTING.sub("", raw).strip()
            if not streamed and whole:                     # said in one go (e.g. a question): send it now
                yield {"type": "delta", "text": ("\n\n" if replies else "") + whole}
            if whole:
                replies.append(whole)
            streamed = False
            state = (await self._get(user_id, session_id)).state
            yield {"type": "screen", **_screen_from_state(state).model_dump(mode="json")}

        session = await self._get(user_id, session_id)
        await self._remember(user_id, before, _memory_from_state(session.state))
        turn = read_turn(session.state)
        action = _ACTIONS.get(turn.intent) if turn else None
        reply = "\n\n".join(replies)
        if not reply and action != "pause":
            reply = "Sorry, I lost my train of thought. Could you say that again?"
            yield {"type": "delta", "text": reply}
        done = ChatTurn(reply=reply, action=action, **_screen_from_state(session.state).model_dump())
        yield {"type": "done", **done.model_dump(mode="json")}

    async def ensure_session(self, user_id: str, session_id: str) -> None:
        """Raises LookupError if this shopper has no such conversation."""
        await self._get(user_id, session_id)

    async def products(self, user_id: str, session_id: str, product_ids: list[str]) -> list[Product]:
        """Products this conversation has shown (options, the suggestion, outfit pieces), in the order asked.
        Raises LookupError if this shopper has no such conversation."""
        state = (await self._get(user_id, session_id)).state
        known = dict(state_keys.shortlist(state))
        suggestion, outfit = state_keys.suggestion(state), state_keys.outfit(state)
        if suggestion:
            known[suggestion.product.id] = suggestion.product
        if outfit:
            known.update({product.id: product for product in outfit.chosen()})
        return [known[i] for i in product_ids if i in known]

    async def _get(self, user_id: str, session_id: str):
        session = await self._sessions.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
        if session is None:
            raise LookupError("Conversation not found")
        return session

    async def _remember(self, user_id: str, before: ShopperMemory, after: ShopperMemory) -> None:
        """Save the shopper's memory if this turn changed it. A failed save never breaks the reply."""
        if after == before:
            return
        try:
            await asyncio.to_thread(self._shoppers.save, user_id, after)
        except Exception:
            logger.exception("Could not save what Scout learned about shopper %s", user_id)


def _state_from_memory(memory: ShopperMemory) -> dict:
    return {
        state_keys.SHOPPER_NAME: memory.preferred_name,
        state_keys.PREFERENCES: memory.preferences.model_dump(),
        state_keys.CART: memory.cart.model_dump(),
    }


def _memory_from_state(state: dict) -> ShopperMemory:
    return ShopperMemory(
        known=True,
        preferred_name=state.get(state_keys.SHOPPER_NAME) or "",
        preferences=state_keys.profile(state),
        cart=state_keys.cart(state),
    )


_ACTIONS = {"pause": "pause", "end_conversation": "end"}   # intents the UI acts on (go quiet / stop listening)


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
_TWO_WAYS = "Shall we plan an outfit together for an occasion, or are you shopping for something in particular?"


# Formatting a model may still add (bold/italic asterisks, headings, code marks, bullet dashes).
# Replies are spoken aloud, so these would be read out as symbols.
_FORMATTING = re.compile(r"[*#`_>]+|^[ \t]*[-•][ \t]+", re.MULTILINE)


def _spoken_cart(cart: Cart, named: int = 2) -> str:
    """The cart in a few words: the first items by a short name, then how many more."""
    # By kind ("the jeans"), not the long catalogue title read aloud ("Jeans Men Stretch Slim Fit Straight").
    names = [f"the {item.kind}" if item.kind else " ".join(item.name.split()[:6]) for item in cart.items[:named]]
    others = len(cart.items) - len(names)
    if others:
        names.append(f"{others} more {'item' if others == 1 else 'items'}")
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


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
                       suggestion=_suggestion_card(state), outfit=_outfit_view(state), spotlight=spotlight,
                       suggestions=[_small_card(s.product).model_copy(update={"goes_with": s.goes_with,
                                                                               "reasons": s.reasons})
                                    for s in state_keys.suggestions(state)])


def _suggestion_card(state: dict) -> Optional[SmallProductCard]:
    suggestion = state_keys.suggestion(state)
    if suggestion is None:
        return None
    return _small_card(suggestion.product).model_copy(update={"goes_with": suggestion.goes_with,
                                                             "reasons": suggestion.reasons})


def _outfit_view(state: dict) -> Optional[OutfitView]:
    outfit = state_keys.outfit(state)
    if outfit is None:
        return None
    pieces = [OutfitPieceView(kind=piece.category, looks_like=piece.looks_like,
                              picked=_small_card(product) if product else None)
              for piece, product in zip(outfit.plan.pieces, outfit.picked)]
    return OutfitView(occasion=outfit.plan.occasion, pieces=pieces, total=outfit.total())


def _small_card(product: Product) -> SmallProductCard:
    return SmallProductCard(id=product.id, name=_card_name(product), price=product.price,
                          image_url=product.image_urls[0] if product.image_urls else None, kind=product.kind or "item")


def _card_name(product: Product) -> str:
    """A product's title, short enough for a card."""
    return " ".join(product.title.split()[:9])
