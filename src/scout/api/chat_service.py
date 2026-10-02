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
import json
import logging
import re
from typing import Optional

from google.adk.agents import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from scout.agents import state_keys
from scout.api.shopper_repository import ShopperMemory, ShopperRepository
from scout.domain import Preference, PreferenceProfile, Product
from scout.tools import Cart, CartItem

logger = logging.getLogger(__name__)

APP_NAME = "scout"


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


class CartView(BaseModel):
    items: list[CartItem] = Field(default_factory=list)
    total: float = 0.0


class ScreenState(BaseModel):
    products: list[ProductCard] = Field(default_factory=list)
    focus_product_id: Optional[str] = None
    cart: CartView = Field(default_factory=CartView)


class ChatTurn(ScreenState):
    reply: str
    action: Optional[str] = None      # "pause" (go quiet, keep listening) or "end" (stop listening)


class StartedSession(ScreenState):
    session_id: str
    greeting: str
    shopper_name: str = ""            # what Scout calls the shopper


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

    async def send(self, user_id: str, session_id: str, text: str) -> ChatTurn:
        before = _memory_from_state((await self._get(user_id, session_id)).state)
        message = types.Content(role="user", parts=[types.Part(text=text)])
        replies = []
        async for event in self._runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
            spoken = _text_of(event)
            if spoken:
                replies.append(spoken)
        session = await self._get(user_id, session_id)
        await self._remember(user_id, before, _memory_from_state(session.state))
        action = _ACTIONS.get(_intent(session.state))
        reply = "\n\n".join(replies)
        if not reply and action != "pause":
            reply = "Sorry, I lost my train of thought. Could you say that again?"
        return ChatTurn(reply=reply, action=action, **_screen_from_state(session.state).model_dump())

    async def screen(self, user_id: str, session_id: str) -> ScreenState:
        return _screen_from_state((await self._get(user_id, session_id)).state)

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
        preferences=PreferenceProfile.model_validate(state.get(state_keys.PREFERENCES) or {}),
        cart=Cart.model_validate(state.get(state_keys.CART) or {}),
    )


_ACTIONS = {"pause": "pause", "end_conversation": "end"}


def _intent(state: dict) -> Optional[str]:
    turn = state.get(state_keys.TURN)
    if isinstance(turn, str):
        try:
            turn = json.loads(turn)
        except json.JSONDecodeError:
            return None
    return turn.get("intent") if isinstance(turn, dict) else None


def _greeting(name: str, returning: bool, cart: Cart) -> str:
    """Scout opens the conversation, and explains once how talking works."""
    hello = f", {name}" if name else ""
    if returning:
        waiting = f" You still have {_spoken_cart(cart)} in your cart." if cart.items else ""
        return f"Welcome back{hello}!{waiting} What are we shopping for today?"
    return (f"Hi{' ' + name if name else ''}! I'm Scout, your shopping companion. Just talk to me naturally, and interrupt me "
            "whenever you like. When you hear a soft chime, it's your turn to speak. "
            "What are you shopping for today?")


# Formatting a model may still add (bold/italic asterisks, headings, code marks, bullet dashes).
# Replies are spoken aloud, so these would be read out as symbols.
_FORMATTING = re.compile(r"[*#`]+|^[ \t]*[-•][ \t]+", re.MULTILINE)


def _spoken_cart(cart: Cart, named: int = 2) -> str:
    """The cart in a few words: the first items by a short name, then how many more."""
    names = [" ".join(item.name.split()[:6]) for item in cart.items[:named]]
    others = len(cart.items) - len(names)
    if others:
        names.append(f"{others} more {'item' if others == 1 else 'items'}")
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _text_of(event) -> str:
    """The spoken text of an agent event, as plain text."""
    if getattr(event, "partial", False) or not event.content or not event.content.parts:
        return ""
    text = "".join(part.text or "" for part in event.content.parts if not getattr(part, "thought", False))
    return _FORMATTING.sub("", text).strip()


def _screen_from_state(state: dict) -> ScreenState:
    by_id = {item.get("id"): item for item in state.get(state_keys.SHORTLIST) or []}
    exact = set(state.get(state_keys.EXACT_MATCHES) or [])
    all_exact = not state.get(state_keys.RELAXED)        # nothing was relaxed: every option matches
    cards = []
    for position, product_id in enumerate(state.get(state_keys.PRESENTED) or [], start=1):
        if product_id not in by_id:
            continue
        product = Product.model_validate(by_id[product_id])
        cards.append(ProductCard(
            id=product.id,
            position=position,
            name=" ".join(product.title.split()[:9]),
            price=product.price,
            rating=product.rating,
            rating_count=product.rating_count,
            image_url=product.image_urls[0] if product.image_urls else None,
            summary=product.short_description,
            exact_match=all_exact or product.id in exact,
        ))
    cart = Cart.model_validate(state.get(state_keys.CART) or {})
    return ScreenState(products=cards, focus_product_id=state.get(state_keys.FOCUS_PRODUCT),
                       cart=CartView(items=cart.items, total=cart.total))
