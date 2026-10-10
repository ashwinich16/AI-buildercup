"""
ScoutChatService: runs the Scout agent for one shopper and turns session
state into what the UI shows (spoken reply, product cards, cart).

Two kinds of memory:
  - the ADK session (in memory): only the current conversation, such as what is
    being searched and which options were just presented
  - the ShopperRepository (Cloud SQL): what Scout remembers about the shopper
    (name, preferences, cart, the clothes they own). It is loaded into the session when a conversation
    starts, and saved after any turn that changed it.
"""

import asyncio
import logging
import re
from typing import AsyncIterator, Optional

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
from scout.api.shopper_repository import ShopperMemory, ShopperRepository
from scout.domain import Capsule, Preference, PreferenceProfile, Product, WardrobeItem
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
    description: str
    image_url: Optional[str] = None
    owned: bool = False
    why: str = ""                                          # for a piece better left at home: the reason
    pending: bool = False                                  # guided: a new piece not chosen yet (an empty slot)
    current: bool = False                                  # guided: the piece being chosen now


class LookView(BaseModel):
    title: str                                             # e.g. "Day 1 · client meeting"
    why: str = ""
    tiles: list[BoardTile] = Field(default_factory=list)
    product_ids: list[str] = Field(default_factory=list)   # the new pieces in it, to try the look on


class NewPieceView(BaseModel):
    key: str
    category: str
    why: str = ""
    looks: int = 0                                         # how many looks it is worn in
    product: Optional[SmallProductCard] = None             # None: not chosen yet, skipped, or none in the shop
    current: bool = False                                  # guided: being chosen now
    skipped: bool = False


class CapsuleView(BaseModel):
    """The outfit board: one look per day or occasion, and the few new pieces to buy."""
    occasion: str
    looks: list[LookView] = Field(default_factory=list)
    packed: list[BoardTile] = Field(default_factory=list)          # their own pieces that work for the trip
    leave_at_home: list[BoardTile] = Field(default_factory=list)   # their own pieces that don't, and why
    new_pieces: list[NewPieceView] = Field(default_factory=list)
    total: float = 0.0
    budget: Optional[float] = None
    guided: bool = False                                   # they choose each new piece, one at a time


class ScreenState(BaseModel):
    products: list[ProductCard] = Field(default_factory=list)
    focus_product_id: Optional[str] = None
    cart: CartView = Field(default_factory=CartView)
    suggestion: Optional[SmallProductCard] = None
    capsule: Optional[CapsuleView] = None
    wardrobe: list[WardrobeItem] = Field(default_factory=list)


class ChatTurn(ScreenState):
    reply: str
    action: Optional[str] = None      # "pause" (go quiet, keep listening) or "end" (stop listening)
    choices: list[str] = Field(default_factory=list)   # short answers to Scout's question, to tap


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
            # The wardrobe (with its thumbnails) is sent once, at the end of the turn, not with every update.
            yield {"type": "screen", **_screen_from_state(state).model_dump(mode="json", exclude={"wardrobe"})}

        session = await self._get(user_id, session_id)
        await self._remember(user_id, before, _memory_from_state(session.state))
        turn = read_turn(session.state)
        action = _ACTIONS.get(turn.intent) if turn else None
        reply = "\n\n".join(replies)
        if not reply and action != "pause":
            reply = "Sorry, I lost my train of thought. Could you say that again?"
            yield {"type": "delta", "text": reply}
        choices = turn.choices if turn and turn.question and not turn.ready_to_search else []
        done = ChatTurn(reply=reply, action=action, choices=choices, **_screen_from_state(session.state).model_dump())
        yield {"type": "done", **done.model_dump(mode="json")}

    async def ensure_session(self, user_id: str, session_id: str) -> None:
        """Raises LookupError if this shopper has no such conversation."""
        await self._get(user_id, session_id)

    async def products(self, user_id: str, session_id: str, product_ids: list[str]) -> list[Product]:
        """Products this conversation has shown (options, the suggestion, the board's new pieces), in the
        order asked. Raises LookupError if this shopper has no such conversation."""
        state = (await self._get(user_id, session_id)).state
        known = dict(state_keys.shortlist(state))
        suggestion, capsule = state_keys.suggestion(state), state_keys.capsule(state)
        if suggestion:
            known[suggestion.product.id] = suggestion.product
        if capsule:
            known.update({product.id: product for product in capsule.products_to_buy()})
        return [known[i] for i in product_ids if i in known]

    async def add_to_wardrobe(self, user_id: str, session_id: str, added: list[WardrobeItem]) -> list[WardrobeItem]:
        """Remember pieces the shopper owns (from a photo). Scout mentions them in its next reply.
        Returns the whole wardrobe."""
        return await self._change_wardrobe(user_id, session_id, lambda items: [*items, *added],
                                           just_added=[item.id for item in added])

    async def remove_from_wardrobe(self, user_id: str, session_id: str, item_id: str) -> list[WardrobeItem]:
        return await self._change_wardrobe(user_id, session_id,
                                           lambda items: [i for i in items if i.id != item_id])

    async def _change_wardrobe(self, user_id: str, session_id: str, change,
                               just_added: Optional[list[str]] = None) -> list[WardrobeItem]:
        """Change the wardrobe in this conversation (so Scout knows at once) and in the shopper's memory."""
        session = await self._get(user_id, session_id)
        before = _memory_from_state(session.state)
        wardrobe = change(state_keys.wardrobe(session.state))
        kept = {item.id for item in wardrobe}
        shown = [i for i in [*(session.state.get(state_keys.WARDROBE_JUST_ADDED) or []), *(just_added or [])]
                 if i in kept]
        delta = {state_keys.WARDROBE: [item.model_dump() for item in wardrobe],
                 state_keys.WARDROBE_JUST_ADDED: shown}
        await self._sessions.append_event(session, Event(author="user", actions=EventActions(state_delta=delta)))
        await self._remember(user_id, before, before.model_copy(update={"wardrobe": wardrobe}))
        return wardrobe

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
        state_keys.WARDROBE: [item.model_dump() for item in memory.wardrobe],
    }


def _memory_from_state(state: dict) -> ShopperMemory:
    return ShopperMemory(
        known=True,
        preferred_name=state.get(state_keys.SHOPPER_NAME) or "",
        preferences=state_keys.profile(state),
        cart=state_keys.cart(state),
        wardrobe=state_keys.wardrobe(state),
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
_TWO_WAYS = ("Shall we plan an outfit together, using the clothes you already own, "
             "or are you shopping for something in particular?")


# Formatting a model may still add (bold/italic asterisks, headings, code marks, bullet dashes).
# Replies are spoken aloud, so these would be read out as symbols.
_FORMATTING = re.compile(r"[*#`_>]+|^[ \t]*[-•][ \t]+", re.MULTILINE)


def _spoken_cart(cart: Cart, named: int = 2) -> str:
    """The cart in a few words: the first items by a short name, then how many more."""
    names = [" ".join(item.name.split()[:6]) for item in cart.items[:named]]
    others = len(cart.items) - len(names)
    if others:
        names.append(f"{others} more {'item' if others == 1 else 'items'}")
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _screen_from_state(state: dict) -> ScreenState:
    need = state_keys.need(state)
    cards = []
    for position, product in enumerate(state_keys.presented(state), start=1):
        cards.append(ProductCard(
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
        ))
    cart = state_keys.cart(state)
    return ScreenState(products=cards, focus_product_id=state.get(state_keys.FOCUS_PRODUCT),
                       cart=CartView(items=cart.items, total=cart.total),
                       suggestion=_suggestion_card(state), capsule=_capsule_view(state.get(state_keys.CAPSULE)),
                       wardrobe=state_keys.wardrobe(state))


def _suggestion_card(state: dict) -> Optional[SmallProductCard]:
    suggestion = state_keys.suggestion(state)
    if suggestion is None:
        return None
    return _small_card(suggestion.product).model_copy(update={"goes_with": suggestion.goes_with,
                                                             "reasons": suggestion.reasons})


def _capsule_view(raw: Optional[dict]) -> Optional[CapsuleView]:
    if not raw:
        return None
    capsule = Capsule.model_validate(raw)
    owned = {item.id: item for item in capsule.wardrobe}
    looks = []
    for look in capsule.plan.looks:
        tiles = []
        for piece in look.pieces:
            if piece.owned in owned:
                item = owned[piece.owned]
                tiles.append(BoardTile(description=item.description, image_url=item.thumbnail, owned=True))
            elif piece.new in capsule.picks:
                card = _small_card(capsule.picks[piece.new])
                tiles.append(BoardTile(description=card.name, image_url=card.image_url))
            elif capsule.guided and (new := capsule.new_piece(piece.new or "")) and new.key not in capsule.skipped:
                tiles.append(BoardTile(description=new.category, pending=True, current=new.key == capsule.current))
        looks.append(LookView(title=look.title, why=look.why, tiles=tiles,
                              product_ids=[p.id for p in capsule.look_products(look)]))
    used = dict.fromkeys(piece.owned for look in capsule.plan.looks for piece in look.pieces if piece.owned in owned)
    packed = [BoardTile(description=owned[i].description, image_url=owned[i].thumbnail, owned=True) for i in used]
    left = [BoardTile(description=owned[x.owned].description, image_url=owned[x.owned].thumbnail, owned=True,
                      why=x.why) for x in capsule.plan.leave_at_home if x.owned in owned]
    worn = [piece.new for look in capsule.plan.looks for piece in look.pieces if piece.new]
    new_pieces = [NewPieceView(key=p.key, category=p.category, why=p.why, looks=worn.count(p.key),
                               product=_small_card(capsule.picks[p.key]) if p.key in capsule.picks else None,
                               current=p.key == capsule.current, skipped=p.key in capsule.skipped)
                  for p in capsule.plan.new_pieces]
    total = sum(p.price or 0 for p in capsule.products_to_buy())
    return CapsuleView(occasion=capsule.plan.occasion, looks=looks, packed=packed, leave_at_home=left,
                       new_pieces=new_pieces,
                       total=round(total, 2), budget=capsule.budget, guided=capsule.guided)


def _small_card(product: Product) -> SmallProductCard:
    return SmallProductCard(id=product.id, name=_card_name(product), price=product.price,
                          image_url=product.image_urls[0] if product.image_urls else None, kind=product.kind or "item")


def _card_name(product: Product) -> str:
    """A product's title, short enough for a card."""
    return " ".join(product.title.split()[:9])
