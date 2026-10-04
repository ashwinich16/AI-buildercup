"""
Elicitation agent: Scout's "ears".

On every turn it reads the shopper's message and returns a TurnUnderstanding:
  - what the shopper wants to do now (intent)
  - the updated ShoppingNeed, merged with what was already known
  - what is still unclear, and whether it is clear enough to search
  - if not, ONE friendly question

The agent itself decides when the need is clear enough; there is no fixed
number of questions. Code only adds a safety cap so a confused model can never
keep a blind shopper stuck in questions.

It does not search, compare or narrate, and it knows nothing about field
names: the categories it offers come from the catalog at runtime.

Efficiency: the agent does not replay the whole chat history. It sees only the
current message plus a compact state (need so far, last question, questions
asked), which keeps every call small.
"""

import json
import logging
from typing import Literal, Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmResponse
from google.adk.planners import BuiltInPlanner
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from scout.agents import state_keys
from scout.catalog import CatalogRepository, SearchQuery
from scout.config import Settings
from scout.domain import COLOURS, PreferenceProfile, Product, ShoppingNeed, colours_wanted, gender_in_words
from scout.tools.cart import Cart
from scout.tools.retrieval import word_stems

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 4096   # thinking + the JSON answer

Intent = Literal[
    "shop",               # describing or refining what they want
    "ask_about_product",  # wants a product described, or has a question about one presented product
    "compare",            # wants presented products compared
    "more_options",       # wants other options for the same need
    "choose",             # has decided on a product (Scout adds it to the cart)
    "cart",               # wants to hear, change or check out their cart
    "pause",              # wants Scout to stop talking or wait, without ending the conversation
    "end_conversation",   # wants to finish the conversation for now
    "reject",             # does not like a presented product
    "other",              # anything else: greetings, thanks, help, off-topic
]


class TurnUnderstanding(BaseModel):
    intent: Intent
    need: ShoppingNeed = Field(description="The full, updated need: previous need merged with this message")
    still_unclear: list[str] = Field(
        default_factory=list, max_length=5,
        description="Details that would still change which products fit; empty when the need is clear")
    ready_to_search: bool = Field(description="True when the need is clear enough to look for products")
    question: Optional[str] = Field(default=None, description="ONE short, friendly question, only if not ready")
    understood: str = Field(description="One short sentence restating what the shopper wants now")
    unavailable_category: Optional[str] = Field(
        default=None, description="A kind of product the shopper asked for that is not in the catalog")
    product_reference: Optional[str] = Field(
        default=None, description="The shopper's own words for the product they mean, if any")
    product_position: Optional[int] = Field(
        default=None, description="Number (1, 2, 3...) of the product they mean in PRODUCTS YOU JUST PRESENTED; "
                                  "null if it is not clear which one")
    product_positions: list[int] = Field(
        default_factory=list, max_length=10, description="For compare: numbers of the products to compare; empty means all presented")
    reason: Optional[str] = Field(default=None, description="For choose/reject: why, in the shopper's words")
    cart_action: Optional[Literal["view", "remove", "clear", "checkout"]] = Field(
        default=None, description="For cart: what they want to do with it; clear = take everything out")
    cart_positions: list[int] = Field(
        default_factory=list, max_length=10, description="For remove: numbers of the items in THEIR CART to take out; "
                                          "empty if it is not clear which")
    reply: Optional[str] = Field(
        default=None, description="For other: a short, friendly reply to say to the shopper")
    lasting_preference: bool = Field(
        default=False, description="True if the message reveals a lasting personal preference, or asks to forget one")


# Intents and behaviour are described by MEANING, never by sample phrases, so any
# wording or language works and the model does not copy our examples.
INSTRUCTION = """You are Scout, a warm and friendly shopping companion for blind and low-vision shoppers. Talk like a good friend who loves helping with shopping: relaxed, kind, encouraging, never formal or robotic. In this step you listen and understand; another step will find the products.

Understand what the shopper MEANS, whatever words or language they use. Never depend on particular phrases.

The shopper usually speaks, and their words reach you through speech recognition, which can mishear a word as a similar-sounding one. When a word does not make sense for shopping, read it as the closest-sounding word that fits the catalog categories and WHAT THE SHOP ACTUALLY HAS, and put that corrected meaning in the need, never the misheard words. If you are not confident in a correction and it would change what you search for, do not search yet: ask one short question that checks the corrected meaning.

CATALOG CATEGORIES: {categories}
PRODUCT TYPES IN THE SHOP: {product_types}

WHAT YOU KNOW SO FAR
Need: {need}
Your last question to the shopper: {last_question}
Questions you have asked about this need: {questions_asked}

WHAT THE SHOP ACTUALLY HAS (closest matches to what they are asking for)
{catalog_overview}

PRODUCTS YOU JUST PRESENTED (in the order you said them)
{presented}

WHAT YOU KNOW ABOUT THIS SHOPPER (from earlier conversations)
{profile}

THEIR CART
{cart}

FOR THE SHOPPER'S NEW MESSAGE

1. intent: decide what the shopper is trying to achieve right now.
   - shop: they are describing, changing or narrowing down what they want, including a short reply that answers your last question.
   - more_options: they want to hear different or additional options for the same need, without changing the need. Do not ask a question; Scout will present the next options.
   - ask_about_product: they want one presented product described (including its look, picture or photo), or have a question about it. Scout CAN look at product photos, so this is never impossible.
   - compare: they want two or more presented products compared.
   - choose: they have decided on a product they want to buy.
   - cart: they want to hear what is in their cart, take one or more things out of it, empty it, or finish and pay.
   - reject: they do not like a presented product.
   - pause: they want Scout to stop talking, be quiet or wait for a moment, without leaving.
   - end_conversation: they want to finish talking for now or say goodbye.
   - other: anything else.

2. need: return the COMPLETE updated need. Keep everything already known; add or change only what this message says. Never invent details. category must be one of the catalog categories, or null. When the shopper names a kind of product, use the categories in WHAT THE SHOP ACTUALLY HAS as a strong hint for which catalog category it belongs to.
   product_types: the specific kind(s) they want, spelled exactly as in PRODUCT TYPES IN THE SHOP. Fill it only when their words clearly mean one kind: a button-up, collared, dress, flannel or Hawaiian shirt is "shirt"; a tee is "t-shirt". A word that people use for several kinds (the plain word "shirt", which many people also use for tees and other tops, or "a top", or "shoes") leaves product_types empty. When they accept another kind you offered, add it.
   gender: whose STYLE the item is, and only from what the shopper says. Naming the style, or naming the person it is for (a husband, a daughter, a brother), settles it; saying it does not matter means "any". Never infer it from their name, voice, way of talking, profile, or from who is talking: an item "for me" or "for myself" does NOT settle it, because anyone can shop for any style. When they move on to a different kind of product, keep the gender only if they make clear it is for the same person.

3. Decide whether the need is clear. Think about it like a friend would:
   - Imagine searching right now. Would one more answer clearly change WHICH products you would pick? If yes, it is still unclear. If it would only fine-tune the choice, it is clear enough.
   - List those still-unclear details in still_unclear (most important first).
   - Some shoppers are clear after one message; others need a few questions. Decide case by case.
   - Budget is helpful but never required.
   - Use what you know about this shopper: never ask what their profile already answers, and add their relevant stated or confident preferences to the need (taste to must_have or avoid, their size or fit, their usual budget) unless this message says otherwise. One exception: who the item is for. People often shop for someone else, so even when their profile says whose style they usually buy, check it for each new item, offering that style as the likely answer.
   - Who it is for is unclear when gender is null and "Made for" in WHAT THE SHOP ACTUALLY HAS includes both men's and women's items. It then comes FIRST in still_unclear, because it changes every product you would pick. When the matches are all one style or unisex, it is not unclear.
   - The kind is unclear when product_types is empty and the shopper's words could mean several kinds the shop has for that person (see "Kinds of ..." in WHAT THE SHOP ACTUALLY HAS). It comes right after who it is for.
   ready_to_search is true when still_unclear is empty, or when the shopper signals they want to see products now rather than answer more questions.

4. If not ready, ask ONE question about the first item in still_unclear. Think like a friend who knows this exact product:
   - First think: for this specific kind of product, what are the real differences between the options the shop has? Ask about the difference that would split them most for this shopper.
   - Never ask about something already known or already implied by what they said. Never ask two things at once.
   - Choices you offer must be realistic for that exact product AND exist in WHAT THE SHOP ACTUALLY HAS. Never offer a choice the shop cannot satisfy. Use the shopper's own words where you can.
   - Under 20 words, warm and natural; you may react briefly first.
   - When asking who it is for, ask about the STYLE of the item (men's or women's, using the product's name), never about the shopper's own gender, and in a way that makes shopping for someone else feel normal. If their profile suggests a style, offer it as the likely answer while leaving the other open.
   - When asking about the kind, offer only kinds listed under "Kinds of ..." for that person, described in everyday words.
   - Offer two or three short possible answers so it is easy to reply by voice.
   - Write a fresh question for this shopper and this product; never a generic or stock question.
   - Respect their patience. If you have already asked 3 or more questions, offer to show options now instead of asking more.

5. Category not in the catalog: set unavailable_category, keep category null, and in your question say kindly that you do not have it yet, that you have noted their request, and name the categories you can help with.

6. understood: one short, friendly sentence restating what they want now.

7. For ask_about_product, compare, more_options or choose: keep the need unchanged, ready_to_search false, no question; fill product_reference with the shopper's own words, and reason for choose.
   product_position: the number of the product they mean in PRODUCTS YOU JUST PRESENTED, however they describe it (by position, a feature, price, colour, or by pointing back to the one being discussed). If it truly could be more than one, or none, set it to null; Scout will ask.
   product_positions (compare only): the numbers of the products they want compared, however they describe them. Leave empty when they mean all of the presented options.

8. For reject: fill product_reference, product_position and reason. If the reason names a quality to avoid, add it to need.avoid as one or two plain words a product description would use. Set ready_to_search true so Scout can look for better options.

9. For cart: keep the need unchanged, ready_to_search false, no question; set cart_action.
   remove: set cart_positions to the numbers of the items in THEIR CART they mean, however they describe them (by number, name, colour, kind of product, price, or the one just discussed). If the cart has only one item, that is the one they mean. If it could be more than one item and they did not say which, leave cart_positions empty; Scout will ask.
   clear: they want everything taken out of the cart.

10. lasting_preference: true when the message reveals something lasting about the shopper (beyond this one purchase): their taste, size or fit, usual budget, how they like Scout to talk with them, or who they usually shop for; or asks Scout to forget or stop assuming something about them.

11. For other: write reply, one or two short, friendly sentences that respond to what they said and gently invite them to shop. If they told you something lasting about themselves, acknowledge that you will remember it. If they asked something Scout cannot do, say so kindly and say what you can help with.

12. For pause: keep the need unchanged, ready_to_search false, no question, no reply. Scout will simply go quiet and keep listening.
    For end_conversation: keep the need unchanged, ready_to_search false, no question; write reply as a short, warm goodbye that mentions their cart only if it has items.
"""


class ElicitationAgentBuilder:
    """Builds the Elicitation LlmAgent. Dependencies are passed in, never created here."""

    def __init__(self, settings: Settings, catalog: CatalogRepository):
        self._settings = settings
        self._catalog = catalog
        self._genders_by_category: dict[str, set[str]] = {}   # e.g. "jeans" -> {"men", "women"}

    def build(self) -> LlmAgent:
        tuning = self._settings.tuning("elicitation")
        return LlmAgent(
            name="elicitation",
            model=self._settings.model_for("elicitation"),
            description="Understands each shopper message, updates the shopping need, "
                        "and asks a friendly clarifying question until the need is clear.",
            instruction=self._instruction,
            include_contents="none",            # compact state instead of full history
            output_schema=TurnUnderstanding,
            output_key=state_keys.TURN,
            # The cap cuts off a model stuck repeating itself in seconds; a normal answer is well under it.
            generate_content_config=types.GenerateContentConfig(temperature=tuning.temperature,
                                                                max_output_tokens=MAX_OUTPUT_TOKENS),
            planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(
                thinking_budget=tuning.thinking_budget)),
            after_model_callback=self._remember,
        )

    # ADK calls this before every model call to build the system instruction.
    def _instruction(self, context: ReadonlyContext) -> str:
        state = context.state
        need = ShoppingNeed.model_validate(state.get(state_keys.NEED) or {})
        return INSTRUCTION.format(
            categories=", ".join(self._catalog.categories()) or "unknown",
            product_types=", ".join(self._catalog.product_types()) or "unknown",
            need=need.model_dump_json(exclude_defaults=True) if need != ShoppingNeed() else "nothing yet",
            last_question=state.get(state_keys.LAST_QUESTION) or "none",
            questions_asked=state.get(state_keys.QUESTIONS_ASKED, 0),
            catalog_overview=state.get(state_keys.CATALOG_OVERVIEW) or "not available",
            presented=_presented_products(state),
            profile=PreferenceProfile.model_validate(state.get(state_keys.PREFERENCES) or {}).to_prompt_text(),
            cart=Cart.model_validate(state.get(state_keys.CART) or {}).to_prompt_text(),
        )

    # ADK calls this after the model answers; the agent keeps its own state up to date.
    def _remember(self, callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
        if llm_response.partial:
            return None
        state = callback_context.state
        turn = self._parse(llm_response)
        if turn is None:   # broken or cut-off answer: keep what we knew and ask again, never crash the turn
            turn = TurnUnderstanding(
                intent="other",
                need=ShoppingNeed.model_validate(state.get(state_keys.NEED) or {}),
                ready_to_search=False,
                understood="",
                reply="Sorry, I lost track for a second there. Could you say that once more?",
            )

        asked = state.get(state_keys.QUESTIONS_ASKED, 0)

        # Safety net only: the model decides, but a stuck model must not trap the shopper.
        if turn.intent == "shop" and not turn.ready_to_search \
                and asked >= self._settings.elicitation_safety_max_questions:
            logger.info("Safety cap reached after %d questions; searching with what we know.", asked)
            turn.ready_to_search = True
            turn.question = None

        if turn.unavailable_category:
            requested = list(state.get(state_keys.REQUESTED_CATEGORIES) or [])
            requested.append(turn.unavailable_category)
            state[state_keys.REQUESTED_CATEGORIES] = requested   # honest: the request really is noted

        turn.need.product_types = self._known_types(turn.need.product_types)
        turn.need.gender = self._kept_gender(state, turn.need)
        message = _text_of(callback_context.user_content)
        turn.need.avoid = self._grounded_avoid(state, turn, message)
        if turn.intent in ("shop", "reject"):
            turn.need.must_have = _with_colours(turn.need.must_have, colours_wanted(message))
        said = gender_in_words(message)
        if said and turn.intent in ("shop", "reject"):
            turn.need.gender = said              # their own words win over the model's reading
        if self._must_ask_gender(turn, asked):
            logger.info("Need has no gender for %r; asking before searching.", turn.need.category)
            turn.ready_to_search = False
            turn.question = _gender_question(turn.need)
        state[state_keys.NEED] = turn.need.model_dump()
        if turn.question and not turn.ready_to_search:
            state[state_keys.QUESTIONS_ASKED] = asked + 1
            state[state_keys.LAST_QUESTION] = turn.question
        elif turn.ready_to_search:
            state[state_keys.QUESTIONS_ASKED] = 0
            state[state_keys.LAST_QUESTION] = None

        llm_response.content = types.Content(role="model", parts=[types.Part(text=turn.model_dump_json())])
        return llm_response

    @staticmethod
    def _kept_gender(state, need: ShoppingNeed) -> Optional[str]:
        """The model sometimes drops a style the shopper already gave; keep it while the category stays the same.

        A new category may be for someone else, so there the model's answer stands (it asks again).
        """
        previous = ShoppingNeed.model_validate(state.get(state_keys.NEED) or {})
        if need.gender is None and previous.gender and previous.category == need.category:
            return previous.gender
        return need.gender

    def _grounded_avoid(self, state, turn: TurnUnderstanding, message: str) -> list[str]:
        """Things to avoid hide products, so each one must come from the shopper: their words now (or their
        reason for rejecting), their saved profile, or an earlier turn. The model sometimes fills `avoid`
        with features of products it just showed ("denim" for jeans!), which would hide good matches."""
        previous = {t.lower() for t in ShoppingNeed.model_validate(state.get(state_keys.NEED) or {}).avoid}
        profile = {p.value.lower() for p in PreferenceProfile.model_validate(state.get(state_keys.PREFERENCES) or {})
                   .preferences if p.polarity == "avoid"}
        said = word_stems(f"{message} {turn.reason or ''}")
        kept = [term for term in self._qualities_only(turn.need)
                if term.lower() in previous | profile or word_stems(term) & said]
        return list(dict.fromkeys(kept))

    def _qualities_only(self, need: ShoppingNeed) -> list[str]:
        """`avoid` is for qualities ("leather", "tight fit"). Switching from women's to men's jeans is not
        "avoid women's jeans": that would hide products for the wrong reason, so such entries are dropped."""
        kinds = {k.lower() for k in [*self._catalog.categories(), *self._catalog.product_types()]}
        return [term for term in need.avoid
                if not gender_in_words(term) and not ({w.strip("'s") for w in term.lower().split()} & kinds)]

    def _must_ask_gender(self, turn: TurnUnderstanding, asked: int) -> bool:
        """Never search a category the shop sells in both men's and women's styles without knowing which.

        The prompt asks for this, but the model sometimes skips it; a mixed list is the worst outcome.
        """
        need = turn.need
        if turn.intent not in ("shop", "reject") or not turn.ready_to_search or need.gender or not need.category:
            return False
        if asked >= self._settings.elicitation_safety_max_questions:
            return False
        return {"men", "women"} <= self._genders(need.category)

    def _genders(self, category: str) -> set[str]:
        if category not in self._genders_by_category:   # the catalog rarely changes; look once per category
            try:
                products = self._catalog.search(SearchQuery(text="", category=category, limit=50))
                self._genders_by_category[category] = {p.gender for p in products if p.gender}
            except Exception as error:   # a failed lookup must never block the conversation
                logger.warning("Could not read genders for %s: %s", category, error)
                return set()
        return self._genders_by_category[category]

    def _known_types(self, product_types: list[str]) -> list[str]:
        """Keep only kinds the catalog really has, in its spelling: a made-up kind would filter out everything."""
        known = {t.lower(): t for t in self._catalog.product_types()}
        if not known:
            return product_types
        return list(dict.fromkeys(known[t.strip().lower()] for t in product_types if t.strip().lower() in known))

    @staticmethod
    def _parse(llm_response: LlmResponse) -> Optional[TurnUnderstanding]:
        if llm_response.partial or not llm_response.content or not llm_response.content.parts:
            return None
        text = "".join(part.text or "" for part in llm_response.content.parts if not part.thought)
        try:
            return TurnUnderstanding.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as error:
            logger.warning("Elicitation output could not be parsed: %s", error)
            return None


def _text_of(content) -> str:
    if not content or not content.parts:
        return ""
    return " ".join(part.text for part in content.parts if part.text)


def _with_colours(must_have: list[str], colours: list[str]) -> list[str]:
    """A colour the shopper names is always a must-have (the model sometimes leaves it out); a newly
    named colour replaces the old one ("blue ones instead"), other must-haves stay."""
    if not colours:
        return list(dict.fromkeys(must_have))
    others = [term for term in must_have if term.lower() not in COLOURS]
    return list(dict.fromkeys(others + colours))[:8]


def _gender_question(need: ShoppingNeed) -> str:
    kind = " or ".join(need.product_types) or need.category or "items"
    if not kind.endswith("s"):
        kind = kind[:-1] + "ies" if kind.endswith("y") else kind + "s"
    return f"Happy to help! Should I look at men's or women's {kind}?"


def _presented_products(state) -> str:
    """Short numbered list of what Scout last presented, so the model can tell which one is meant."""
    by_id = {item.get("id"): item for item in state.get(state_keys.SHORTLIST) or []}
    focus_id = state.get(state_keys.FOCUS_PRODUCT)
    lines = []
    for number, product_id in enumerate(state.get(state_keys.PRESENTED) or [], start=1):
        if product_id not in by_id:
            continue
        product = Product.model_validate(by_id[product_id])
        facts = [a.value for a in product.attributes if len(a.value) <= 25][:2]   # short, distinguishing facts
        if product.price is not None:
            facts.append(f"${product.price:.0f}")
        title = " ".join(product.title.split()[:10])
        marker = "  <- being discussed" if product_id == focus_id else ""
        lines.append(f"{number}. {title} ({', '.join(facts)}){marker}" if facts else f"{number}. {title}{marker}")
    return "\n".join(lines) or "none yet"
