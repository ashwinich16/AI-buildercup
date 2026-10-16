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

import logging
from typing import Literal, Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmResponse
from google.adk.planners import BuiltInPlanner
from google.genai import types
from pydantic import BaseModel, Field

from scout.agents import state_keys
from scout.domain.sizes import size_scale
from scout.agents.model_io import content_text, response_json
from scout.catalog import CatalogRepository, SearchQuery
from scout.config import Settings
from scout.domain import Product, ShoppingNeed
from scout.tools.retrieval import word_stems

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 4096   # thinking + the JSON answer
GENDER_STYLES = ("men", "women")   # the styles ShoppingNeed.gender can name

Intent = Literal[
    "shop",               # describing or refining what they want
    "ask_about_product",  # wants a product described, or has a question about one presented product
    "compare",            # wants presented products compared
    "more_options",       # wants other options for the same need
    "choose",             # has decided on a product (Scout adds it to the cart)
    "cart",               # wants to hear, change or check out their cart
    "wishlist",           # wants a product saved for later, or their saved products shown, compared or removed
    "see_suggestion",     # wants to see other items like the one Scout suggested to go with their cart
    "outfit",             # wants their outfits planned for a trip, a week or an event, or is steering the plan
    "pause",              # wants Scout to stop talking or wait, without ending the conversation
    "end_conversation",   # wants to finish the conversation for now
    "reject",             # does not like a presented product
    "other",              # anything else: greetings, thanks, help, off-topic
]


class OwnedPiece(BaseModel):
    category: str = Field(description="One of the catalog categories")
    description: str = Field(description="The piece in plain words: colour, kind, material, how formal")


class OutfitStep(BaseModel):
    action: Literal["plan", "change", "remove", "to_cart", "leave", "talk"] = Field(
        description="plan (or plan again) the looks; change a piece, or add a kind of piece, in one or more looks; "
                    "remove a piece from one or more looks; put the new pieces in the cart; leave planning for "
                    "regular shopping; talk about the plan (a question about it, or to hear it)")
    looks: list[int] = Field(
        default_factory=list,
        description="For change or remove: the numbers of the looks they mean (Look 1 is 1); empty if they did not say")
    all_looks: bool = Field(default=False, description="For change or remove: they mean every look")
    category: Optional[str] = Field(
        default=None, description="For change or remove: the piece's category as it is shown in brackets on the "
                                  "board; for a kind of piece to add, one of the catalog categories")
    wish: Optional[str] = Field(
        default=None, description="For change: what they want instead, in plain words a product description would "
                                  "use; null if they did not say")
    look_count: Optional[int] = Field(
        default=None, description="For plan: how many looks or outfits they asked for, if they said a number")
    set_aside: list[int] = Field(
        default_factory=list,
        description="For plan: numbers of the pieces in CLOTHES THEY ALREADY OWN they cannot or do not want to use")


class TurnUnderstanding(BaseModel):
    intent: Intent
    need: ShoppingNeed = Field(description="The full, updated need: previous need merged with this message")
    still_unclear: list[str] = Field(
        default_factory=list, max_length=5,
        description="Details that would still change which products fit; empty when the need is clear")
    ready_to_search: bool = Field(description="True when the need is clear enough to look for products")
    question: Optional[str] = Field(default=None, description="ONE short, friendly question, only if not ready")
    choices: list[str] = Field(
        default_factory=list, max_length=5,
        description="With a question: two to five very short answers the shopper can tap (a few words each)")
    understood: str = Field(description="One short sentence restating what the shopper wants now")
    style_answer: Optional[Literal["men", "women", "any"]] = Field(
        default=None, description="Only when your last question asked whose style it is: the style their reply "
                                  "names (any = it does not matter); otherwise null")
    next_up: list[str] = Field(
        default_factory=list, max_length=5,
        description="Other kinds of product they asked for in this message besides the one in need, as catalog "
                    "categories, to look at afterwards")
    unavailable_category: Optional[str] = Field(
        default=None, description="A kind of product the shopper asked for that is not in the catalog")
    product_reference: Optional[str] = Field(
        default=None, description="The shopper's own words for the product they mean, if any")
    product_position: Optional[int] = Field(
        default=None, description="Number (1, 2, 3...) of the product they mean in PRODUCTS YOU JUST PRESENTED; "
                                  "null if it is not clear which one")
    keep_positions: list[int] = Field(
        default_factory=list, max_length=10,
        description="For more_options: numbers of the presented options they want to keep; the others are "
                    "replaced. Empty when all of them should be replaced")
    product_positions: list[int] = Field(
        default_factory=list, max_length=10, description="For compare: numbers of the products to compare; empty means all presented")
    reason: Optional[str] = Field(default=None, description="For choose/reject: why, in the shopper's words")
    about_suggestion: bool = Field(
        default=False, description="True when the product they mean is the ITEM SCOUT SUGGESTED to go with "
                                   "their cart, not one of the presented options")
    suggestion_position: Optional[int] = Field(
        default=None, description="When about_suggestion: the number of the suggested item they mean (1 is the "
                                  "first); null if it is the first or not clear")
    outfit: Optional[OutfitStep] = Field(default=None, description="For outfit: what they want done with the outfit")
    wardrobe_remove: list[int] = Field(
        default_factory=list,
        description="Numbers of the pieces in CLOTHES THEY ALREADY OWN they want taken out of their wardrobe for "
                    "good (they no longer own it, or ask to delete it)")
    owned_items: list[OwnedPiece] = Field(
        default_factory=list, max_length=8,
        description="Clothes, shoes or accessories they say they already own and have not mentioned before")
    wishlist_action: Optional[Literal["save", "view", "compare", "remove"]] = Field(
        default=None, description="For wishlist: save a product for later, show the wishlist, compare products "
                                  "on it, or take products off it")
    wishlist_positions: list[int] = Field(
        default_factory=list, description="For wishlist compare or remove: numbers of the items in THEIR WISHLIST "
                                          "they mean; empty if they did not say which")
    cart_action: Optional[Literal["view", "remove", "clear", "checkout", "size"]] = Field(
        default=None, description="For cart: what they want to do with it; clear = take everything out; "
                                  "size = change the size of an item already in the cart")
    size: Optional[str] = Field(
        default=None, description="A size they give, in their words (\"medium\", \"9\", \"32 waist\"): when choosing a "
                                  "product, answering Scout's question about size, or for cart_action size")
    cart_positions: list[int] = Field(
        default_factory=list, max_length=10, description="For remove: numbers of the items in THEIR CART to take out; "
                                          "empty if it is not clear which")
    reply: Optional[str] = Field(
        default=None, description="For other: a short, friendly reply to say to the shopper")
    lasting_preference: bool = Field(
        default=False, description="True if the message reveals a lasting personal preference, or asks to forget one")

    def outfit_action(self) -> Optional[str]:
        """What to do with the outfit, when this turn is about it."""
        return self.outfit.action if self.intent == "outfit" and self.outfit else None


# Intents and behaviour are described by MEANING, never by sample phrases, so any
# wording or language works and the model does not copy our examples.
INSTRUCTION = """You are Scout, a warm and friendly shopping companion for blind and low-vision shoppers. Talk like a good friend who loves helping with shopping: relaxed, kind, encouraging, never formal or robotic. In this step you listen and understand; another step will find the products.

Understand what the shopper MEANS, whatever words or language they use. Never depend on particular phrases.

The shopper usually speaks, and their words reach you through speech recognition, which can mishear a word as a similar-sounding one. When a word does not make sense for shopping, read it as the closest-sounding word that fits the catalog categories and WHAT THE SHOP ACTUALLY HAS, and put that corrected meaning in the need, never the misheard words. If you are not confident in a correction and it would change what you search for, do not search yet: ask one short question that checks the corrected meaning.

CATALOG CATEGORIES: {categories}
PRODUCT TYPES IN THE SHOP: {product_types}

THE CONVERSATION SO FAR (the last few exchanges, oldest first)
{recent}

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

THEIR TASTE, LEARNED FROM WHAT THEY OWN AND SHOP (for the style being shopped)
{taste}
This is only a leaning: Scout uses it to put options they may like first. Never put it in the need, never ask about it, and never let it decide whose style the item is. If they ask what Scout knows about their taste, tell them from here and from their profile.

THEIR CART
{cart}

THEIR WISHLIST (products they saved for later)
{wishlist}

STILL TO LOOK FOR (other things they asked for earlier, after the current one)
{next_up}

ITEMS SCOUT SUGGESTED TO GO WITH THEIR CART (shown on screen; Scout named the first)
{suggestion}

SIZE SCOUT IS WAITING FOR
{size_wait}

CLOTHES THEY ALREADY OWN
{wardrobe}

PIECES THEY JUST SHOWED YOU IN PHOTOS
{just_added}

HOW THEY ARE SHOPPING
{mode}

OUTFIT PLAN ON THE BOARD
{outfit}

FOR THE SHOPPER'S NEW MESSAGE

1. intent: decide what the shopper is trying to achieve right now.
   - shop: they are describing, changing or narrowing down what they want, including a short reply that answers your last question.
   - more_options: they want to hear different or additional options for the same need, without changing the need. Do not ask a question; Scout will present the next options. If they also change what they want (a price limit, a colour, a detail), it is still more_options: update the need with the change. When they want to keep some of the presented options and replace only the others, it is still more_options: put the numbers of the ones to keep in keep_positions.
   - ask_about_product: they want one presented product described (including its look, picture or photo), or have a question about it. Scout CAN look at product photos, so this is never impossible.
   - compare: they want two or more presented products compared.
   - choose: they have decided on a product they want to buy.
   - cart: they want to hear what is in their cart, take one or more things out of it, empty it, or finish and pay.
   - wishlist: they want a product kept for later, or want to do something with the products they saved (THEIR WISHLIST). Set wishlist_action:
       save: keep a product for later. Set product_position to the presented product they mean (or about_suggestion and suggestion_position for a suggested item).
       view: show or hear their wishlist.
       compare: compare products on their wishlist. Put the numbers of the items in THEIR WISHLIST in wishlist_positions (empty if they did not say which).
       remove: take products off their wishlist: their numbers in THEIR WISHLIST in wishlist_positions.
     Once the wishlist is shown, its products are the presented ones, so asking about one, comparing them or choosing one works as with any presented products.
   - The ITEMS SCOUT SUGGESTED (above, if any) are products they can act on just like the presented ones. When the product they mean is one of those items, set about_suggestion true and suggestion_position to its number (by number, kind, colour or name), keep the need unchanged, ready_to_search false, no question, and pick the intent by what they want to do with it:
       choose: have it added to the cart. If Scout offered it in its last reply, a plain yes means choose.
       ask_about_product: a question about it, or to hear it described.
       see_suggestion: to see other items like it.
       reject: they do not want it. Write reply: a short, easy-going acknowledgement (no pressure, and invite what is next).
   - reject: they do not like a presented product.
   - outfit: they want what to wear planned for an occasion (a day at the office, a wedding, a trip, a week), or they are steering the outfit on the board (above). Asking for a whole outfit or look is outfit even for a single event, however they put it ("dress me for my first day at the office", "dress myself for an office day", "what should I wear to a wedding", "help me get ready for a party"); asking for one kind of product ("a formal shirt") is shop. Fill outfit.action:
       plan: plan the looks, or plan them again with a change (another occasion, more looks, a different feel). Put the occasion in their words in need.purpose, whose style in need.gender, a budget for all the new pieces in need.budget_max, any wish about the looks in need.looking_for, and leave need.category null. If they say how many looks or outfits they want, put that number in outfit.look_count. The occasion and whose style are all Scout needs: never ask about anything else for a plan.
         When they say they cannot or do not want to use some of their own pieces, it is plan, with the numbers of those pieces from CLOTHES THEY ALREADY OWN in outfit.set_aside.
       change: they want a different piece in one or more looks (a different product, other options, cheaper, another colour), or a kind of piece added to them. Put its category in outfit.category (as shown in brackets on the board; for a kind the looks do not have yet, a catalog category), the look numbers in outfit.looks (or all_looks), and what they want instead in outfit.wish. If the board says a piece is being chosen now and they do not name another one, it is that piece.
       remove: they want a piece taken out of one or more looks: outfit.category and outfit.looks (or all_looks).
       to_cart: they want the new pieces put in their cart.
       leave: they want to stop planning and go back to regular shopping.
       talk: they ask about the plan or want to hear it.
     While they are planning an outfit (HOW THEY ARE SHOPPING), everything they ask for is for the outfit until they finish it or ask for regular shopping. Never reply with a generic line while they are planning.
     While the board says a piece is being chosen now, naming one of the presented options in any way (its number, colour, a detail) is choose: it goes into the outfit.
   - If PIECES THEY JUST SHOWED YOU lists anything, start your question or reply by briefly saying what you see (name up to four pieces, then how many more), so they know you looked.
   - wardrobe_remove: whatever the intent, when they want pieces taken out of their wardrobe for good (delete it, remove it from my wardrobe, I don't own it any more, I gave it away), put their numbers from CLOTHES THEY ALREADY OWN here; if nothing else is asked, intent is other. Not having a piece with them, or not wanting it for this outfit, is set_aside instead (it stays in the wardrobe). When it is unclear which they mean, use set_aside while planning an outfit: it can be undone.
   - owned_items: whatever the intent, when they say they already own clothes, shoes or accessories not listed in CLOTHES THEY ALREADY OWN, list each one they describe well enough to style with (at least its colour or what it looks like), with its catalog category and a short description in their words. When they name a piece without saying what it looks like, do not list it yet: ask one short question about its colour and style (the piece is not added until they answer). Owning something is never a reason to search for it.
   - pause: they want Scout to stop talking, be quiet or wait for a moment, without leaving.
   - end_conversation: they want to finish talking for now or say goodbye.
   - other: anything else.

2. need: return the COMPLETE updated need. Keep everything already known; add or change only what this message says. Never invent details. category must be one of the catalog categories, or null. When the shopper names a kind of product, use the categories in WHAT THE SHOP ACTUALLY HAS as a strong hint for which catalog category it belongs to.
   product_types: the specific kind(s) they want, spelled exactly as in PRODUCT TYPES IN THE SHOP. Fill it only when their words clearly mean one of those kinds. A general word that people use for several of the shop's kinds leaves product_types empty. When they accept another kind you offered, add it.
   purpose: the occasion or use, in their words. When it points to particular kinds the shop has for that person (see "Kinds of ..." in WHAT THE SHOP ACTUALLY HAS), also put those kinds in product_types, so the search looks for the right kind.
   gender: whose STYLE the item is, and only from what the shopper says. When they correct themselves ("for a woman... no, actually men's"), in one message or across messages, the LAST thing they say wins. Naming the style, or naming the person it is for, settles it; saying it does not matter means "any". Never infer it from their name, voice, way of talking, profile, or from who is talking: an item they want for themselves does NOT settle it, because anyone can shop for any style. When they move on to a different kind of product, keep the gender only if they make clear it is for the same person. When your last question asked whose style it is, a reply that names a style answers it: set gender from it, even when the reply is only a few words.
   for_someone_else: true when they say this item is for someone else (a gift, a partner, a child); keep it for the same item, and set it back to false when they move on to something for themselves.
   budget_min / budget_max: the price range they give, as a lower and an upper limit (a single limit sets only that one; a range sets both). A new limit of one kind replaces the old one of that kind.
   must_have: every specific quality they ask the product to have, in plain words a product description would use. When they ask for a different value of a quality they already gave, it replaces the old one.

3. Decide whether the need is clear. Think about it like a friend would:
   - Imagine searching right now. Would one more answer clearly change WHICH products you would pick? If yes, it is still unclear. If it would only fine-tune the choice, it is clear enough.
   - List those still-unclear details in still_unclear (most important first).
   - Some shoppers are clear after one message; others need a few questions. Decide case by case.
   - Budget is helpful but never required.
   - Use what you know about this shopper: never ask what their profile already answers, and add their relevant stated or confident preferences to the need (taste to must_have or avoid, their size or fit, their usual budget) unless this message says otherwise. One exception: who the item is for. People often shop for someone else, so even when their profile says whose style they usually buy, check it for each new item, offering that style as the likely answer.
   - Who it is for is unclear when gender is null and "Made for" in WHAT THE SHOP ACTUALLY HAS includes both men's and women's items. It then comes FIRST in still_unclear, because it changes every product you would pick. When the matches are all one style or unisex, it is not unclear.
   - What it is for (the occasion or use) is unclear when need.purpose is empty and the use would change which products you pick, unless the shopper said it does not matter or named a kind that already implies the use. It decides the kind, the style and what matters most, so ask it right after who it is for, before showing anything. Ask about the occasion rather than listing kinds: the occasion usually settles the kind.
   - The kind is unclear only when product_types is empty and, even with the occasion known, the shopper's words could still mean very different kinds the shop has for that person (see "Kinds of ..." in WHAT THE SHOP ACTUALLY HAS). Then ask it after the occasion.
   - Usually that is all: who it is for, what it is for, and the kind if still open. Do not ask about details the shopper has not brought up unless the options really differ by them.
   - The kind is ALWAYS unclear when need.category is empty and product_types is empty ("something to wear", "an outfit"): there is nothing to search for yet. Ask which kind, offering the two or three kinds the shop has that suit their occasion best (from "Kinds of ..." in WHAT THE SHOP ACTUALLY HAS), or offer to plan a whole outfit.
   ready_to_search is true when still_unclear is empty, or when the shopper signals they want to see products now rather than answer more questions.

4. If not ready, ask ONE question about the first item in still_unclear. Think like a friend who knows this exact product:
   - First think: for this specific kind of product, what are the real differences between the options the shop has? Ask about the difference that would split them most for this shopper.
   - Never ask about something already known or already implied by what they said. Never ask two things at once.
   - Choices you offer must be realistic for that exact product AND exist in WHAT THE SHOP ACTUALLY HAS. Never offer a choice the shop cannot satisfy. Use the shopper's own words where you can.
   - Under 20 words, warm and natural; you may react briefly first.
   - When asking who it is for, ask about the STYLE of the item (men's or women's, using the product's name), never about the shopper's own gender, and in a way that makes shopping for someone else feel normal. If their profile suggests a style, offer it as the likely answer while leaving the other open.
   - When asking about the kind, offer only kinds listed under "Kinds of ..." for that person, described in everyday words.
   - Offer two or three short possible answers so it is easy to reply by voice, and put the same answers in choices so they can also tap one.
   - Write a fresh question for this shopper and this product; never a generic or stock question.
   - Respect their patience. If you have already asked 3 or more questions, offer to show options now instead of asking more.

   - Several kinds of product at once: put the first in need and the others in next_up, and when you ask your question, say briefly that you will start with the first and find the others next.
   - Moving on to something in STILL TO LOOK FOR: it was asked for together with the earlier item, so it is for the same person: keep the gender.

5. Category not in the catalog: set unavailable_category, keep category null, and in your question say kindly that you do not have it yet, that you have noted their request, and name the categories you can help with.

6. understood: one short, friendly sentence restating what they want now.

7. For ask_about_product, compare, more_options or choose: keep the need unchanged (for more_options, apart from what this message changes), ready_to_search false, no question; fill product_reference with the shopper's own words, and reason for choose.
   product_position: the number of the product they mean in PRODUCTS YOU JUST PRESENTED, however they describe it (by position, a feature, price, colour, or by pointing back to the one being discussed). If it truly could be more than one, or none, set it to null; Scout will ask.
   product_positions (compare only): the numbers of the products they want compared, however they describe them. Leave empty when they mean all of the presented options.

8. For reject: fill product_reference, product_position and reason. If the reason names a quality to avoid, add it to need.avoid as one or two plain words a product description would use. Set ready_to_search true so Scout can look for better options.

9. For cart or wishlist: keep the need unchanged, ready_to_search false, no question; set cart_action (or wishlist_action). To change the size of something in the cart, set cart_action size, cart_positions to that item and size.
   Sizes: when SIZE SCOUT IS WAITING FOR names a product and they give a size, set intent choose and size (keep the need unchanged). If they ask about sizes or for help choosing one, set intent other and reply briefly using the sizes listed there; it is still waiting. When they choose a product and say its size at once, set size too.
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
            model=self._settings.llm("elicitation"),
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
        need = state_keys.need(state)
        return INSTRUCTION.format(
            categories=", ".join(self._catalog.categories()) or "unknown",
            product_types=", ".join(self._catalog.product_types()) or "unknown",
            need=need.model_dump_json(exclude_defaults=True) if need != ShoppingNeed() else "nothing yet",
            last_question=state.get(state_keys.LAST_QUESTION) or "none",
            questions_asked=state.get(state_keys.QUESTIONS_ASKED, 0),
            catalog_overview=state.get(state_keys.CATALOG_OVERVIEW) or "not available",
            next_up=", ".join(state.get(state_keys.NEXT_UP) or []) or "nothing",
            presented=_presented_products(state),
            profile=state_keys.profile(state).to_prompt_text(),
            taste=state_keys.taste_text(state),
            cart=state_keys.cart(state).to_prompt_text(),
            wishlist=state_keys.wishlist(state).to_prompt_text(),
            suggestion=_suggestion_text(state),
            size_wait=_size_wait_text(state),
            wardrobe=_wardrobe_text(state),
            just_added=_just_added_text(state),
            outfit=_outfit_text(state),
            mode=("Planning an outfit (outfit mode)" if state_keys.planning_outfit(state)
                  else "Regular shopping"),
            recent=state_keys.recent_turns_text(state),
        )

    # ADK calls this after the model answers; the agent keeps its own state up to date.
    def _remember(self, callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
        if llm_response.partial:
            return None
        state = callback_context.state
        turn = response_json(llm_response, TurnUnderstanding)
        if turn is None:   # broken or cut-off answer: keep what we knew and ask again, never crash the turn
            turn = TurnUnderstanding(
                intent="other",
                need=state_keys.need(state),
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

        self._settle_suggestion(state, turn)
        self._carry_outfit_request(state, turn)
        turn.need.product_types = self._known_types(turn.need.product_types)
        if turn.need.gender is None and turn.style_answer:   # the answer to "whose style?", asked on its own
            turn.need.gender = turn.style_answer
        turn.need.gender = self._kept_gender(state, turn.need)
        state[state_keys.NEXT_UP] = self._next_up(state, turn)
        message = content_text(callback_context.user_content, separator=" ")
        turn.need.avoid = self._grounded_avoid(state, turn, message)
        if self._must_ask_kind(turn, asked) and not state.get(state_keys.OUTFIT_REQUESTED) \
                and not state_keys.planning_outfit(state):   # an outfit is planned by the Stylist, not by kind
            logger.info("Need has no kind of product; asking which before searching.")
            turn.ready_to_search = False
            turn.question = turn.question or _kind_question(turn.need, self._catalog.categories())
        if self._must_ask_gender(turn, asked):
            logger.info("Need has no gender for %r; asking before searching.", turn.need.category)
            turn.ready_to_search = False
            turn.question = _gender_question(turn.need)
            turn.choices = ["Men's style", "Women's style"]
        planning = turn.outfit_action() == "plan"
        state[state_keys.OUTFIT_REQUESTED] = True if planning and not turn.ready_to_search else None
        self._forget_heard_if_need_changed(state, turn.need)
        state[state_keys.NEED] = turn.need.model_dump()
        if turn.question and not turn.ready_to_search:
            state[state_keys.QUESTIONS_ASKED] = asked + 1
            state[state_keys.LAST_QUESTION] = turn.question
        if not turn.question or turn.ready_to_search:
            turn.choices = []                       # answers to tap belong to a question
        if turn.ready_to_search:
            state[state_keys.QUESTIONS_ASKED] = 0
            state[state_keys.LAST_QUESTION] = None

        llm_response.content = types.Content(role="model", parts=[types.Part(text=turn.model_dump_json())])
        return llm_response

    @staticmethod
    def _kept_gender(state, need: ShoppingNeed) -> Optional[str]:
        """The model sometimes drops a style the shopper already gave; keep it while the category stays the same,
        or when they move on to something they asked for together with it (STILL TO LOOK FOR).

        Any other new category may be for someone else, so there the model's answer stands (it asks again).
        """
        previous = state_keys.need(state)
        asked_together = need.category in (state.get(state_keys.NEXT_UP) or [])
        if need.gender is None and previous.gender and (previous.category == need.category or asked_together):
            return previous.gender
        return need.gender

    def _next_up(self, state, turn: TurnUnderstanding) -> list[str]:
        """What they still want to look for after the current item: kept until they move on to it."""
        known = {c.lower(): c for c in self._catalog.categories()}
        wanted = [*(state.get(state_keys.NEXT_UP) or []), *turn.next_up]
        current = (turn.need.category or "").lower()
        return list(dict.fromkeys(known[c.lower()] for c in wanted if c.lower() in known and c.lower() != current))

    @staticmethod
    def _forget_heard_if_need_changed(state, need: ShoppingNeed) -> None:
        """Options were heard against the need as it was. Once it changes (a quality is no longer avoided, a new
        price limit...), the ones that fit the new need count as new again, so "more" or "others" can show them."""
        wording = {"looking_for"}   # every other field changes which products fit
        if need.model_dump(exclude=wording) != state_keys.need(state).model_dump(exclude=wording):
            state[state_keys.SEEN] = []

    @staticmethod
    def _carry_outfit_request(state, turn: TurnUnderstanding) -> None:
        """A plan needs the occasion (asked here) and whose style (asked by _must_ask_gender); the Stylist plans
        the rest, so nothing else is asked. An answer to one of those questions ("women's") reads like ordinary
        shopping, so a plan waiting on it carries on into the reply."""
        if state.get(state_keys.OUTFIT_REQUESTED) and turn.intent in ("shop", "other"):
            turn.intent = "outfit"
            turn.outfit = OutfitStep(action="plan")
        if turn.intent == "outfit" and turn.outfit is None:
            turn.outfit = OutfitStep(action="talk" if state_keys.outfit(state) else "plan")
        if turn.outfit_action() == "plan":
            turn.need.category, turn.need.product_types = None, []   # a plan starts from the occasion
            if turn.need.purpose:
                turn.ready_to_search, turn.question, turn.choices = True, None, []
            else:
                turn.ready_to_search = False
                turn.question = turn.question or "Lovely! What's the occasion?"

    @staticmethod
    def _settle_suggestion(state, turn: TurnUnderstanding) -> None:
        """The suggested item can only be meant while one is open, and acting on it never starts a search."""
        if state_keys.suggestion(state) is None:
            turn.about_suggestion = False
        if turn.about_suggestion:
            turn.ready_to_search = False
            turn.question = None
            if turn.intent in ("shop", "more_options", "other"):   # about the item, but not choose/ask/reject:
                turn.intent = "see_suggestion"                     # show it with a few like it

    def _grounded_avoid(self, state, turn: TurnUnderstanding, message: str) -> list[str]:
        """Things to avoid hide products, so each one must come from the shopper: their words now (or their
        reason for rejecting), their saved profile, or an earlier turn. The model sometimes fills `avoid`
        with features of products it just showed ("denim" for jeans!), which would hide good matches."""
        previous = {t.lower() for t in state_keys.need(state).avoid}
        profile = {p.value.lower() for p in state_keys.profile(state).preferences if p.polarity == "avoid"}
        said = word_stems(f"{message} {turn.reason or ''}")
        kept = [term for term in self._qualities_only(turn.need)
                if term.lower() in previous | profile or word_stems(term) & said]
        return list(dict.fromkeys(kept))

    def _qualities_only(self, need: ShoppingNeed) -> list[str]:
        """`avoid` is for qualities. Moving from one person's style or one kind of product to another is not
        a quality to avoid: it would hide products for the wrong reason, so entries that name a catalog
        category, kind or style (men / women, the values gender takes) as whole words are dropped."""
        names = [*self._catalog.categories(), *self._catalog.product_types(), *GENDER_STYLES]
        return [term for term in need.avoid if not any(_whole_words(name) <= _whole_words(term) for name in names)]

    def _must_ask_kind(self, turn: TurnUnderstanding, asked: int) -> bool:
        """Never search with no kind of product at all ("something to wear for a wedding"): the search finds
        nothing and Scout can only apologise. The prompt asks for this; this is the safety net."""
        return (turn.intent in ("shop", "reject") and turn.ready_to_search and not turn.need.category
                and not turn.need.product_types and asked < self._settings.elicitation_safety_max_questions)

    def _must_ask_gender(self, turn: TurnUnderstanding, asked: int) -> bool:
        """Never search a category the shop sells in both men's and women's styles without knowing which.

        The prompt asks for this, but the model sometimes skips it; a mixed list is the worst outcome.
        """
        need = turn.need
        if turn.outfit_action() == "plan" and turn.ready_to_search:
            return need.gender is None and asked < self._settings.elicitation_safety_max_questions   # never mixed
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



def _whole_words(text: str) -> set[str]:
    """Lower-case words split on spaces only ("tie-dye" stays one word), without a possessive or plural "s"."""
    words = set()
    for word in text.lower().replace("’", "'").split():
        word = word.removesuffix("'s")
        words.add(word[:-1] if word.endswith("s") and len(word) > 3 else word)
    return words


def read_turn(state) -> Optional[TurnUnderstanding]:
    """The latest TurnUnderstanding stored in session state (ADK may store it as a JSON string)."""
    raw = state.get(state_keys.TURN)
    try:
        if isinstance(raw, str):
            return TurnUnderstanding.model_validate_json(raw)
        return TurnUnderstanding.model_validate(raw) if raw else None
    except ValueError as error:
        logger.warning("Could not read the turn understanding: %s", error)
        return None


def _suggestion_text(state) -> str:
    options = state_keys.suggestions(state)
    if not options:
        return "none"
    offered = "\nScout offered the first one in its last reply." if state.get(state_keys.OFFER_OPEN) else ""
    return "\n".join(f"{n}. {' '.join(s.product.title.split()[:10])} (a {s.product.kind}, goes with their {s.goes_with})"
                     for n, s in enumerate(options, start=1)) + offered


def _wardrobe_text(state) -> str:
    """Numbered, so the shopper can set pieces aside; ones already set aside say so."""
    aside = set(state.get(state_keys.SET_ASIDE) or [])
    return "\n".join(f"{n}. {item.description} ({item.category})" + (" [set aside for now]" if item.id in aside else "")
                     for n, item in enumerate(state_keys.wardrobe(state), start=1)) or "none yet"


def _just_added_text(state) -> str:
    added = set(state.get(state_keys.WARDROBE_JUST_ADDED) or [])
    return "\n".join(f"- {item.description} ({item.category})"
                     for item in state_keys.wardrobe(state) if item.id in added) or "none"


def _outfit_text(state) -> str:
    outfit = state_keys.outfit(state)
    if outfit is None and state.get(state_keys.OUTFIT_REQUESTED):
        return ("Not started yet: the shopper asked for a plan and Scout asked one question first. "
                "Their reply continues that request (intent outfit, plan) unless they clearly want something else.")
    return outfit.to_prompt_text() if outfit else "none"


def _gender_question(need: ShoppingNeed) -> str:
    """Asked only when the model skipped this question; it names the category as the catalog spells it."""
    kind = " or ".join(need.product_types) or need.category or "outfit"
    return f"Happy to help! Is this for a men's or a women's style of {kind}?"


def _size_wait_text(state) -> str:
    pending = state.get(state_keys.PENDING_SIZE)
    if not pending:
        return "nothing"
    if pending.get("outfit"):
        return (f"Scout asked which size they want for the {pending.get('kinds', 'outfit')}, to add the outfit to "
                f"the cart. It comes in {pending['scale']['spoken']}.")
    product = Product.model_validate(pending["product"])
    scale = size_scale(product)
    return (f"Scout asked which size of the {product.kind or 'item'} ({' '.join(product.title.split()[:8])}) "
            f"they want, to add it to the cart. It comes in {scale.spoken if scale else 'one size'}.")


def _kind_question(need: ShoppingNeed, categories: list[str]) -> str:
    """Asked only when the model wanted to search with no kind of product at all."""
    occasion = f" for {need.purpose.removeprefix('for ')}" if need.purpose else ""
    offered = [c.lower() for c in categories[:3]]
    choices = ", ".join(offered[:-1]) + f" or {offered[-1]}" if len(offered) > 1 else "".join(offered)
    return f"Lovely! What would you like to find{occasion}: {choices}? Or I can plan a whole outfit with you."


def _presented_products(state) -> str:
    """Short numbered list of what Scout last presented, so the model can tell which one is meant."""
    focus = state_keys.focus(state)
    lines = []
    for number, product in enumerate(state_keys.presented(state), start=1):
        facts = [a.value for a in product.attributes if len(a.value) <= 25][:2]   # short, distinguishing facts
        if product.price is not None:
            facts.append(f"${product.price:.0f}")
        title = " ".join(product.title.split()[:10])
        marker = "  <- being discussed" if focus and product.id == focus.id else ""
        lines.append(f"{number}. {title} ({', '.join(facts)}){marker}" if facts else f"{number}. {title}{marker}")
    return "\n".join(lines) or "none yet"
