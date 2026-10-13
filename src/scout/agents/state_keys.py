"""
The session state the agents share: the key names, and typed readers for them.

Keeping both in one place means no agent spells another agent's key by hand or
re-parses the stored data its own way. Keys starting with "user:" are ADK
user-scoped: they belong to the shopper and are shared across their sessions.
"""

from typing import Optional

from pydantic import ValidationError

from scout.domain import Outfit, PreferenceProfile, Product, ShoppingNeed, TasteProfile, WardrobeItem
from scout.tools.cart import Cart
from scout.tools.wishlist import Wishlist
from scout.tools.recommendations import Suggestion

# Current conversation
NEED = "need"                        # ShoppingNeed as a dict, built up over the conversation
TURN = "turn"                        # latest TurnUnderstanding from the Elicitation agent
QUESTIONS_ASKED = "questions_asked"  # clarifying questions asked for the current need
LAST_QUESTION = "last_question"      # the question Scout asked last, to read short answers
NEXT_UP = "next_up"                  # other categories they asked for together, to look at after the current one
CATALOG_OVERVIEW = "catalog_overview"  # text summary of what the catalog has close to the request

# Search results and what was said about them
SHORTLIST = "shortlist"              # products found by Retrieval (list of Product dicts)
EXACT_MATCHES = "exact_matches"      # ids in the shortlist that match everything asked for
RELAXED = "relaxed"                  # what Retrieval had to relax, e.g. ["details"]
OTHER_TYPES = "other_types"          # other kinds in stock for the same person, offered when few matched
KEPT = "kept"                        # ids among PRESENTED the shopper chose to keep while the others were replaced
PRESENTED = "presented"              # ids of the products spoken this time, in spoken order
SEEN = "seen"                        # ids of every product presented for the current need
FOCUS_PRODUCT = "focus_product"      # id of the product currently being discussed
COMPARE_IDS = "compare_ids"          # ids of the products being compared
COMPARISON = "comparison"            # latest ComparisonResult from the Comparison agent
NARRATION_TASK = "narration_task"    # what the Narration agent should speak next
CONVERSATION = "conversation"        # every exchange, word for word (capped): [{"shopper": ..., "scout": ...}]

# Cart and preferences
CART_EVENT = "cart_event"            # what just happened to the cart, for Narration to speak
SUGGESTION = "suggestion"            # one item that goes with the cart (Recommender), or None
SUGGESTIONS = "suggestions"          # the few options shown with it (the first is SUGGESTION), best first
FIT_NOTES = "fit_notes"              # product id -> why it suits the occasion they named
PENDING_SIZE = "pending_size"        # waiting for the size to add a product in: {product, suggested, reason}
SIZE_CHOICES = "size_choices"        # the sizes offered for it, to tap
SEARCHED_NEED = "searched_need"      # the need the shown options were found for
KNOWN_SIZES = "known_sizes"          # the shopper's sizes this conversation, by scale: {"shoe": "9", "top": "M"}
SUGGESTED_IDS = "suggested_ids"      # every product suggested this conversation: never suggest one twice
SUGGESTED_KINDS = "suggested_kinds"  # kinds already suggested: never offer the same kind twice
OFFER_OPEN = "offer_open"            # Scout offered the suggestion in its LAST reply, so a plain "yes" answers it
OFFER_AT_CHECKOUT = "offer_at_checkout"  # that offer came at checkout: declining it means "check me out"
CHECKOUT_REMINDED = "checkout_reminded"  # the one reminder before checkout was already given

# Planning outfits for any occasion (a trip, a week, a day at the office...)
MODE = "mode"                        # "outfit" while planning an outfit, until it is finished or they leave it
OUTFIT_MODE = "outfit"
STYLIST_PLAN = "stylist_plan"        # the Stylist agent's latest OutfitPlan
OUTFIT = "outfit_board"              # the Outfit on the board (every look with its own pieces), or None
OUTFIT_REQUESTED = "outfit_requested"  # a plan was asked for, waiting on one answer (the occasion or whose style)
OUTFIT_EVENT = "outfit_event"        # what just happened to the outfit, in plain words, for Narration to speak
OUTFIT_PLANNED = "A new plan was just made."   # the OUTFIT_EVENT for a fresh plan
PREFERENCE_EVIDENCE = "preference_evidence"  # what the Preference agent should learn from this turn
PREFERENCE_PENDING = "preference_pending"    # turns to learn from after they end: [{"evidence", "message"}]

# What Scout remembers about the shopper. Loaded from the ShopperRepository (Cloud SQL)
# when a conversation starts and saved back after any turn that changes it.
SHOPPER_NAME = "shopper_name"        # what the shopper asked Scout to call them
CART = "cart"
WISHLIST = "wishlist"                # products they saved to come back to, compare or buy later
WISHLIST_EVENT = "wishlist_event"    # what just happened to the wishlist, said by code
PREFERENCES = "preferences"
TASTE = "taste"                      # colours, materials, patterns, styles they lean towards (learned)
TASTE_EVIDENCE = "taste_evidence"    # items bought, saved or owned this turn, read in the background
WARDROBE = "wardrobe"                # clothes they already own (list of WardrobeItem dicts)
LOOK_COUNT = "look_count"            # how many looks they asked for in the outfit plan, or None (one per occasion)
SET_ASIDE = "set_aside"              # ids of their own pieces not to use in this plan ("I don't have it with me")
WARDROBE_JUST_ADDED = "wardrobe_just_added"  # ids of pieces added from photos since Scout last replied

# Kept by ADK across this shopper's sessions (in memory); moves to BigQuery with the analytics work
REQUESTED_CATEGORIES = "user:requested_categories"  # categories they asked for that the catalog lacks


# ---------------------------------------------------------------------
# Typed readers (any mapping with .get: session state, ADK context state)
# ---------------------------------------------------------------------

def need(state) -> ShoppingNeed:
    return ShoppingNeed.model_validate(state.get(NEED) or {})


def profile(state) -> PreferenceProfile:
    return PreferenceProfile.model_validate(state.get(PREFERENCES) or {})


def taste(state) -> TasteProfile:
    return TasteProfile.model_validate(state.get(TASTE) or {})


def taste_text(state) -> str:
    """Their taste for the style being shopped (any style when it is not known yet), without what they said
    they avoid: what they say always wins over what was learned."""
    avoid = {p.value.strip().lower() for p in profile(state).preferences if p.polarity == "avoid"}
    return taste(state).to_prompt_text(need(state).gender_filter(), avoid)


def cart(state) -> Cart:
    return Cart.model_validate(state.get(CART) or {})


def wishlist(state) -> Wishlist:
    return Wishlist.model_validate(state.get(WISHLIST) or {})


def shortlist(state) -> dict[str, Product]:
    """The products Retrieval found, by id."""
    return {item["id"]: Product.model_validate(item) for item in state.get(SHORTLIST) or [] if item.get("id")}


def presented(state) -> list[Product]:
    """The products Scout last spoke about, in the order it said them."""
    found = shortlist(state)
    return [found[i] for i in state.get(PRESENTED) or [] if i in found]


def focus(state) -> Optional[Product]:
    """The product currently being discussed."""
    return shortlist(state).get(state.get(FOCUS_PRODUCT))


def is_exact(state, product_id: str) -> bool:
    """Does this product match everything the shopper asked for? (Nothing relaxed means all of them do.)"""
    return not state.get(RELAXED) or product_id in set(state.get(EXACT_MATCHES) or [])


def suggestion(state, position: Optional[int] = None) -> Optional[Suggestion]:
    """The item Scout suggested to go with the cart, if one is open; `position` picks one of the options
    shown with it (1 = the first)."""
    raw = state.get(SUGGESTION)
    if not raw:
        return None
    if position:
        options = suggestions(state)
        if 1 <= position <= len(options):
            return options[position - 1]
    return Suggestion.model_validate(raw)


def suggestions(state) -> list[Suggestion]:
    """The options shown with the open suggestion (empty when none is open)."""
    if not state.get(SUGGESTION):
        return []
    return [Suggestion.model_validate(raw) for raw in state.get(SUGGESTIONS) or []] or \
        [Suggestion.model_validate(state.get(SUGGESTION))]


def outfit(state) -> Optional[Outfit]:
    """The outfit on the board, if any (one saved in an older shape is not shown)."""
    raw = state.get(OUTFIT)
    try:
        return Outfit.model_validate(raw) if raw else None
    except ValidationError:
        return None


def wardrobe(state) -> list[WardrobeItem]:
    """The clothes the shopper already owns."""
    return [WardrobeItem.model_validate(item) for item in state.get(WARDROBE) or []]


RECENT_TURNS_SHOWN = 6   # exchanges an agent sees word for word (the need, cart and profile hold the rest)
SCOUT_WORDS_SHOWN = 400  # characters of each reply: enough to know what was said, small enough to stay fast


def conversation(state) -> list[dict]:
    """Every exchange so far, oldest first (shown again when the shopper comes back)."""
    return list(state.get(CONVERSATION) or [])


def recent_turns_text(state) -> str:
    """The last few exchanges, oldest first, for an agent that needs the flow of the conversation."""
    lines = []
    for turn in conversation(state)[-RECENT_TURNS_SHOWN:]:
        reply = " ".join(turn.get("scout", "").split())[:SCOUT_WORDS_SHOWN]
        lines += [f"Shopper: {turn.get('shopper', '')}", f"Scout: {reply}"]
    return "\n".join(lines) or "This is the start of the conversation."


def planning_outfit(state) -> bool:
    """In outfit mode: every request is for the outfit until it is finished or they ask for regular shopping."""
    return state.get(MODE) == OUTFIT_MODE


def usable_wardrobe(state) -> list[WardrobeItem]:
    """Their own pieces the outfit plan may use: the wardrobe, without the ones they set aside for now."""
    aside = set(state.get(SET_ASIDE) or [])
    return [item for item in wardrobe(state) if item.id not in aside]
