"""
The session state the agents share: the key names, and typed readers for them.

Keeping both in one place means no agent spells another agent's key by hand or
re-parses the stored data its own way. Keys starting with "user:" are ADK
user-scoped: they belong to the shopper and are shared across their sessions.
"""

from typing import Optional

from scout.domain import Capsule, PreferenceProfile, Product, ShoppingNeed, WardrobeItem
from scout.tools.cart import Cart
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
SUGGESTED_IDS = "suggested_ids"      # every product suggested this conversation: never suggest one twice
SUGGESTED_KINDS = "suggested_kinds"  # kinds already suggested: never offer the same kind twice
OFFER_OPEN = "offer_open"            # Scout offered the suggestion in its LAST reply, so a plain "yes" answers it
OFFER_AT_CHECKOUT = "offer_at_checkout"  # that offer came at checkout: declining it means "check me out"
CHECKOUT_REMINDED = "checkout_reminded"  # the one reminder before checkout was already given

# Planning outfits for any occasion (a trip, a week, a day at the office...)
MODE = "mode"                        # "outfit" while planning an outfit, until it is finished or they leave it
OUTFIT_MODE = "outfit"
STYLIST_PLAN = "stylist_plan"        # the Stylist agent's latest CapsulePlan
CAPSULE = "capsule"                  # the plan on the board (looks + the products chosen for new pieces), or None
OUTFIT_REQUESTED = "outfit_requested"  # asked for, waiting on one question (whose style): True or None
CAPSULE_EVENT = "capsule_event"      # what just happened to the plan, in plain words, for Narration to speak
CAPSULE_PLANNED = "A new plan was just made."   # the CAPSULE_EVENT for a fresh plan
PREFERENCE_EVIDENCE = "preference_evidence"  # what the Preference agent should learn from this turn

# What Scout remembers about the shopper. Loaded from the ShopperRepository (Cloud SQL)
# when a conversation starts and saved back after any turn that changes it.
SHOPPER_NAME = "shopper_name"        # what the shopper asked Scout to call them
CART = "cart"
PREFERENCES = "preferences"
WARDROBE = "wardrobe"                # clothes they already own (list of WardrobeItem dicts)
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


def cart(state) -> Cart:
    return Cart.model_validate(state.get(CART) or {})


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


def suggestion(state) -> Optional[Suggestion]:
    """The item Scout suggested to go with the cart, if one is open."""
    raw = state.get(SUGGESTION)
    return Suggestion.model_validate(raw) if raw else None


def capsule(state) -> Optional[Capsule]:
    """The outfit plan on the board, if any."""
    raw = state.get(CAPSULE)
    return Capsule.model_validate(raw) if raw else None


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
