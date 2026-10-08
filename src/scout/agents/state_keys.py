"""
The session state the agents share: the key names, and typed readers for them.

Keeping both in one place means no agent spells another agent's key by hand or
re-parses the stored data its own way. Keys starting with "user:" are ADK
user-scoped: they belong to the shopper and are shared across their sessions.
"""

from typing import Optional

from scout.domain import Outfit, PreferenceProfile, Product, ShoppingNeed
from scout.tools.cart import Cart
from scout.tools.recommendations import Suggestion

# Current conversation
NEED = "need"                        # ShoppingNeed as a dict, built up over the conversation
TURN = "turn"                        # latest TurnUnderstanding from the Elicitation agent
QUESTIONS_ASKED = "questions_asked"  # clarifying questions asked for the current need
LAST_QUESTION = "last_question"      # the question Scout asked last, to read short answers
CATALOG_OVERVIEW = "catalog_overview"  # text summary of what the catalog has close to the request

# Search results and what was said about them
SHORTLIST = "shortlist"              # products found by Retrieval (list of Product dicts)
EXACT_MATCHES = "exact_matches"      # ids in the shortlist that match everything asked for
RELAXED = "relaxed"                  # what Retrieval had to relax, e.g. ["details"]
OTHER_TYPES = "other_types"          # other kinds in stock for the same person, offered when few matched
PRESENTED = "presented"              # ids of the products spoken this time, in spoken order
SEEN = "seen"                        # ids of every product presented for the current need
FOCUS_PRODUCT = "focus_product"      # id of the product currently being discussed
COMPARE_IDS = "compare_ids"          # ids of the products being compared
COMPARISON = "comparison"            # latest ComparisonResult from the Comparison agent
NARRATION_TASK = "narration_task"    # what the Narration agent should speak next
RECENT_TURNS = "recent_turns"        # the last few exchanges, word for word: [{"shopper": ..., "scout": ...}]

# Cart and preferences
CART_EVENT = "cart_event"            # what just happened to the cart, for Narration to speak
SUGGESTION = "suggestion"            # one item that goes with the cart (Recommender), or None
SUGGESTED_IDS = "suggested_ids"      # every product suggested this conversation: never suggest one twice
SUGGESTED_KINDS = "suggested_kinds"  # kinds already suggested: never offer the same kind twice
OFFER_OPEN = "offer_open"            # Scout offered the suggestion in its LAST reply, so a plain "yes" answers it
OFFER_AT_CHECKOUT = "offer_at_checkout"  # that offer came at checkout: declining it means "check me out"
CHECKOUT_REMINDED = "checkout_reminded"  # the one reminder before checkout was already given

# Styling an outfit for an occasion
STYLIST_PLAN = "stylist_plan"        # the Stylist agent's latest OutfitPlan
OUTFIT = "outfit"                    # the Outfit being put together (plan + pieces picked), or None
OUTFIT_REQUESTED = "outfit_requested"  # asked for, waiting on one question (whose style): {"pieces": [...]}
PREFERENCE_EVIDENCE = "preference_evidence"  # what the Preference agent should learn from this turn

# What Scout remembers about the shopper. Loaded from the ShopperRepository (Cloud SQL)
# when a conversation starts and saved back after any turn that changes it.
SHOPPER_NAME = "shopper_name"        # what the shopper asked Scout to call them
CART = "cart"
PREFERENCES = "preferences"

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


def outfit(state) -> Optional[Outfit]:
    """The outfit being put together, if any."""
    raw = state.get(OUTFIT)
    return Outfit.model_validate(raw) if raw else None


def recent_turns_text(state) -> str:
    """The last few exchanges, oldest first, for an agent that needs the flow of the conversation."""
    lines = []
    for turn in state.get(RECENT_TURNS) or []:
        lines += [f"Shopper: {turn.get('shopper', '')}", f"Scout: {turn.get('scout', '')}"]
    return "\n".join(lines) or "This is the start of the conversation."
