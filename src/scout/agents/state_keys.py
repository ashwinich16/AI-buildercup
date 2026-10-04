"""
Names of the session-state keys the agents share.

Keeping them in one place means no agent spells another agent's key by hand.
Keys starting with "user:" are ADK user-scoped: they belong to the shopper and
are shared across all their sessions (persisted when the session service is
backed by a database such as Cloud SQL).
"""

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

# Cart and preferences
CART_EVENT = "cart_event"            # what just happened to the cart, for Narration to speak
PREFERENCE_EVIDENCE = "preference_evidence"  # what the Preference agent should learn from this turn

# What Scout remembers about the shopper. Loaded from the ShopperRepository (Cloud SQL)
# when a conversation starts and saved back after any turn that changes it.
SHOPPER_NAME = "shopper_name"        # what the shopper asked Scout to call them
CART = "cart"
PREFERENCES = "preferences"

# Kept by ADK across this shopper's sessions (in memory); moves to BigQuery with the analytics work
REQUESTED_CATEGORIES = "user:requested_categories"  # categories they asked for that the catalog lacks
