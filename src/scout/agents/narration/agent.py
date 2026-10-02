"""
Narration agent: Scout's "voice".

It turns what the other steps found into short, warm, spoken language that a
blind or low-vision shopper can picture. It does not search, decide or
remember anything; it only speaks, and only from the facts it is given.

What to narrate is a NarrationTask the Orchestrator puts in session state.
Each task has its own prompt, so new tasks are added without changing
existing ones.
"""

import json
from enum import Enum
from typing import Callable

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.planners import BuiltInPlanner
from google.genai import types

from scout.agents import state_keys
from scout.agents.voice import VOICE
from scout.config import Settings
from scout.domain import PreferenceProfile, Product, ShoppingNeed
from scout.tools.cart import Cart


class NarrationTask(str, Enum):
    PRESENT_OPTIONS = "present_options"   # speak the first options after a search
    MORE_OPTIONS = "more_options"         # the shopper asked for more; speak the next ones
    NO_MORE_OPTIONS = "no_more_options"   # the shopper asked for more, but there are none
    SPEAK_COMPARISON = "speak_comparison"  # speak the Comparison agent's analysis
    CART_UPDATE = "cart_update"           # confirm what happened to the cart


PRESENT_OPTIONS = """{voice}
WHAT THE SHOPPER WANTS
{need}

WHAT YOU KNOW ABOUT THIS SHOPPER (follow how they like you to talk; mention a taste preference only when it explains why an option suits them)
{profile}

{search_note}

OPTIONS TO PRESENT (best match first)
{products}

YOUR TASK
Present these options in about 80 to 120 words:
1. Start with a short, natural reaction.
2. Be honest about matches. Options marked [exact match] have everything they asked for. Options marked [related] do not: say so plainly and kindly, name what they lack, and say what they DO offer for this shopper's need.
3. Lead with the best option and WHY it suits what they want, plus one detail that helps them picture it.
4. Then each other option in one sentence, with the one way it differs.
5. Mention a review point only if it matters for their need.
6. Make the order clear as you speak (first, second...) so they can refer back.
7. End with ONE easy question offering a useful next step (such as hearing one described, comparing, or hearing more).
If there are no options, say so kindly and suggest one thing they could change.
"""

MORE_OPTIONS = """{voice}
WHAT THE SHOPPER WANTS
{need}

They asked to hear MORE options. They have already heard {already_heard} other option(s).

{search_note}

NEXT OPTIONS TO PRESENT
{products}

YOUR TASK
Make clear these are additional options, then present them: honest about [exact match] vs [related], one sentence each with what makes it different, making their order clear so they can refer back. About 60 to 100 words. End with ONE easy question.
"""

NO_MORE_OPTIONS = """{voice}
WHAT THE SHOPPER WANTS
{need}

They asked for more options, but they have already heard all {already_heard} options the shop has for this.

YOUR TASK
In about 30 words, say kindly and honestly that those were all the options you have for this, and offer ONE helpful change to their need that would open up new options. Do not invent products.
"""


SPEAK_COMPARISON = """{voice}
WHAT THE SHOPPER WANTS
{need}

They asked you to compare some options. Your comparison expert has analysed them:

{comparison}

YOUR TASK
Speak this comparison in about 90 to 130 words:
- Start with the difference that matters most for their decision.
- Then each product in one or two sentences: its spoken name, what it does best for them, and its main drawback.
- If there is a recommendation, give it with the reason. If it depends on something, say what, and ask them about it.
- Respect the sources: photo-based facts are how it looks; review points are other people's experience; interpretations are your opinion.
- End with ONE easy question offering a useful next step.
"""


CART_UPDATE = """{voice}
WHAT JUST HAPPENED
{event}

THEIR CART NOW
{cart}

YOUR TASK
In one to three short sentences, tell them clearly what happened and where things stand:
- Added: confirm what you added, using a short spoken name, and the cart total.
- Already there: say so kindly.
- Removed: confirm what was removed (one item or several, using short spoken names) and the new total, or that the cart is now empty.
- Already empty: say kindly there was nothing in the cart.
- Not clear which item: ask which one, naming the items by number.
- Viewing: list the items with prices and the total. If empty, say so warmly.
- Checkout: read back the items and total, then say honestly that payment is not connected in this prototype yet, and that their cart is saved.
Then offer ONE useful next step. Never invent items or prices.
"""


class NarrationAgentBuilder:
    """Builds the Narration LlmAgent. One prompt builder per NarrationTask."""

    def __init__(self, settings: Settings, options_to_present: int = 3):
        self._settings = settings
        self._options_to_present = options_to_present
        self._prompts: dict[NarrationTask, Callable[[dict], str]] = {
            NarrationTask.PRESENT_OPTIONS: self._present_options_prompt,
            NarrationTask.MORE_OPTIONS: self._more_options_prompt,
            NarrationTask.NO_MORE_OPTIONS: self._no_more_options_prompt,
            NarrationTask.SPEAK_COMPARISON: self._speak_comparison_prompt,
            NarrationTask.CART_UPDATE: self._cart_update_prompt,
        }

    def build(self) -> LlmAgent:
        tuning = self._settings.tuning("narration")
        return LlmAgent(
            name="narration",
            model=self._settings.model_for("narration"),
            description="Speaks results to the shopper in a warm, vivid, grounded way.",
            instruction=self._instruction,
            include_contents="none",            # it speaks from state, not from chat history
            generate_content_config=types.GenerateContentConfig(temperature=tuning.temperature),
            planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(
                thinking_budget=tuning.thinking_budget)),
        )

    def _instruction(self, context: ReadonlyContext) -> str:
        state = dict(context.state)
        task = NarrationTask(state.get(state_keys.NARRATION_TASK, NarrationTask.PRESENT_OPTIONS))
        return self._prompts[task](state)

    # -----------------------------------------------------------------
    # Prompt builders
    # -----------------------------------------------------------------

    def _present_options_prompt(self, state: dict) -> str:
        return PRESENT_OPTIONS.format(
            voice=VOICE,
            need=_need(state).summary(),
            profile=PreferenceProfile.model_validate(state.get(state_keys.PREFERENCES) or {}).to_prompt_text(),
            search_note=_search_note(state),
            products=_presented_text(state),
        )

    def _more_options_prompt(self, state: dict) -> str:
        presented = state.get(state_keys.PRESENTED) or []
        already = max(len(state.get(state_keys.SEEN) or []) - len(presented), 0)
        return MORE_OPTIONS.format(
            voice=VOICE,
            need=_need(state).summary(),
            already_heard=already,
            search_note=_search_note(state),
            products=_presented_text(state),
        )

    def _no_more_options_prompt(self, state: dict) -> str:
        return NO_MORE_OPTIONS.format(
            voice=VOICE,
            need=_need(state).summary(),
            already_heard=len(state.get(state_keys.SEEN) or []),
        )

    def _speak_comparison_prompt(self, state: dict) -> str:
        comparison = state.get(state_keys.COMPARISON)
        text = json.dumps(comparison, ensure_ascii=False, indent=1) if comparison else "No comparison available."
        return SPEAK_COMPARISON.format(voice=VOICE, need=_need(state).summary(), comparison=text)

    def _cart_update_prompt(self, state: dict) -> str:
        return CART_UPDATE.format(
            voice=VOICE,
            event=json.dumps(state.get(state_keys.CART_EVENT) or {}, ensure_ascii=False),
            cart=Cart.model_validate(state.get(state_keys.CART) or {}).to_prompt_text(),
        )


def _need(state: dict) -> ShoppingNeed:
    return ShoppingNeed.model_validate(state.get(state_keys.NEED) or {})


def _presented_text(state: dict) -> str:
    """The products chosen for this turn, each labelled exact or related."""
    by_id = {item.get("id"): item for item in state.get(state_keys.SHORTLIST) or []}
    exact = set(state.get(state_keys.EXACT_MATCHES) or [])
    blocks = []
    for number, product_id in enumerate(state.get(state_keys.PRESENTED) or [], start=1):
        if product_id in by_id:
            label = "[exact match]" if product_id in exact else "[related]"
            product = Product.model_validate(by_id[product_id])
            blocks.append(f"Option {number} {label}:\n{product.to_prompt_text()}")
    return "\n\n".join(blocks) or "None."


def _search_note(state: dict) -> str:
    relaxed = state.get(state_keys.RELAXED) or []
    if "everything" in relaxed:
        return "SEARCH NOTE: nothing matched at all."
    if "budget" in relaxed:
        return "SEARCH NOTE: nothing fit their budget, so these may cost more than they wanted. Say so."
    if relaxed:
        return ("SEARCH NOTE: options marked [related] do not match every detail they asked for; "
                "they are the closest alternatives in the shop.")
    return "SEARCH NOTE: these match what they asked for."
