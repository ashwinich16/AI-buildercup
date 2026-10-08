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
from scout.tools.review_evidence import EvidencePicker

_EVIDENCE = EvidencePicker()


class NarrationTask(str, Enum):
    PRESENT_OPTIONS = "present_options"   # speak the first options after a search
    MORE_OPTIONS = "more_options"         # the shopper asked for more; speak the next ones
    NO_MORE_OPTIONS = "no_more_options"   # the shopper asked for more, but there are none
    SPEAK_COMPARISON = "speak_comparison"  # speak the Comparison agent's analysis
    CART_UPDATE = "cart_update"           # confirm what happened to the cart
    CHECKOUT_REMINDER = "checkout_reminder"  # before checkout: the one item they might have forgotten
    OUTFIT_STEP = "outfit_step"           # options for the next piece of the outfit being put together
    OUTFIT_DONE = "outfit_done"           # every piece picked: the whole outfit, and one offer


# Lights up each option on screen while Scout talks about it (the frontend reads and hides the tags).
SCREEN_CUE = ("Screen cue: right before you start talking about an option, write its tag, [[1]] for Option 1, "
              "[[2]] for Option 2, and so on (the option numbers below). Tags are never spoken; they light up that "
              "option on screen while you describe it. Write each tag once, at the start of the sentence that begins "
              "that option.")


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
3. Lead with the best option and WHY it suits what they want, plus one detail that helps them picture it. Back the why with ONE short quote from its review evidence, said naturally with how many reviews say it (for example, that three of ten reviewers say it). Quote only the evidence given, word for word.
4. Then each other option in one sentence, with the one way it differs.
5. Mention a review point only if it matters for their need.
6. Make the order clear as you speak (first, second...) so they can refer back.
{cue}
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
{cue}
"""

NO_MORE_OPTIONS = """{voice}
WHAT THE SHOPPER WANTS
{need}

They asked for more or different options. {heard}
{other_kinds}
YOUR TASK
In about 30 words, say kindly and honestly that those were all the options you have for this, and offer ONE helpful change to their need that would open up new options (if other kinds are listed above, offer one or two of them). Do not invent products, and never claim more than the line above says.
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
{suggestion}
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
Refer to cart items by the kind given, never by a guessed kind.
If a SUGGESTION is given above, end with it instead of another next step: in one natural sentence, the way a friend in a shop would, mention that item, say why it goes with what they just chose (only the reasons given), and ask whether you should add it to their cart or tell them more about it first. Never mention any other item.
"""


CHECKOUT_REMINDER = """{voice}
They asked to check out. Before they do, there is one item that would complete what is in their cart:

SUGGESTION
{suggestion}

THEIR CART
{cart}

YOUR TASK
In one or two short, natural sentences, like a helpful friend at the till: say that before they check out they might like this one item, why it goes with what they chose (only the reasons given), and ask whether you should add it, tell them more about it, or go ahead and check out. Do not read back the cart or the total yet.
"""


OUTFIT_STEP = """{voice}
You are helping them put together an outfit, one piece at a time, like a stylist friend.

THE OUTFIT
{outfit}

NOW CHOOSING
{piece}

{search_note}

OPTIONS FOR THIS PIECE
{products}

YOUR TASK
{opening}
Then present these options for this piece in about 70 to 100 words: how each would go with the occasion and with what they have picked so far, honest about [exact match] vs [related], one detail that helps them picture it, making the order clear.
{cue}
End by asking which one they like, or whether they would rather skip this piece. If there are no options, say so kindly and offer to skip this piece or change what you look for.
"""


OUTFIT_DONE = """{voice}
They have picked every piece of their outfit.

THE OUTFIT
{outfit}

YOUR TASK
In two to four short sentences, describe the whole outfit together like a stylist friend: how the pieces work together for the occasion, and the total. Then {offer}
"""


class NarrationAgentBuilder:
    """Builds the Narration LlmAgent. One prompt builder per NarrationTask."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._prompts: dict[NarrationTask, Callable[[dict], str]] = {
            NarrationTask.PRESENT_OPTIONS: self._present_options_prompt,
            NarrationTask.MORE_OPTIONS: self._more_options_prompt,
            NarrationTask.NO_MORE_OPTIONS: self._no_more_options_prompt,
            NarrationTask.SPEAK_COMPARISON: self._speak_comparison_prompt,
            NarrationTask.CART_UPDATE: self._cart_update_prompt,
            NarrationTask.CHECKOUT_REMINDER: self._checkout_reminder_prompt,
            NarrationTask.OUTFIT_STEP: self._outfit_step_prompt,
            NarrationTask.OUTFIT_DONE: self._outfit_done_prompt,
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
            cue=SCREEN_CUE,
            voice=VOICE,
            need=state_keys.need(state).summary(),
            profile=state_keys.profile(state).to_prompt_text(),
            search_note=_search_note(state),
            products=_presented_text(state),
        )

    def _more_options_prompt(self, state: dict) -> str:
        presented = state.get(state_keys.PRESENTED) or []
        already = max(len(state.get(state_keys.SEEN) or []) - len(presented), 0)
        return MORE_OPTIONS.format(
            cue=SCREEN_CUE,
            voice=VOICE,
            need=state_keys.need(state).summary(),
            already_heard=already,
            search_note=_search_note(state),
            products=_presented_text(state),
        )

    def _no_more_options_prompt(self, state: dict) -> str:
        other_types = state.get(state_keys.OTHER_TYPES) or []
        return NO_MORE_OPTIONS.format(
            voice=VOICE,
            need=state_keys.need(state).summary(),
            heard=_heard_text(len(state.get(state_keys.SEEN) or [])),
            other_kinds=f"Other kinds the shop has for them: {', '.join(other_types)}.\n" if other_types else "",
        )

    def _speak_comparison_prompt(self, state: dict) -> str:
        comparison = state.get(state_keys.COMPARISON)
        text = json.dumps(comparison, ensure_ascii=False, indent=1) if comparison else "No comparison available."
        return SPEAK_COMPARISON.format(voice=VOICE, need=state_keys.need(state).summary(), comparison=text)

    def _cart_update_prompt(self, state: dict) -> str:
        event = state.get(state_keys.CART_EVENT) or {}
        suggestion = state_keys.suggestion(state)
        offer = ""
        if suggestion and event.get("action") == "added":   # only right after adding, never when removing
            offer = f"\nSUGGESTION (one item that goes with what they just added)\n{suggestion.to_prompt_text()}\n"
        return CART_UPDATE.format(
            voice=VOICE,
            event=json.dumps(event, ensure_ascii=False),
            cart=state_keys.cart(state).to_prompt_text(),
            suggestion=offer,
        )

    def _checkout_reminder_prompt(self, state: dict) -> str:
        suggestion = state_keys.suggestion(state)
        return CHECKOUT_REMINDER.format(
            voice=VOICE,
            suggestion=suggestion.to_prompt_text() if suggestion else "None.",
            cart=state_keys.cart(state).to_prompt_text(),
        )


    def _outfit_step_prompt(self, state: dict) -> str:
        outfit = state_keys.outfit(state)
        need = state_keys.need(state)
        # The piece these options are for: the plan's piece of that kind, or what they asked for instead.
        piece = next((p for p in outfit.plan.pieces if p.category == need.category), None) if outfit else None
        if outfit and not outfit.introduced:
            opening = ("Start by sharing the plan in one or two sentences: the pieces you would put together for "
                       "this occasion and why, and that you will start with this piece. Mention the pieces that "
                       "would finish the look only as ideas for later.")
        else:
            opening = ("Start with a short, warm reaction to what just happened (their pick, or the piece they skipped "
                       "or added), and say which piece comes next.")
        return OUTFIT_STEP.format(
            cue=SCREEN_CUE,
            voice=VOICE,
            outfit=outfit.to_prompt_text() if outfit else "None.",
            piece=f"{piece.category}: {piece.looks_like} ({piece.why})" if piece else need.summary(),
            search_note=_search_note(state),
            products=_presented_text(state),
            opening=opening,
        )

    def _outfit_done_prompt(self, state: dict) -> str:
        outfit = state_keys.outfit(state)
        extra = outfit.offered_extra if outfit else None
        if extra:
            offer = (f"ask, without any pressure, whether they would like you to find a {extra.category} to finish "
                     f"the look ({extra.looks_like}; {extra.why}). Make clear it is optional.")
        else:
            offer = "ask whether you should put the whole outfit in their cart."
        return OUTFIT_DONE.format(voice=VOICE, outfit=outfit.to_prompt_text() if outfit else "None.", offer=offer)


def _heard_text(heard: int) -> str:
    """What is true about the options left, said so the model cannot turn "all heard" into "none in stock"."""
    if heard:
        return (f"They have already heard all {heard} options the shop has that fit. The shop does have those "
                f"{heard}: never say it has none of this kind or is out of stock.")
    return "The shop has nothing that fits what they want now."


def _presented_text(state: dict) -> str:
    """The products chosen for this turn, each labelled exact or related."""
    need = state_keys.need(state)
    blocks = []
    for number, product in enumerate(state_keys.presented(state), start=1):
        label = "[exact match]" if state_keys.is_exact(state, product.id) else "[related]"
        evidence = _EVIDENCE.pick(product, need)
        proof = ("\nReview evidence (real quotes): " + "; ".join(e.as_text() for e in evidence)) if evidence else ""
        blocks.append(f"Option {number} {label}:\n{product.to_prompt_text()}{proof}")
    return "\n\n".join(blocks) or "None."


def _search_note(state: dict) -> str:
    relaxed = state.get(state_keys.RELAXED) or []
    if "everything" in relaxed:
        note = "SEARCH NOTE: nothing matched at all."
    elif "budget" in relaxed:
        budget = state_keys.need(state).budget_max
        limit = f"under ${budget:.0f}" if budget is not None else "within their budget"
        note = (f"SEARCH NOTE: the shop has no {_kind(state)} {limit}. Start by saying exactly that, plainly, in one "
                "short sentence. These are the closest prices above it, cheapest first: present them as that, "
                "with each price.")
    elif relaxed:
        note = ("SEARCH NOTE: options marked [related] do not match every detail they asked for; "
                "they are the closest alternatives in the shop.")
    else:
        note = "SEARCH NOTE: these match what they asked for."
    other_types = state.get(state_keys.OTHER_TYPES) or []
    if other_types:
        note += (f"\nThat is everything the shop has for: {_kind(state)}. It also has these other kinds for them: "
                 f"{', '.join(other_types)}. Say naturally, like a friend, that these are all you have of what they "
                 "asked for (no talk of how items are labelled or 'called'), and in your closing question offer to "
                 "look at one or two of those other kinds instead. Never present them as what they asked for.")
    return note


def _kind(state: dict) -> str:
    need = state_keys.need(state)
    kind = " or ".join(need.product_types) or need.category or "items"
    return f"{need.gender_filter()}'s {kind}" if need.gender_filter() else kind
