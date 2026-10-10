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
from typing import Callable, Optional

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.planners import BuiltInPlanner
from google.genai import types

from scout.agents import state_keys
from scout.agents.voice import VOICE
from scout.config import Settings
from scout.domain import ShoppingNeed
from scout.tools.review_evidence import EvidencePicker

_EVIDENCE = EvidencePicker()


class NarrationTask(str, Enum):
    PRESENT_OPTIONS = "present_options"   # speak the first options after a search
    MORE_OPTIONS = "more_options"         # the shopper asked for more; speak the next ones
    NO_MORE_OPTIONS = "no_more_options"   # the shopper asked for more, but there are none
    SPEAK_COMPARISON = "speak_comparison"  # speak the Comparison agent's analysis
    CART_UPDATE = "cart_update"           # confirm what happened to the cart
    CHECKOUT_REMINDER = "checkout_reminder"  # before checkout: the one item they might have forgotten
    CAPSULE = "capsule"                   # the outfit plan on the board: what was planned or changed


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
Make clear these are additional options, then present them: honest about [exact match] vs [related], one sentence each with what makes it different, making their order clear so they can refer back. Options marked [kept] are ones they chose to keep: say in a few words that you kept them, without describing them again, and describe only the new ones. About 60 to 100 words. End with ONE easy question.
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
- Outfit finished (outfit_finished): say their outfit is complete and its new pieces are in the cart, with the total, then ask what they would like to shop for next.
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


CAPSULE = """{voice}
You plan outfits like a stylist friend: their own clothes first, and only the few new pieces worth buying. The plan is also shown on screen as a board, one row per look.

WHAT JUST HAPPENED
{event}

THE PLAN
{capsule}
{options}
YOUR TASK
{task}
Use only what THE PLAN and the options say. Say "your" for the pieces they own. Never read every look piece by piece.
"""

# The end of every reply about the board, so the shopper always knows what they can do next.
BOARD_DONE = "End by asking whether they would like to swap a piece, try a look on, or add the new pieces to their cart."
CHOOSING_ENDS = ("End by asking which one they like, or whether they would like other options or to skip this piece. "
                 "If there are no options, say so kindly and offer to skip it or look for something different.")


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
            NarrationTask.CAPSULE: self._capsule_prompt,
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
            products=_presented_text(state, kept=set(state.get(state_keys.KEPT) or [])),
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
        next_up = state.get(state_keys.NEXT_UP) or []
        offer = ""
        if next_up and event.get("action") == "added":      # they asked for more things: those come first
            offer = (f"\nSTILL TO LOOK FOR (they asked for these too): {', '.join(next_up)}\n"
                     f"Your next step: offer to find the {next_up[0]} now.\n")
        elif suggestion and event.get("action") == "added":   # only right after adding, never when removing
            offer = f"\nSUGGESTION (one item that goes with what they just added)\n{suggestion.to_prompt_text()}\n"
        # The cart below already lists a just-added item, so say plainly that it is new, or the reply can
        # read it as "already there".
        said = {"added": "They asked for it and you have JUST ADDED it; it was not in the cart before.",
                "already_in_cart": "It was ALREADY in the cart before they asked; nothing was added."}
        plain = said.get(event.get("action"), "")
        return CART_UPDATE.format(
            voice=VOICE,
            event=json.dumps(event, ensure_ascii=False) + (f"\n{plain}" if plain else ""),
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


    def _capsule_prompt(self, state: dict) -> str:
        capsule = state_keys.capsule(state)
        event = state.get(state_keys.CAPSULE_EVENT) or "They asked about the plan."
        piece = capsule.new_piece(capsule.current) if capsule and capsule.current else None
        options = ""
        if piece is not None:                    # guided: they are choosing this piece now
            suits = [f"- option {n}: {capsule.fit_notes[pid]}"
                     for n, pid in enumerate(state.get(state_keys.PRESENTED) or [], start=1) if pid in capsule.fit_notes]
            options = (f"\nOPTIONS FOR THE {piece.category.upper()} (what to look for: {piece.looks_like})\n"
                       f"{_search_note(state, capsule.need_for(piece, state_keys.need(state)))}\n"
                       f"{_presented_text(state)}\n"
                       + ("WHY EACH SUITS THE OCCASION (say this for each one)\n" + "\n".join(suits) + "\n"
                          if suits else ""))
            present = (f"present these options for the {piece.category} in about 60 to 90 words: how each would "
                       "suit the occasion and go with the rest of the outfit, honest about [exact match] vs "
                       f"[related], making the order clear.\n{SCREEN_CUE}\n{CHOOSING_ENDS}")
            if event == state_keys.CAPSULE_PLANNED:
                task = ("Share the plan in one or two short sentences: the pieces you would put together for this "
                        "occasion and why, and that you will choose them together one at a time, starting with the "
                        f"{piece.category}. Do not repeat that you are planning. Then {present}")
            else:
                task = f"React to what just happened in one short sentence. Then {present}"
        elif capsule and capsule.guided:         # guided, and every piece is settled
            task = ("In two to four short sentences, describe the whole outfit together like a stylist friend: how "
                    f"the pieces work for the occasion, and the total. {BOARD_DONE}")
        elif event == state_keys.CAPSULE_PLANNED:
            task = ("In about 60 to 100 words, share the plan: how many looks, how much comes from their own wardrobe, "
                    "and which few new pieces you suggest buying, each with how many looks it unlocks. Do not list "
                    "the pieces you left out; at most say in a few words that you left out ones that don't suit it. Mention the total, and the budget if they gave "
                    "one. If a new piece could not be found in the shop, say so honestly. Do not repeat that you "
                    f"are planning: you already said so. {BOARD_DONE}")
        else:
            task = ("In a few short sentences, react to what just happened, then say how it changes the plan and the "
                    f"total. {BOARD_DONE}")
        return CAPSULE.format(voice=VOICE, event=event, capsule=capsule.to_prompt_text() if capsule else "None.",
                              options=options, task=task)


def _heard_text(heard: int) -> str:
    """What is true about the options left, said so the model cannot turn "all heard" into "none in stock"."""
    if heard:
        return (f"They have already heard all {heard} options the shop has that fit. The shop does have those "
                f"{heard}: never say it has none of this kind or is out of stock.")
    return "The shop has nothing that fits what they want now."


def _presented_text(state: dict, kept: frozenset[str] | set[str] = frozenset()) -> str:
    """The products chosen for this turn, each labelled exact or related (and kept, if the shopper kept it)."""
    need = state_keys.need(state)
    blocks = []
    for number, product in enumerate(state_keys.presented(state), start=1):
        label = "[kept]" if product.id in kept else (
            "[exact match]" if state_keys.is_exact(state, product.id) else "[related]")
        evidence = _EVIDENCE.pick(product, need)
        proof = ("\nReview evidence (real quotes): " + "; ".join(e.as_text() for e in evidence)) if evidence else ""
        suits = (state.get(state_keys.FIT_NOTES) or {}).get(product.id)
        occasion = f"\nWhy it suits {need.purpose}: {suits}" if suits and need.purpose else ""
        blocks.append(f"Option {number} {label}:\n{product.to_prompt_text()}{proof}{occasion}")
    return "\n\n".join(blocks) or "None."


def _search_note(state: dict, need: Optional[ShoppingNeed] = None) -> str:
    """What the search had to relax, said plainly. `need`: what was searched, when it is not the shopping need
    (an outfit piece, with its share of the budget)."""
    need = need or state_keys.need(state)
    relaxed = state.get(state_keys.RELAXED) or []
    if "everything" in relaxed:
        note = "SEARCH NOTE: nothing matched at all."
    elif "budget" in relaxed:
        limit = f"under ${need.budget_max:.0f}" if need.budget_max is not None else "within their budget"
        note = (f"SEARCH NOTE: the shop has no {_kind(need)} {limit}. Start by saying exactly that, plainly, in one "
                "short sentence. These are the closest prices above it, cheapest first: present them as that, "
                "with each price.")
    elif relaxed:
        note = ("SEARCH NOTE: options marked [related] do not match every detail they asked for; "
                "they are the closest alternatives in the shop.")
    else:
        note = "SEARCH NOTE: these match what they asked for."
    other_types = state.get(state_keys.OTHER_TYPES) or []
    if other_types:
        note += (f"\nThat is everything the shop has for: {_kind(need)}. It also has these other kinds for them: "
                 f"{', '.join(other_types)}. Say naturally, like a friend, that these are all you have of what they "
                 "asked for (no talk of how items are labelled or 'called'), and in your closing question offer to "
                 "look at one or two of those other kinds instead. Never present them as what they asked for.")
    return note


def _kind(need: ShoppingNeed) -> str:
    kind = " or ".join(need.product_types) or need.category or "items"
    return f"{need.gender_filter()}'s {kind}" if need.gender_filter() else kind
