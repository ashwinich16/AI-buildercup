"""
Preference agent: Scout's long-term memory of the shopper (taste, size and fit,
budget habits, and how they like Scout to talk with them).

Runs AFTER Scout has replied, so it never slows the conversation. From what
happened this turn (a product chosen or rejected, or something the shopper
said about themselves) it extracts LASTING preferences and merges them into
the shopper's profile. Merging is done in code (PreferenceProfile.merge):
stated preferences are certain, learned ones start with low confidence and
only grow with repeated evidence.
"""

import json
import logging
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmResponse
from google.adk.planners import BuiltInPlanner
from google.genai import types
from pydantic import BaseModel, Field

from scout.agents import state_keys
from scout.agents.model_io import response_json
from scout.config import Settings
from scout.domain import Preference, PreferenceKey

logger = logging.getLogger(__name__)


class PreferenceUpdate(BaseModel):
    updates: list[Preference] = Field(default_factory=list, description="New or reinforced lasting preferences")
    forget: list[PreferenceKey] = Field(default_factory=list, description="Preferences the shopper asked to drop")


INSTRUCTION = """You keep the long-term profile of a shopper using Scout, a shopping companion for blind and low-vision people. Good memory means Scout never makes them repeat themselves and gets better at helping them each time.

CURRENT PROFILE
{profile}

THEIR TASTE, LEARNED FROM WHAT THEY OWN AND SHOP (kept by code; you never add to it)
{taste}

WHAT HAPPENED THIS TURN
{evidence}

The shopper's own message is the user message.

YOUR JOB
Return only LASTING personal preferences: things likely to still be true next time they shop. Ignore details that only apply to this one purchase.

What is worth remembering, all of it only when it helps choose products or talk with them:
- Taste: colours, materials, styles, feel and details they like or avoid.
- Size and fit: their sizes and how they like things to fit.
- Budget habits: the price range they are usually comfortable with.
- How they like Scout to talk: how much detail they want, how many options at a time, the pace.
- Who they usually shop for, and practical needs that decide which products suit them.
Never record guesses about their personality or mood, and never record private matters that do not affect shopping.

How to write each one:
- source "stated": they said it about themselves in general. Confidence 1.0.
- source "learned": you infer it from what they chose or rejected, and their reason. Confidence 0.3 to 0.5. Only infer when the reason points to a lasting taste, not to this one product.
- attribute: a short, general name for the kind of preference, so the same kind is always stored the same way. Reuse an attribute name already in CURRENT PROFILE when it fits.
- value: plain words; for taste, the words a product description would use.
- scope: the catalog category if it only applies to that kind of product; null if it applies to everything.
- polarity: like or avoid.
- If they ask Scout to forget or stop assuming something, put it in forget (also for something in THEIR TASTE: its value in value, the kind of taste in attribute).
- If nothing lasting was revealed, return empty lists. That is the most common, correct answer.
"""


class PreferenceAgentBuilder:
    def __init__(self, settings: Settings):
        self._settings = settings

    def build(self) -> LlmAgent:
        tuning = self._settings.tuning("preference")
        return LlmAgent(
            name="preference",
            model=self._settings.llm("preference"),
            description="Learns lasting shopper preferences after each turn.",
            instruction=self._instruction,
            include_contents="none",
            output_schema=PreferenceUpdate,
            generate_content_config=types.GenerateContentConfig(temperature=tuning.temperature),
            planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(
                thinking_budget=tuning.thinking_budget)),
            after_model_callback=self._merge,
        )

    def _instruction(self, context: ReadonlyContext) -> str:
        state = context.state
        profile = state_keys.profile(state)
        return INSTRUCTION.format(
            profile=json.dumps([p.model_dump() for p in profile.preferences], ensure_ascii=False) or "[]",
            evidence=state.get(state_keys.PREFERENCE_EVIDENCE) or "The shopper said something about themselves.",
            taste=state_keys.taste(state).to_prompt_text(None),
        )

    # Merge in code, so the confidence rules are applied exactly, not left to the model.
    def _merge(self, callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
        update = response_json(llm_response, PreferenceUpdate)
        if update and (update.updates or update.forget):
            state = callback_context.state
            profile = state_keys.profile(state)
            state[state_keys.PREFERENCES] = profile.merge(update.updates, update.forget).model_dump()
            taste = state_keys.taste(state)
            if taste.forget([key.value for key in update.forget]):     # "forget that I like navy": learned too
                state[state_keys.TASTE] = taste.model_dump()
            logger.info("Preferences updated: +%d, forgotten %d", len(update.updates), len(update.forget))
        return None
