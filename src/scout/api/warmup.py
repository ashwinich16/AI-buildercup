"""
Keeps Scout's cloud connections warm, so the shopper never waits for a cold start.

The first call on a new Gemini connection takes about 5 seconds (credentials and
connection set-up); later calls take under 1 second. Each agent has its own
connection, so without this the first search of a conversation paid that price
twice (Elicitation, then Narration). Vertex AI Search and Text-to-Speech have a
smaller cold start of their own.

At server start, and every few minutes after (idle connections go cold again),
this sends one tiny request through each of them. It never affects a
conversation: failures are logged and ignored.
"""

import asyncio
import logging
import time
from typing import Callable

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.models import LlmRequest
from google.genai import types

logger = logging.getLogger(__name__)

KEEP_WARM_SECONDS = 240


def _llm_agents(agent: BaseAgent) -> list[LlmAgent]:
    found = [agent] if isinstance(agent, LlmAgent) else []
    for child in agent.sub_agents:
        found += _llm_agents(child)
    return found


async def _warm_model(agent: LlmAgent) -> None:
    model = agent.canonical_model
    request = LlmRequest(
        model=model.model,
        contents=[types.Content(role="user", parts=[types.Part(text="Reply with OK.")])],
        config=types.GenerateContentConfig(max_output_tokens=8,
                                           thinking_config=types.ThinkingConfig(thinking_budget=0)),
    )
    async for _ in model.generate_content_async(request):
        pass


async def warm_up(root: BaseAgent, others: list[Callable[[], object]]) -> None:
    """One tiny request through every Gemini agent, plus each blocking client in `others`."""
    started = time.monotonic()
    jobs = [_warm_model(agent) for agent in _llm_agents(root)]
    jobs += [asyncio.to_thread(call) for call in others]
    results = await asyncio.gather(*jobs, return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception)]
    for failure in failures:
        logger.warning("Warm-up call failed (ignored): %s", failure)
    logger.info("Warmed %d connections in %.1fs.", len(results) - len(failures), time.monotonic() - started)


async def keep_warm(root: BaseAgent, others: list[Callable[[], object]]) -> None:
    while True:
        await warm_up(root, others)
        await asyncio.sleep(KEEP_WARM_SECONDS)
