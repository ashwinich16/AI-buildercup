"""
Scout settings.

Every setting comes from environment variables (or a .env file when running
locally), so the same code runs in Cloud Shell, Cloud Run or a laptop.
Nothing else in the app reads os.environ directly: other modules ask for
`get_settings()` and receive one immutable Settings object.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class MissingSettingError(RuntimeError):
    """Raised when a required environment variable is not set."""


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingSettingError(f"Missing environment variable: {name}")
    return value


def _optional(name: str, default: str = "") -> str:
    """The variable's value, or the default when it is missing or left empty (e.g. `NAME=` in .env)."""
    return os.environ.get(name, "").strip() or default


@dataclass(frozen=True)
class AgentTuning:
    temperature: float      # wording variety; does not control reasoning
    thinking_budget: int    # tokens Gemini may spend reasoning; 0 = off (fastest)
    model: str = ""         # empty = use SCOUT_AGENT_MODEL


# Defaults per LLM agent. Override any of them with SCOUT_<AGENT>_TEMPERATURE,
# SCOUT_<AGENT>_THINKING_BUDGET and SCOUT_<AGENT>_MODEL, e.g. SCOUT_PRODUCT_EXPERT_MODEL.
# A new agent only needs one line here.
AGENT_TUNING_DEFAULTS = {
    "elicitation": AgentTuning(temperature=0.5, thinking_budget=512),       # 512 keeps the asking rules; 256 broke them
    "narration": AgentTuning(temperature=0.7, thinking_budget=0),           # wording only (no thinking: ~1s faster)
    "product_expert": AgentTuning(temperature=0.3, thinking_budget=1024),   # factual + visual
    "comparison": AgentTuning(temperature=0.2, thinking_budget=2048),       # deepest reasoning, consistent
    "preference": AgentTuning(temperature=0.2, thinking_budget=1024),       # runs after the reply: no rush
    "recommender": AgentTuning(temperature=0.2, thinking_budget=0),         # which kinds go together (cached)
    "stylist": AgentTuning(temperature=0.6, thinking_budget=512),           # plans looks from the wardrobe
    "wardrobe": AgentTuning(temperature=0.2, thinking_budget=0),            # describes a photo of owned clothes
    "occasion": AgentTuning(temperature=0.1, thinking_budget=512),         # judges each product for the occasion
}


def _tuning_from_env(agent: str, default: AgentTuning) -> AgentTuning:
    prefix = f"SCOUT_{agent.upper()}_"
    return AgentTuning(
        temperature=float(_optional(prefix + "TEMPERATURE", str(default.temperature))),
        thinking_budget=int(_optional(prefix + "THINKING_BUDGET", str(default.thinking_budget))),
        model=_optional(prefix + "MODEL", default.model),
    )


@dataclass(frozen=True)
class Settings:
    # Google Cloud / Gemini
    project_id: str
    vertex_location: str
    agent_model: str
    agent_tuning: dict[str, AgentTuning]

    # Elicitation
    elicitation_safety_max_questions: int   # guard against loops, not a target

    # Virtual try-on (Vertex AI)
    try_on_model: str
    try_on_location: str        # virtual try-on is offered in a few regions only

    # Web app
    auth_mode: str              # "demo" (persona picker) or "firebase" (Sign in with Google)
    firebase_api_key: str       # Firebase web API key (public by design); firebase mode only
    firebase_auth_domain: str   # usually <project>.firebaseapp.com
    tts_voice: str              # Google Cloud Text-to-Speech voice name

    # Cloud SQL: shopper memory (name, preferences, cart); empty instance = in memory
    db_instance: str            # project:region:instance
    db_name: str
    db_user: str
    db_password_secret: str     # Secret Manager secret id holding the password

    # Vertex AI Search
    search_location: str
    search_engine_id: str
    search_data_store_id: str
    search_result_limit: int

    def tuning(self, agent: str) -> AgentTuning:
        return self.agent_tuning.get(agent, AgentTuning(temperature=0.5, thinking_budget=0))

    def model_for(self, agent: str) -> str:
        """Each agent can run on its own model; falls back to SCOUT_AGENT_MODEL."""
        return self.tuning(agent).model or self.agent_model

    def llm(self, agent: str):
        """The Gemini model for an agent. When Gemini is briefly busy (429) or unavailable, the call is tried
        again after a short wait instead of failing the shopper's turn."""
        from google.adk.models import Gemini
        return Gemini(model=self.model_for(agent), retry_options=_retry())

    def genai_client(self, location: str = ""):
        """A Gemini client for direct calls (pairings, occasion ranking, wardrobe photos, try-on), with the
        same retries."""
        from google import genai
        from google.genai import types
        return genai.Client(vertexai=True, project=self.project_id, location=location or self.vertex_location,
                            http_options=types.HttpOptions(retry_options=_retry()))

    @property
    def search_serving_config(self) -> str:
        """Full resource name of the Vertex AI Search serving config."""
        base = (f"projects/{self.project_id}/locations/{self.search_location}"
                f"/collections/default_collection")
        if self.search_engine_id:
            return f"{base}/engines/{self.search_engine_id}/servingConfigs/default_search"
        if self.search_data_store_id:
            return f"{base}/dataStores/{self.search_data_store_id}/servingConfigs/default_search"
        raise MissingSettingError(
            "Set SCOUT_SEARCH_ENGINE_ID or SCOUT_SEARCH_DATA_STORE_ID")

    @classmethod
    def from_env(cls) -> "Settings":
        project_id = _required("GOOGLE_CLOUD_PROJECT")
        return cls(
            project_id=project_id,
            vertex_location=_optional("GOOGLE_CLOUD_LOCATION", "us-central1"),
            agent_model=_optional("SCOUT_AGENT_MODEL", "gemini-2.5-flash"),
            agent_tuning={name: _tuning_from_env(name, default)
                          for name, default in AGENT_TUNING_DEFAULTS.items()},
            elicitation_safety_max_questions=int(_optional("SCOUT_ELICITATION_SAFETY_MAX_QUESTIONS", "6")),
            try_on_model=_optional("SCOUT_TRY_ON_MODEL", "virtual-try-on-001"),
            try_on_location=_optional("SCOUT_TRY_ON_LOCATION", "us-central1"),
            auth_mode=_optional("SCOUT_AUTH_MODE", "demo").lower(),
            firebase_api_key=_optional("SCOUT_FIREBASE_API_KEY"),
            firebase_auth_domain=_optional("SCOUT_FIREBASE_AUTH_DOMAIN", f"{project_id}.firebaseapp.com"),
            tts_voice=_optional("SCOUT_TTS_VOICE", "en-US-Chirp3-HD-Aoede"),
            db_instance=_optional("SCOUT_DB_INSTANCE"),
            db_name=_optional("SCOUT_DB_NAME", "scout_catalog"),
            db_user=_optional("SCOUT_DB_USER", "scout_app"),
            db_password_secret=_optional("SCOUT_DB_PASSWORD_SECRET"),
            search_location=_optional("SCOUT_SEARCH_LOCATION", "global"),
            search_engine_id=_optional("SCOUT_SEARCH_ENGINE_ID"),
            search_data_store_id=_optional("SCOUT_SEARCH_DATA_STORE_ID"),
            search_result_limit=int(_optional("SCOUT_SEARCH_RESULT_LIMIT", "8")),
        )


def _retry():
    from google.genai import types
    return types.HttpRetryOptions(attempts=4, initial_delay=1.0, max_delay=8.0, exp_base=2.0,
                                  http_status_codes=[429, 500, 503])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read the environment once and reuse the same Settings everywhere."""
    return Settings.from_env()
