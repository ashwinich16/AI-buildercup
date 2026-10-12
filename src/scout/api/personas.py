"""
Demo shoppers for the hackathon. The data lives in personas.json (not in code),
so personas can be changed without touching Python.
"""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from scout.domain import Preference

_FILE = Path(__file__).with_name("personas.json")


class Persona(BaseModel):
    id: str
    name: str
    description: str
    guest: bool = False      # a fresh, nameless shopper each time it is picked (for trying Scout out)
    preferences: list[Preference] = Field(default_factory=list)


@lru_cache(maxsize=1)
def load_personas() -> dict[str, Persona]:
    items = json.loads(_FILE.read_text(encoding="utf-8"))
    return {item["id"]: Persona.model_validate(item) for item in items}
