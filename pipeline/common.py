"""
What every pipeline job shares, so each script only holds its own job:

  - PipelineSettings   project, Gemini model and Cloud SQL settings, from .env
  - CloudSqlDatabase   a Cloud SQL connection (password from Secret Manager)
  - Gemini             structured (JSON) answers from Gemini, retried when Vertex AI is busy
  - image_part         a product photo, ready to send to Gemini

Install:
  pip install "cloud-sql-python-connector[pg8000]" google-cloud-secret-manager google-genai \
              pydantic requests python-dotenv
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, TypeVar

import requests
from dotenv import load_dotenv
from google import genai
from google.cloud import secretmanager
from google.cloud.sql.connector import Connector
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

load_dotenv()

Answer = TypeVar("Answer", bound=BaseModel)


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"Missing environment variable: {name}")
    return value


@dataclass(frozen=True)
class PipelineSettings:
    project_id: str
    vertex_location: str
    model: str                  # Gemini model for every pipeline job
    db_instance: str            # project:region:instance
    db_name: str
    db_user: str
    db_password_secret: str     # Secret Manager secret id holding the password

    @classmethod
    def from_env(cls) -> "PipelineSettings":
        return cls(
            project_id=require_env("GOOGLE_CLOUD_PROJECT"),
            vertex_location=os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip() or "us-central1",
            model=os.environ.get("SCOUT_ENRICH_MODEL", "").strip() or "gemini-2.5-flash",
            db_instance=require_env("SCOUT_DB_INSTANCE"),
            db_name=require_env("SCOUT_DB_NAME"),
            db_user=require_env("SCOUT_DB_USER"),
            db_password_secret=require_env("SCOUT_DB_PASSWORD_SECRET"),
        )


class CloudSqlDatabase:
    """`with CloudSqlDatabase(settings) as connection:` opens and always closes a Cloud SQL connection."""

    def __init__(self, settings: PipelineSettings):
        self._settings = settings
        self._connector = None
        self._connection = None

    def __enter__(self):
        self._connector = Connector()
        self._connection = self._connector.connect(
            self._settings.db_instance, "pg8000",
            user=self._settings.db_user, password=self._read_password(), db=self._settings.db_name,
        )
        return self._connection

    def __exit__(self, *exc):
        if self._connection:
            self._connection.close()
        if self._connector:
            self._connector.close()

    def _read_password(self) -> str:
        name = f"projects/{self._settings.project_id}/secrets/{self._settings.db_password_secret}/versions/latest"
        client = secretmanager.SecretManagerServiceClient()
        return client.access_secret_version(name=name).payload.data.decode("UTF-8").strip()


class Gemini:
    """Structured answers from Gemini on Vertex AI. Waits and retries when the service is busy."""

    RETRY_CODES = (429, 500, 503)          # quota / temporarily unavailable
    RETRY_WAITS_SECONDS = (10, 20, 40, 80)

    def __init__(self, settings: PipelineSettings):
        self._client = genai.Client(vertexai=True, project=settings.project_id, location=settings.vertex_location)
        self.model = settings.model

    def answer(self, parts: list[types.Part], schema: type[Answer], temperature: float = 0.0) -> Answer:
        """Gemini's answer to `parts`, in the shape of `schema`."""
        for attempt, wait in enumerate((*self.RETRY_WAITS_SECONDS, None), start=1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(temperature=temperature,
                                                       response_mime_type="application/json",
                                                       response_schema=schema),
                )
                return schema.model_validate_json(response.text)
            except genai_errors.APIError as error:
                if error.code not in self.RETRY_CODES or wait is None:
                    raise
                print(f"  ~ Gemini returned {error.code}; retry {attempt} in {wait}s")
                time.sleep(wait)


def image_part(url: Optional[str], timeout_seconds: float = 15) -> Optional[types.Part]:
    """A product photo as a Gemini part, or None if it can't be downloaded (the text is usually enough)."""
    if not url:
        return None
    try:
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"  ! image download failed ({url}): {error}")
        return None
    mime_type = response.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    return types.Part.from_bytes(data=response.content, mime_type=mime_type)
