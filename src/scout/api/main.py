"""
Scout web API (FastAPI). Serves the React app and these endpoints:

  GET  /api/auth-config     how the frontend signs in (demo personas or Firebase settings)
  POST /api/sessions        start a conversation; returns Scout's opening words
  POST /api/chat            send a message; get Scout's reply + what to show
  GET  /api/screen          current products and cart (e.g. after a reload)
  POST /api/speech          Scout's reply as natural speech (MP3)

The shopper (id and name) always comes from the Authenticator, never from the
request body, so one shopper can never read or change another shopper's
conversation or cart, and Scout greets them by their verified name.

Run locally (from the repository root):
    PYTHONPATH=src uvicorn scout.api.main:app --reload --port 8000
"""

import asyncio
import logging
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scout.api.auth import Authenticator, DemoPersonaAuthenticator, FirebaseAuthenticator, Shopper
from scout.api.chat_service import ChatTurn, ScoutChatService, ScreenState, StartedSession
from scout.api.personas import load_personas
from scout.api.shopper_repository import build_shopper_repository
from scout.api.speech import SpeechSynthesizer
from scout.app import build_root_agent
from scout.config import Settings, get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=2000)


class StartRequest(BaseModel):
    preferred_name: str = Field(default="", max_length=40)   # "What should Scout call you?" (optional)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    rate: float = 1.0


def build_authenticator(settings: Settings) -> Authenticator:
    if settings.auth_mode == "firebase":
        return FirebaseAuthenticator(settings.project_id, settings.firebase_api_key, settings.firebase_auth_domain)
    return DemoPersonaAuthenticator(load_personas())


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    chat = ScoutChatService(build_root_agent(settings), build_shopper_repository(settings))
    auth = build_authenticator(settings)
    speech = SpeechSynthesizer(settings.tts_voice)
    personas = load_personas()

    app = FastAPI(title="Scout", description="Voice-first shopping companion")

    async def current_shopper(request: Request) -> Shopper:
        return await auth.shopper(request)

    @app.get("/api/auth-config")
    async def auth_config() -> dict:
        return auth.client_config()

    @app.post("/api/sessions")
    async def start_session(body: StartRequest, shopper: Shopper = Depends(current_shopper)) -> StartedSession:
        persona = personas.get(shopper.id)
        try:
            return await chat.start_session(shopper.id, shopper.first_name,
                                            persona.preferences if persona else None,
                                            preferred_name=body.preferred_name.strip())
        except Exception:
            logger.exception("Could not start a conversation for %s", shopper.id)
            raise HTTPException(status_code=503, detail="I can't reach your saved cart and preferences "
                                                        "right now. Please try again in a moment.")

    @app.post("/api/chat")
    async def send_message(body: ChatRequest, shopper: Shopper = Depends(current_shopper)) -> ChatTurn:
        try:
            return await chat.send(shopper.id, body.session_id, body.message.strip())
        except LookupError:
            raise HTTPException(status_code=404, detail="Conversation not found. Please start again.")
        except ValueError as error:          # ADK raises this for an unknown session
            raise HTTPException(status_code=404, detail=str(error))

    @app.get("/api/screen")
    async def current_screen(session_id: str, shopper: Shopper = Depends(current_shopper)) -> ScreenState:
        try:
            return await chat.screen(shopper.id, session_id)
        except LookupError:
            raise HTTPException(status_code=404, detail="Conversation not found.")

    @app.post("/api/speech")
    async def speak(body: SpeechRequest, _: Shopper = Depends(current_shopper)) -> Response:
        audio = await asyncio.to_thread(speech.synthesize, body.text, body.rate)
        return Response(content=audio, media_type="audio/mpeg")

    if FRONTEND_DIST.exists():               # the built React app, served from the same Cloud Run service
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

    return app


app = create_app()
