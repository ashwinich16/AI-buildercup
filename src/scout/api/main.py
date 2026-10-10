"""
Scout web API (FastAPI). Serves the React app and these endpoints:

  GET  /api/auth-config     how the frontend signs in (demo personas or Firebase settings)
  POST /api/sessions        carry on the shopper's last conversation (with what was said), or start a new one
                            ({"fresh": true}); returns Scout's opening words
  DELETE /api/sessions/{id} clear a conversation (the shopper's name, preferences, cart and wardrobe are kept)
  POST /api/chat/stream     send a message; Scout's reply streams back as it is written, then what
                            to show (newline-delimited JSON)
  POST /api/speech          Scout's reply as natural speech (MP3)
  POST /api/try-on          the shopper's photo wearing the chosen products (PNG); the photo is never stored
  POST /api/wardrobe        add the pieces they own in a photo (only a small thumbnail is kept); the chat
                            sends each photo as soon as it is attached, so it is read before they press send
  DELETE /api/wardrobe/{id} take a piece out of their wardrobe

The shopper (id and name) always comes from the Authenticator, never from the
request body, so one shopper can never read or change another shopper's
conversation or cart, and Scout greets them by their verified name.

Run locally (from the repository root):
    PYTHONPATH=src uvicorn scout.api.main:app --reload --port 8000
"""

import asyncio
import base64
import binascii
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scout.api.auth import Authenticator, DemoPersonaAuthenticator, FirebaseAuthenticator, Shopper
from scout.api.chat_service import ScoutChatService, StartedSession
from scout.api.personas import load_personas
from scout.api.shopper_repository import build_shopper_repository
from scout.api.speech import SpeechSynthesizer
from scout.api.warmup import keep_warm
from scout.app import build_catalog, build_root_agent
from scout.catalog import SearchQuery
from scout.config import Settings, get_settings
from scout.tools import ImageFetcher
from scout.tools.try_on import TryOnError, TryOnStudio
from scout.tools.wardrobe import WardrobeTagger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=2000)


class StartRequest(BaseModel):
    preferred_name: str = Field(default="", max_length=40)   # "What should Scout call you?" (optional)
    fresh: bool = False          # start a new conversation instead of carrying on the last one ("New chat")


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    rate: float = 1.0


class TryOnRequest(BaseModel):
    session_id: str
    product_ids: list[str] = Field(default_factory=list, max_length=4)   # one product, or an outfit's new pieces
    wardrobe_ids: list[str] = Field(default_factory=list, max_length=4)  # their own pieces in that look
    photo: str                                                   # the shopper's photo, base64
    mime_type: str = Field(default="image/jpeg", pattern=r"^image/(jpeg|png)$")   # what the model accepts


class WardrobePhotoRequest(BaseModel):
    session_id: str
    photo: str                                   # the piece of clothing, base64
    mime_type: str = Field(default="image/jpeg", pattern=r"^image/(jpeg|png|webp)$")
    thumbnail: str = Field(default="", max_length=300_000)   # data URL made by the browser: shown, and tried on


MAX_PHOTO_BYTES = 10 * 1024 * 1024   # Vertex AI Virtual Try-On's input limit


def build_authenticator(settings: Settings) -> Authenticator:
    if settings.auth_mode == "firebase":
        return FirebaseAuthenticator(settings.project_id, settings.firebase_api_key, settings.firebase_auth_domain)
    return DemoPersonaAuthenticator(load_personas())


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    catalog = build_catalog(settings)
    root = build_root_agent(settings, catalog)
    store = build_shopper_repository(settings)          # the shopper's memory and the conversation summaries
    chat = ScoutChatService(root, store, store)
    auth = build_authenticator(settings)
    speech = SpeechSynthesizer(settings.tts_voice)
    studio = TryOnStudio(settings, ImageFetcher())
    tagger = WardrobeTagger(settings, catalog)
    personas = load_personas()

    # Warm every cloud connection at start-up and keep it warm, so no shopper waits for a cold start.
    # An empty search matches any catalog; nothing here depends on what the shop sells.
    warm_calls = [catalog.categories, catalog.product_types,
                  lambda: catalog.search(SearchQuery(text="", limit=1)),
                  lambda: speech.synthesize("Hi"),
                  root.recommender.prepare]   # what goes with what, so adding to the cart never waits on it

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(keep_warm(root, warm_calls))
        yield
        task.cancel()

    app = FastAPI(title="Scout", description="Voice-first shopping companion", lifespan=lifespan)

    async def current_shopper(request: Request) -> Shopper:
        return await auth.shopper(request)

    @app.get("/api/auth-config")
    async def auth_config() -> dict:
        return auth.client_config()

    @app.post("/api/sessions")
    async def start_session(body: StartRequest, shopper: Shopper = Depends(current_shopper)) -> StartedSession:
        persona = personas.get(shopper.id)
        try:
            started = await chat.start_session(shopper.id, shopper.first_name,
                                               persona.preferences if persona else None,
                                               preferred_name=body.preferred_name.strip(), fresh=body.fresh)
            return started.model_copy(update={"categories": await asyncio.to_thread(catalog.categories)})
        except Exception:
            logger.exception("Could not start a conversation for %s", shopper.id)
            raise HTTPException(status_code=503, detail="I can't reach your saved cart and preferences "
                                                        "right now. Please try again in a moment.")

    @app.delete("/api/sessions/{session_id}")
    async def end_session(session_id: str, shopper: Shopper = Depends(current_shopper)) -> dict:
        try:
            await chat.end_session(shopper.id, session_id)
        except LookupError:
            pass                                   # already gone: nothing to clear
        return {"cleared": True}

    @app.post("/api/chat/stream")
    async def stream_message(body: ChatRequest, shopper: Shopper = Depends(current_shopper)) -> StreamingResponse:
        try:
            await chat.ensure_session(shopper.id, body.session_id)   # 404 before streaming starts, not halfway
        except LookupError:
            raise HTTPException(status_code=404, detail="Conversation not found. Please start again.")

        async def lines():
            try:
                async for item in chat.send_stream(shopper.id, body.session_id, body.message.strip()):
                    yield json.dumps(item) + "\n"
            except Exception:
                logger.exception("Streaming turn failed for %s", shopper.id)
                yield json.dumps({"type": "error", "detail": "Something went wrong on my side."}) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    @app.post("/api/speech")
    async def speak(body: SpeechRequest, _: Shopper = Depends(current_shopper)) -> Response:
        audio = await asyncio.to_thread(speech.synthesize, body.text, body.rate)
        return Response(content=audio, media_type="audio/mpeg")

    @app.post("/api/try-on")
    async def try_on(body: TryOnRequest, shopper: Shopper = Depends(current_shopper)) -> Response:
        try:
            products = await chat.products(shopper.id, body.session_id, body.product_ids)
            owned = await chat.wardrobe_items(shopper.id, body.session_id, body.wardrobe_ids)
            photo = base64.b64decode(body.photo, validate=True)
        except LookupError:
            raise HTTPException(status_code=404, detail="Conversation not found. Please start again.")
        except binascii.Error:
            raise HTTPException(status_code=400, detail="That photo couldn't be read.")
        if not products and not owned:
            raise HTTPException(status_code=404, detail="I can't find those items in our conversation.")
        if len(photo) > MAX_PHOTO_BYTES:
            raise HTTPException(status_code=413, detail="That photo is too large. Please use one under 10 MB.")
        try:
            image = await asyncio.to_thread(studio.dress, photo, body.mime_type, products, owned)
        except TryOnError as error:
            raise HTTPException(status_code=422, detail=str(error))
        return Response(content=image, media_type="image/png")

    @app.post("/api/wardrobe")
    async def add_to_wardrobe(body: WardrobePhotoRequest, shopper: Shopper = Depends(current_shopper)) -> dict:
        try:
            await chat.ensure_session(shopper.id, body.session_id)
            photo = base64.b64decode(body.photo, validate=True)
        except LookupError:
            raise HTTPException(status_code=404, detail="Conversation not found. Please start again.")
        except binascii.Error:
            raise HTTPException(status_code=400, detail="That photo couldn't be read.")
        if len(photo) > MAX_PHOTO_BYTES:
            raise HTTPException(status_code=413, detail="That photo is too large. Please use one under 10 MB.")
        try:
            items = await asyncio.to_thread(tagger.from_photo, photo, body.mime_type, body.thumbnail or None)
        except Exception:
            logger.exception("Could not read a wardrobe photo for %s", shopper.id)
            items = []
        if not items:
            raise HTTPException(status_code=422, detail="I couldn't make out any clothes in that photo. "
                                                        "Could you try a clearer one?")
        wardrobe = await chat.add_to_wardrobe(shopper.id, body.session_id, items)
        return {"wardrobe": [i.model_dump() for i in wardrobe], "added": [i.model_dump() for i in items]}

    @app.delete("/api/wardrobe/{item_id}")
    async def remove_from_wardrobe(item_id: str, session_id: str,
                                   shopper: Shopper = Depends(current_shopper)) -> list[dict]:
        try:
            wardrobe = await chat.remove_from_wardrobe(shopper.id, session_id, item_id)
        except LookupError:
            raise HTTPException(status_code=404, detail="Conversation not found. Please start again.")
        return [i.model_dump() for i in wardrobe]

    if FRONTEND_DIST.exists():               # the built React app, served from the same Cloud Run service
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

    return app


app = create_app()
