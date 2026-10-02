"""
Who is the shopper? The backend decides from the request, never trusting a
user id or name typed into a message.

Two interchangeable implementations (switch with SCOUT_AUTH_MODE):
  demo      - the credential is a persona id; only ids from the demo persona
              list are accepted (hackathon demos)
  firebase  - the credential is a Firebase ID token (Sign in with Google); it is
              verified with firebase-admin, and the Firebase uid becomes the user id

Each authenticator also tells the frontend how to sign in (`client_config`),
so the same React build works in either mode.
"""

from abc import ABC, abstractmethod

from fastapi import HTTPException, Request
from pydantic import BaseModel

from scout.api.personas import Persona

from dotenv import load_dotenv

load_dotenv()


class Shopper(BaseModel):
    """The signed-in shopper, as verified by the backend."""
    id: str
    name: str = ""

    @property
    def first_name(self) -> str:
        """What Scout calls the shopper, the way a friend would."""
        return self.name.split()[0] if self.name.strip() else ""


class Authenticator(ABC):
    async def shopper(self, request: Request) -> Shopper:
        """The verified shopper for an HTTP request, or HTTP 401."""
        return await self.verify(self.credential_from(request))

    @abstractmethod
    def credential_from(self, request: Request) -> str:
        """Where this kind of credential lives in an HTTP request."""

    @abstractmethod
    async def verify(self, credential: str) -> Shopper:
        """The verified shopper for a credential, or HTTP 401."""

    @abstractmethod
    def client_config(self) -> dict:
        """What the frontend needs to sign the shopper in (never secrets)."""


class DemoPersonaAuthenticator(Authenticator):
    HEADER = "X-Scout-User"

    def __init__(self, personas: dict[str, Persona]):
        self._personas = personas

    def credential_from(self, request: Request) -> str:
        return request.headers.get(self.HEADER, "")

    async def verify(self, credential: str) -> Shopper:
        persona = self._personas.get((credential or "").strip())
        if persona is None:
            raise HTTPException(status_code=401, detail="Pick a demo shopper first.")
        return Shopper(id=persona.id, name=persona.name)

    def client_config(self) -> dict:
        return {
            "mode": "demo",
            "personas": [{"id": p.id, "name": p.name, "description": p.description}
                         for p in self._personas.values()],
        }


class FirebaseAuthenticator(Authenticator):
    def __init__(self, project_id: str, web_api_key: str, auth_domain: str):
        import firebase_admin                      # only needed in firebase mode
        from firebase_admin import auth
        if not firebase_admin._apps:
            firebase_admin.initialize_app(options={"projectId": project_id})
        self._auth = auth
        # Firebase's web settings are public by design (they identify the project, they grant nothing).
        self._web_config = {"apiKey": web_api_key, "authDomain": auth_domain, "projectId": project_id}

    def credential_from(self, request: Request) -> str:
        return request.headers.get("Authorization", "").removeprefix("Bearer ").strip()

    async def verify(self, credential: str) -> Shopper:
        if not credential:
            raise HTTPException(status_code=401, detail="Please sign in.")
        try:
            claims = self._auth.verify_id_token(credential)
        except Exception:
            raise HTTPException(status_code=401, detail="Your sign-in has expired. Please sign in again.")
        return Shopper(id=claims["uid"], name=claims.get("name", ""))   # name from the Google account

    def client_config(self) -> dict:
        return {"mode": "firebase", "firebase": self._web_config}
