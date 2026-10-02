# Scout

A voice-first shopping companion for blind and low-vision shoppers, built on Google Cloud
(Gemini on Vertex AI, Google ADK, Vertex AI Search, Cloud Text-to-Speech, Cloud SQL, Cloud Run).

Scout talks with the shopper like a friend: it asks a few questions to understand the need, searches the
catalog, describes options vividly, answers questions about a product (including what its photos show),
compares products, keeps a cart, and remembers lasting preferences. The conversation is hands-free:
the browser listens, Google Cloud Text-to-Speech speaks, and short chimes signal whose turn it is.

## Repository layout

```
conversationalAgent/
├── src/scout/
│   ├── config.py, app.py        settings and the composition root (wires every agent)
│   ├── domain/                  Product, ShoppingNeed, PreferenceProfile
│   ├── catalog/                 CatalogRepository + Vertex AI Search implementation
│   ├── tools/                   retrieval, catalog overview, cart, images, product references
│   ├── agents/                  orchestrator, elicitation, narration, product_expert, comparison, preference
│   └── api/                     FastAPI web API: auth, chat sessions, speech, demo personas
├── frontend/                    React app (Vite): accessible voice-first UI
│   └── src/
│       ├── hooks/useVoice.js    listening (browser speech recognition) and speaking (Cloud TTS)
│       ├── audio/earcons.js     the "your turn" / "thinking" / "paused" chimes
│       └── components/          persona picker, header settings, conversation, products, cart
├── pipeline/                    offline data jobs: enrich.py, export.py
├── playground/                  `adk web` entry points: elicitation/, scout_app/
├── scripts/                     smoke tests against real cloud services
├── Dockerfile                   one Cloud Run service (API + built React app)
├── requirements.txt, pyproject.toml, .env.example
```

## Prerequisites

- Python 3.11 or newer, Node.js 20 or newer, the Google Cloud CLI (`gcloud`)
- A Google Cloud project with these APIs enabled:

  ```bash
  gcloud services enable aiplatform.googleapis.com discoveryengine.googleapis.com \
    texttospeech.googleapis.com sqladmin.googleapis.com secretmanager.googleapis.com storage.googleapis.com
  ```

- The catalog already exported and imported into a Vertex AI Search structured data store
  (see [Data pipeline](#data-pipeline))
- Chrome or Edge for voice (they provide the browser speech recognition Scout listens with)

## One-time setup

**1. Sign in to Google Cloud** (your local credentials are used by Gemini, Search and Text-to-Speech):

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

**2. Configure:** copy `.env.example` to `.env` and fill in your values. At minimum set
`GOOGLE_CLOUD_PROJECT` and either `SCOUT_SEARCH_ENGINE_ID` or `SCOUT_SEARCH_DATA_STORE_ID`.

```bash
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
```

**3. Python environment:**

```bash
python -m venv .venv
source .venv/bin/activate     # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**4. Frontend packages:**

```bash
cd frontend
npm install
cd ..
```

## Run the app (development)

Run the API and the React app in two terminals, both from the repository root with the virtual
environment active.

**Terminal 1 – API on http://localhost:8000**

```bash
# macOS / Linux
PYTHONPATH=src uvicorn scout.api.main:app --reload --port 8000
```

```powershell
# Windows PowerShell
$env:PYTHONPATH="src"; uvicorn scout.api.main:app --reload --port 8000
```

**Terminal 2 – React app on http://localhost:5173**

```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** in Chrome or Edge, pick a demo shopper, and allow the microphone.
Scout greets you and then listens continuously; speak after the rising chime. Speaking while Scout
talks interrupts it, Escape silences it, and the Talk button pauses or resumes listening. You can
also type. Headphones give the best results, because speakers let the microphone hear Scout.

In the header settings you can choose the **listening language/accent** (English India, US, UK,
Hindi, Telugu, Tamil), speech speed, high contrast and larger text, or switch shopper.

## Test the agents without the UI (ADK web)

```bash
PYTHONPATH=src adk web playground --port 8080                    # macOS / Linux
```

```powershell
$env:PYTHONPATH="src"; adk web playground --port 8080            # Windows PowerShell
```

Open http://localhost:8080 and choose `scout_app` (the full Scout conversation) or `elicitation`
(only the need-understanding agent).

## Smoke tests

```bash
# Can we reach Vertex AI Search, and what fields do the documents have?
PYTHONPATH=src python scripts/check_vertex_search.py "warm jacket for winter"
PYTHONPATH=src python scripts/check_vertex_search.py "office shoes" --filter 'category: ANY("shoe")'

# Does retrieval (search + relaxation) find sensible products for sample needs?
PYTHONPATH=src python scripts/check_retrieval.py
```

On Windows PowerShell run `$env:PYTHONPATH="src"` once in the terminal, then the `python ...` commands.

## Data pipeline

Run once (or whenever the catalog changes). All settings come from `.env`.

```bash
python pipeline/enrich.py     # Gemini reads photos, descriptions and reviews -> product_intelligence in Cloud SQL
python pipeline/export.py     # Cloud SQL -> JSONL in gs://$SCOUT_EXPORT_BUCKET/$SCOUT_EXPORT_OBJECT
```

Then import the JSONL into Vertex AI Search: Console → AI Applications → Data Stores → your
structured data store → Import → Cloud Storage → the exported file (choose a full/replace import
when re-importing). Make sure `title` and `search_text` are searchable, and `category` and `price`
are filterable, in the data store schema.

`enrich.py` only processes products that have no `product_intelligence` row yet; to redo everything,
run `DELETE FROM product_intelligence;` first.

## Key environment variables

| Variable | Purpose |
|---|---|
| `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_GENAI_USE_VERTEXAI` | Gemini on Vertex AI |
| `SCOUT_AGENT_MODEL` | default model for every agent |
| `SCOUT_<AGENT>_MODEL / _TEMPERATURE / _THINKING_BUDGET` | per-agent tuning (elicitation, narration, product_expert, comparison, preference) |
| `SCOUT_SEARCH_LOCATION`, `SCOUT_SEARCH_ENGINE_ID` or `SCOUT_SEARCH_DATA_STORE_ID` | Vertex AI Search |
| `SCOUT_AUTH_MODE` | `demo` (persona picker) or `firebase` (real sign-in) |
| `SCOUT_TTS_VOICE` | Cloud Text-to-Speech voice |
| `SCOUT_DB_INSTANCE`, `SCOUT_DB_NAME`, `SCOUT_DB_USER`, `SCOUT_DB_PASSWORD_SECRET` | Cloud SQL: the pipeline, and Scout's memory of each shopper (tables `shopper`, `user_preference`, `cart_item`, created automatically). Empty instance = memory only, lost on restart |
| `SCOUT_EXPORT_*` | Cloud Storage, used by the pipeline |

## Production build (single server)

FastAPI serves the built React app, so one process (or one Cloud Run service) runs everything:

```bash
cd frontend && npm run build && cd ..
PYTHONPATH=src uvicorn scout.api.main:app --port 8000      # then open http://localhost:8000
```

## Deploy to Cloud Run

```bash
gcloud run deploy scout --source . --region us-central1 \
  --set-env-vars "$(grep -v '^#' .env | grep -v '^$' | paste -sd, -)" \
  --session-affinity
```

The Dockerfile builds the React app and starts the API, so no separate frontend deploy is needed.
Names, preferences and carts are kept in Cloud SQL (`SCOUT_DB_*`), so they survive restarts and are shared by
every instance. The current conversation lives in the instance's memory, so `--session-affinity` keeps a
shopper on the same instance. The service account needs: Vertex AI User, Discovery Engine Viewer,
Cloud Text-to-Speech access, Cloud SQL Client, Secret Manager Secret Accessor.

## Troubleshooting

| Symptom | What to do |
|---|---|
| Code changes don't show up | Restart both servers; delete `__pycache__` folders if `adk web` shows old behaviour. |
| A file reverted to an old version | Your editor saved an old open copy over it: close or revert open files before saving. |
| Demo shoppers don't appear | Check the API terminal for an error; make sure the API runs on port 8000. |
| Scout doesn't hear you | Use Chrome or Edge, allow the microphone, and check the listening language in settings. |
| Scout mishears words | Pick the listening language that matches your accent; Scout also corrects or checks unclear words. |
| Scout stops talking by itself | The microphone is hearing Scout through the speakers: use headphones or lower the volume. |
| Search errors | Check `SCOUT_SEARCH_*` in `.env` and run `scripts/check_vertex_search.py`. |
| `npm run dev` warns about Vite | `@vitejs/plugin-react` 4 supports Vite up to 7: run `npm install vite@^7` in `frontend`. |
| Permission / credential errors | Run `gcloud auth application-default login` again and check the project in `.env`. |
