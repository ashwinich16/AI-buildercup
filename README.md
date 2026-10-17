# Scout

**A conversational styling companion who knows your style.**
Live app: https://scout-49799314902.asia-south2.run.app

## Table of contents

- [About](#about)
- [Built with](#built-with)
- [Getting started](#getting-started)
- [How to run](#how-to-run)

## About

Online shopping means endless scrolling, and we still buy clothes that don't suit the occasion or what we
already own. Scout fixes that through conversation, by voice or chat.

- **Everyday shopping:** tell Scout what you need; it asks a few questions and brings a handpicked shortlist,
  answers questions (even about product photos), compares options and quotes real reviews.
- **Occasion outfits, wardrobe first:** Scout plans complete looks from the clothes you own and suggests new
  pieces only for the gaps, within your budget. Change or remove any piece in any look.
- **Knows your style:** learns your colours, fabrics, patterns and styles from what you own, save and buy.
- **Also:** virtual try-on on your own photo, wishlist and compare, cart with a size check, and it picks up
  your last conversation where you left off.

Under the hood: 7 Google ADK agents (an Orchestrator that routes each turn in code, plus Understanding,
Narration, Product Expert, Comparison, Stylist and Preference). Answers come only from products retrieved
from Vertex AI Search and verified review quotes. Learning runs after each reply, so it never slows the
conversation.

## Built with

| Layer | Technology |
|---|---|
| AI & agents | Google ADK, Gemini on Vertex AI, Gemini Image / Vertex AI Virtual Try-On |
| Search & data | Vertex AI Search, Cloud SQL (PostgreSQL), Cloud Storage |
| Voice | Web Speech API (listening), Cloud Text-to-Speech (Scout's voice) |
| App | React + Vite, FastAPI (streamed replies) |
| Platform & security | Cloud Run, Firebase Auth (Google sign-in), Secret Manager, IAM |
| Data | Amazon Reviews 2023 (fashion), enriched with Gemini |

## Getting started

### Prerequisites

- Python 3.11+, Node.js 20+, the Google Cloud CLI (`gcloud`)
- Chrome or Edge (for voice)
- A Google Cloud project with these APIs enabled:

```bash
gcloud services enable aiplatform.googleapis.com discoveryengine.googleapis.com texttospeech.googleapis.com \
  sqladmin.googleapis.com secretmanager.googleapis.com storage.googleapis.com
```

### Installation

```bash
# 1. Sign in to Google Cloud
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# 2. Configure: copy the example and set at least GOOGLE_CLOUD_PROJECT and SCOUT_SEARCH_ENGINE_ID
cp .env.example .env

# 3. Python packages
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 4. Frontend packages
cd frontend && npm install && cd ..
```

### Catalog (one time)

Build the product catalog and load it into Vertex AI Search (settings come from `.env`):

```bash
python pipeline/ingestion.py && python pipeline/gender.py && python pipeline/product_type.py
python pipeline/enrich.py && python pipeline/review_evidence.py && python pipeline/export.py
```

Then import the exported JSONL into your Vertex AI Search data store (Console → AI Applications → Data Stores
→ Import → Cloud Storage).

## How to run

### Locally

Two terminals from the repository root:

```bash
# Terminal 1: API on http://localhost:8000
PYTHONPATH=src uvicorn scout.api.main:app --reload --port 8000
# Windows: $env:PYTHONPATH="src"; uvicorn scout.api.main:app --reload --port 8000

# Terminal 2: web app on http://localhost:5173
cd frontend && npm run dev
```

Open http://localhost:5173, sign in, allow the microphone and start talking (or type).

### Deploy to Cloud Run

One service runs the API and the built web app. In PowerShell, with a service account `scout-run` that has
Vertex AI user, Discovery Engine viewer, Cloud SQL client and Secret Manager accessor roles:

```powershell
$SA = "scout-run@YOUR_PROJECT_ID.iam.gserviceaccount.com"
$envs = (Get-Content .env | Where-Object { $_ -match '^[A-Z_]+=.+' }) -join ','
gcloud run deploy scout --source . --region asia-south2 --service-account $SA `
  --set-env-vars $envs --allow-unauthenticated --no-cpu-throttling `
  --session-affinity --min-instances 1 --max-instances 1 --memory 2Gi --cpu 1 --timeout 600
```

With Google sign-in, add the `run.app` address in Firebase console → Authentication → Settings → Authorized
domains.