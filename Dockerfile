# One Cloud Run service: FastAPI serves the API and the built React app.

# 1. Build the React app
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# 2. Python runtime
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY --from=web /web/dist ./frontend/dist

# Cloud Run sets PORT (8080 by default)
CMD ["sh", "-c", "uvicorn scout.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
