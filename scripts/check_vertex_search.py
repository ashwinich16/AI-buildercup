"""
Smoke test for the Scout catalog in Vertex AI Search.

Checks three things before we build any agent on top of search:
  1. We can authenticate and reach the data store / search app.
  2. A natural-language query returns products.
  3. A filter (e.g. by category) works, and which fields each document has.

It prints whatever fields the documents contain, without assuming a schema,
so it keeps working if the catalog changes (fashion today, electronics later).

Configuration comes from environment variables (see .env.example):
  GOOGLE_CLOUD_PROJECT         your GCP project id
  SCOUT_SEARCH_LOCATION        "global" unless you created the store in a region
  SCOUT_SEARCH_ENGINE_ID       search app (engine) id  -- preferred, or
  SCOUT_SEARCH_DATA_STORE_ID   data store id           -- used if no engine id

Usage (from the repository root, with PYTHONPATH=src):
  python scripts/check_vertex_search.py "formal shirt for the office"
  python scripts/check_vertex_search.py "office shoes" --filter 'category: ANY("shoe")'
"""

import argparse
import json
import sys

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import discoveryengine_v1 as discoveryengine
from google.protobuf.json_format import MessageToDict

from scout.config import MissingSettingError, get_settings


def search(query: str, filter_expr: str, limit: int) -> list[dict]:
    """Raw documents (no mapping into Product), so every stored field is visible."""
    settings = get_settings()                 # the same serving config the app uses
    location = settings.search_location
    options = None if location == "global" else ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
    request = discoveryengine.SearchRequest(
        serving_config=settings.search_serving_config, query=query, filter=filter_expr, page_size=limit)

    documents = []
    for result in discoveryengine.SearchServiceClient(client_options=options).search(request):
        doc = MessageToDict(result.document._pb)
        documents.append({"id": doc.get("id"), "fields": doc.get("structData", {})})
        if len(documents) >= limit:
            break
    return documents


def short(value, max_chars: int = 90) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def print_results(documents: list[dict]) -> None:
    if not documents:
        print("No results. Check the query, the filter, and that the import finished.")
        return

    print(f"\n{len(documents)} result(s):\n")
    for rank, doc in enumerate(documents, start=1):
        fields = doc["fields"]
        print(f"{rank}. {fields.get('title', doc['id'])}")
        for key, value in fields.items():
            if key != "title" and value not in (None, "", [], {}):
                print(f"     {key}: {short(value)}")
        print()

    all_fields = sorted({key for doc in documents for key in doc["fields"]})
    print("Fields seen across results:", ", ".join(all_fields))


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Scout catalog in Vertex AI Search.")
    parser.add_argument("query", help="natural-language search text")
    parser.add_argument("--filter", default="", help='e.g. category: ANY("shoe")')
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    try:
        print_results(search(args.query, args.filter, args.limit))
    except MissingSettingError as error:
        sys.exit(str(error))
    except GoogleAPICallError as error:
        sys.exit(f"Search call failed: {error.message}")


if __name__ == "__main__":
    main()
