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

Usage:
  python scripts/check_vertex_search.py "warm jacket for winter"
  python scripts/check_vertex_search.py "office shoes" --filter 'category: ANY("shoe")'
"""

import argparse
import json
import os
import sys

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import discoveryengine_v1 as discoveryengine
from google.protobuf.json_format import MessageToDict
from dotenv import load_dotenv

load_dotenv()


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"Missing environment variable: {name}")
    return value


def build_serving_config(project: str, location: str) -> str:
    base = f"projects/{project}/locations/{location}/collections/default_collection"
    engine_id = os.environ.get("SCOUT_SEARCH_ENGINE_ID", "").strip()
    if engine_id:
        return f"{base}/engines/{engine_id}/servingConfigs/default_search"
    data_store_id = require_env("SCOUT_SEARCH_DATA_STORE_ID")
    return f"{base}/dataStores/{data_store_id}/servingConfigs/default_search"


def build_client(location: str) -> discoveryengine.SearchServiceClient:
    if location == "global":
        return discoveryengine.SearchServiceClient()
    endpoint = f"{location}-discoveryengine.googleapis.com"
    return discoveryengine.SearchServiceClient(client_options=ClientOptions(api_endpoint=endpoint))


def search(query: str, filter_expr: str, limit: int) -> list[dict]:
    project = require_env("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("SCOUT_SEARCH_LOCATION", "global").strip()

    request = discoveryengine.SearchRequest(
        serving_config=build_serving_config(project, location),
        query=query,
        filter=filter_expr,
        page_size=limit,
    )
    pager = build_client(location).search(request)

    documents = []
    for result in pager:
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
    except GoogleAPICallError as error:
        sys.exit(f"Search call failed: {error.message}")


if __name__ == "__main__":
    main()
