"""
CatalogRepository backed by Vertex AI Search.

Responsibilities (and nothing else):
  - build Vertex AI Search requests from a SearchQuery
  - read documents back and hand each one to ProductMapper

It knows only the three filter field names (category, price, rating) through
FilterFields, so a different catalog just passes different names. If the data
store schema does not mark those fields filterable, it falls back to filtering
the results locally instead of failing.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InvalidArgument, NotFound
from google.cloud import discoveryengine_v1 as discoveryengine
from google.protobuf.json_format import MessageToDict

from scout.catalog.product_mapper import ProductMapper
from scout.catalog.repository import CatalogRepository, SearchQuery
from scout.config import MissingSettingError, Settings
from scout.domain import Product

logger = logging.getLogger(__name__)

_Request = discoveryengine.SearchRequest


@dataclass(frozen=True)
class FilterFields:
    """Names of the filterable fields in the Vertex AI Search schema."""

    category: str = "category"
    price: str = "price"
    rating: str = "average_rating"


class VertexSearchCatalog(CatalogRepository):
    def __init__(self, settings: Settings,
                 mapper: Optional[ProductMapper] = None,
                 filter_fields: Optional[FilterFields] = None):
        self._settings = settings
        self._mapper = mapper or ProductMapper()
        self._fields = filter_fields or FilterFields()
        self._search_client = discoveryengine.SearchServiceClient(client_options=self._client_options())
        self._server_filtering = True        # switched off once if the schema rejects our filters
        self._categories: Optional[list[str]] = None

    # -----------------------------------------------------------------
    # CatalogRepository
    # -----------------------------------------------------------------

    def search(self, query: SearchQuery) -> list[Product]:
        if self._server_filtering:
            try:
                return self._run_search(query.text, self.build_filter(query), query.limit)
            except InvalidArgument as error:
                if not self.build_filter(query):
                    raise
                logger.warning("Filter rejected by Vertex AI Search (%s). "
                               "Filtering locally from now on.", error.message)
                self._server_filtering = False

        candidates = self._run_search(query.text, "", min(query.limit * 4, 50))
        return [p for p in candidates if query.matches(p)][: query.limit]

    def get_product(self, product_id: str) -> Optional[Product]:
        if not self._settings.search_data_store_id:
            raise MissingSettingError("get_product needs SCOUT_SEARCH_DATA_STORE_ID")
        name = (f"projects/{self._settings.project_id}/locations/{self._settings.search_location}"
                f"/collections/default_collection/dataStores/{self._settings.search_data_store_id}"
                f"/branches/default_branch/documents/{product_id}")
        client = discoveryengine.DocumentServiceClient(client_options=self._client_options())
        try:
            return self._to_product(client.get_document(name=name))
        except NotFound:
            return None

    def categories(self) -> list[str]:
        if self._categories is None:        # the catalog rarely changes; ask once per process
            self._categories = self._categories_from_facets() or self._categories_from_documents()
        return self._categories

    # -----------------------------------------------------------------
    # Request building (public so it can be unit-tested without the cloud)
    # -----------------------------------------------------------------

    def build_filter(self, query: SearchQuery) -> str:
        """e.g.  category: ANY("shoe") AND price <= 80.0 AND average_rating >= 4.0"""
        clauses = []
        if query.category:
            clauses.append(f'{self._fields.category}: ANY("{_escape(query.category)}")')
        if query.min_price is not None:
            clauses.append(f"{self._fields.price} >= {query.min_price}")
        if query.max_price is not None:
            clauses.append(f"{self._fields.price} <= {query.max_price}")
        if query.min_rating is not None:
            clauses.append(f"{self._fields.rating} >= {query.min_rating}")
        return " AND ".join(clauses)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _run_search(self, text: str, filter_expr: str, limit: int) -> list[Product]:
        request = _Request(
            serving_config=self._settings.search_serving_config,
            query=text,
            filter=filter_expr,
            page_size=limit,
            query_expansion_spec=_Request.QueryExpansionSpec(
                condition=_Request.QueryExpansionSpec.Condition.AUTO),
            spell_correction_spec=_Request.SpellCorrectionSpec(
                mode=_Request.SpellCorrectionSpec.Mode.AUTO),
        )
        products = []
        for result in self._search_client.search(request):
            products.append(self._to_product(result.document))
            if len(products) >= limit:
                break
        return products

    def _categories_from_facets(self) -> list[str]:
        request = _Request(
            serving_config=self._settings.search_serving_config,
            query="",
            page_size=1,
            facet_specs=[_Request.FacetSpec(
                facet_key=_Request.FacetSpec.FacetKey(key=self._fields.category), limit=100)],
        )
        try:
            response = self._search_client.search(request)
            return sorted(v.value for facet in response.facets for v in facet.values if v.value)
        except InvalidArgument as error:
            logger.warning("Category facet unavailable (%s). Reading categories from documents.",
                           error.message)
            return []

    def _categories_from_documents(self) -> list[str]:
        return sorted({p.category for p in self._run_search("", "", 50) if p.category})

    def _to_product(self, document) -> Product:
        data = MessageToDict(document._pb)
        return self._mapper.to_product(data.get("structData", {}), fallback_id=data.get("id", ""))

    def _client_options(self) -> Optional[ClientOptions]:
        location = self._settings.search_location
        if location == "global":
            return None
        return ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')
