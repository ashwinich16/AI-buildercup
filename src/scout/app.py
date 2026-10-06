"""
Composition root: the ONE place that creates concrete classes and wires them.

Every agent and tool receives its dependencies from here. To swap Vertex AI
Search for another catalog, change only `build_catalog`.
"""

from google.adk.agents import BaseAgent

from scout.agents.comparison import ComparisonAgentBuilder
from scout.agents.elicitation import ElicitationAgentBuilder
from scout.agents.narration import NarrationAgentBuilder
from scout.agents.orchestrator import ScoutOrchestrator
from scout.agents.preference import PreferenceAgentBuilder
from scout.agents.product_expert import ProductExpertAgentBuilder
from scout.catalog import CatalogRepository
from scout.config import Settings, get_settings
from scout.tools import CatalogOverviewBuilder, ImageFetcher, ProductReferenceResolver, ProductRetriever


def build_catalog(settings: Settings) -> CatalogRepository:
    from scout.catalog.vertex_search import VertexSearchCatalog
    return VertexSearchCatalog(settings)


def build_root_agent(settings: Settings | None = None, catalog: CatalogRepository | None = None) -> BaseAgent:
    settings = settings or get_settings()
    catalog = catalog or build_catalog(settings)
    images = ImageFetcher()                  # shared, so photos downloaded once are reused by every agent
    return ScoutOrchestrator(
        elicitation=ElicitationAgentBuilder(settings, catalog).build(),
        narration=NarrationAgentBuilder(settings).build(),
        product_expert=ProductExpertAgentBuilder(settings, images).build(),
        comparison=ComparisonAgentBuilder(settings, images).build(),
        preference=PreferenceAgentBuilder(settings).build(),
        retriever=ProductRetriever(catalog, limit=settings.search_result_limit),
        overview_builder=CatalogOverviewBuilder(catalog),
        resolver=ProductReferenceResolver(),
    )
