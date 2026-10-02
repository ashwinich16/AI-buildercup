"""Catalog layer. Agents import only CatalogRepository and SearchQuery from here.

VertexSearchCatalog is imported directly from scout.catalog.vertex_search by the
code that wires the app together, so this package loads without cloud libraries.
"""

from scout.catalog.product_mapper import FieldMap, ProductMapper
from scout.catalog.repository import CatalogRepository, SearchQuery

__all__ = ["CatalogRepository", "SearchQuery", "FieldMap", "ProductMapper"]
