"""
Try the Elicitation agent on its own in the ADK web UI.

From the repository root:
    PYTHONPATH=src adk web playground --port 8080
Then open the web preview on port 8080 and pick "elicitation".
"""

from scout.agents.elicitation import ElicitationAgentBuilder
from scout.catalog.vertex_search import VertexSearchCatalog
from scout.config import get_settings

settings = get_settings()
root_agent = ElicitationAgentBuilder(settings, VertexSearchCatalog(settings)).build()
