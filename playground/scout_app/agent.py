"""
The full Scout agent (Orchestrator + everything built so far) in the ADK web UI.

This folder must NOT be named "scout": adk web puts playground/ on the import
path, and a folder called "scout" would hide the real scout package in src/.

From the repository root (PowerShell):
    $env:PYTHONPATH = "src"
    adk web playground --port 8080
Then open http://localhost:8080 and pick "scout_app".
"""

from scout.app import build_root_agent

root_agent = build_root_agent()
