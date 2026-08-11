"""Shared CPACS versioned management system.

Central CPACS file infrastructure for the MCP ecosystem.  Each MCP reads
its inputs from and writes its outputs back into a single CPACS XML,
tracked with per-write version history.
"""

from shared_cpacs.manager import CPACSManager, CPACSVersion
from shared_cpacs.xpath_registry import MCPDomain, XPathRegistry

__all__ = [
    "CPACSManager",
    "CPACSVersion",
    "MCPDomain",
    "XPathRegistry",
]
