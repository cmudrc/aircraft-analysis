"""XPath registry defining which CPACS sections each MCP owns.

Each MCP domain declares:
- *reads*: XPaths it consumes as input
- *writes*: XPaths it populates with results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MCPDomain(str, Enum):
    TIGL = "tigl"
    SU2 = "su2"
    PYCYCLE = "pycycle"
    MISSION = "mission"


@dataclass(frozen=True)
class DomainPaths:
    """Read and write XPath sets for a single MCP domain."""

    reads: tuple[str, ...]
    writes: tuple[str, ...]


_REGISTRY: dict[MCPDomain, DomainPaths] = {
    MCPDomain.TIGL: DomainPaths(
        reads=(
            ".//vehicles/aircraft/model",
            ".//vehicles/profiles",
        ),
        writes=(
            ".//vehicles/aircraft/model/analysisResults/tigl",
        ),
    ),
    MCPDomain.SU2: DomainPaths(
        reads=(
            ".//vehicles/aircraft/model/reference",
            ".//vehicles/aircraft/model/analysisResults/tigl",
        ),
        writes=(
            ".//vehicles/aircraft/model/analysisResults/aero",
        ),
    ),
    MCPDomain.PYCYCLE: DomainPaths(
        reads=(
            ".//vehicles/engines",
        ),
        writes=(
            ".//vehicles/engines/engine/analysis/mcpResults",
        ),
    ),
    MCPDomain.MISSION: DomainPaths(
        reads=(
            ".//vehicles/aircraft/model/reference",
            ".//vehicles/aircraft/model/analysisResults/aero",
            ".//vehicles/engines/engine/analysis/mcpResults",
        ),
        writes=(
            ".//vehicles/aircraft/model/analysisResults/mission",
        ),
    ),
}


class XPathRegistry:
    """Query the domain-to-XPath mapping."""

    @staticmethod
    def get(domain: MCPDomain) -> DomainPaths:
        return _REGISTRY[domain]

    @staticmethod
    def reads(domain: MCPDomain) -> tuple[str, ...]:
        return _REGISTRY[domain].reads

    @staticmethod
    def writes(domain: MCPDomain) -> tuple[str, ...]:
        return _REGISTRY[domain].writes

    @staticmethod
    def all_domains() -> list[MCPDomain]:
        return list(_REGISTRY.keys())

    @staticmethod
    def write_conflict_check() -> dict[str, list[MCPDomain]]:
        """Find XPaths written by more than one domain."""
        path_owners: dict[str, list[MCPDomain]] = {}
        for domain, paths in _REGISTRY.items():
            for wp in paths.writes:
                path_owners.setdefault(wp, []).append(domain)
        return {p: owners for p, owners in path_owners.items() if len(owners) > 1}
