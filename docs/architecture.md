# Architecture

## Shared-CPACS Design

The pipeline uses a **single CPACS XML file** as the central data store.
Each MCP server operates independently — reading its inputs from and
writing its results back into designated sections of the XML tree.

### Benefits

1. **Independent operation**: MCPs can run in any order or individually
2. **Version tracking**: Every MCP commit creates a numbered snapshot
3. **No coupling**: MCPs don't import each other's code
4. **Reproducibility**: Full audit trail from input to final output
5. **Flexibility**: Add/remove/reorder MCPs without code changes

### CPACSManager

The `CPACSManager` class (`shared_cpacs/manager.py`) provides:

- **Thread-safe** access to the CPACS XML tree
- **XPath-based** read/write operations
- **Version snapshots** on every `commit()` call
- **Restore** any previous version by ID
- **Save** versioned snapshots to disk

### XPathRegistry

The `XPathRegistry` (`shared_cpacs/xpath_registry.py`) declares which
XPaths each MCP owns. This enforces separation of concerns and enables
conflict detection.

## Data Flow

```
CPACS v0 (loaded from disk)
    │
    ▼  TiGL adapter reads geometry, writes to //analysisResults/tigl
CPACS v1
    │
    ▼  SU2 adapter reads reference + TiGL results, writes to //analysisResults/aero
CPACS v2
    │
    ▼  pyCycle adapter reads engines + drag, writes to //mcpResults
CPACS v3
    │
    ▼  Mission adapter reads aero + engine + geometry, writes to //analysisResults/mission
CPACS v4 (final)
```

## Intermediate Artifacts

Some MCPs produce non-XML artifacts that are passed forward:

| Artifact | Producer | Consumer | Format |
|----------|----------|----------|--------|
| STEP geometry | TiGL | SU2 | `.step` file or bytes |
| SU2 mesh | SU2 (via Gmsh) | — | `.su2` mesh file |
| VTU flow field | SU2 | — (visualization) | `.vtu` ParaView file |

These are managed via the `shared_artifacts` dict in the orchestrator.

## Orchestrator

The `shared_cpacs_orchestrator.py` script:

1. Loads the CPACS file into a `CPACSManager`
2. For each MCP in order: imports its adapter, runs it, commits the result
3. Saves all version snapshots and a JSON results summary
4. Prints a human-readable progress report

## Mission MCP Backends

The Mission MCP supports two analysis backends:

### Aviary (Primary)

NASA's open-source aircraft design/analysis tool:
- Gradient-based trajectory optimization (climb/cruise/descent)
- Fuel burn, GTOW, wing mass, reserve fuel
- Detailed timeseries trajectory data
- Requires: `aviary==0.9.10`, `openmdao==3.36.0`, `dymos==1.13.1`

### NSEG (Fallback)

Built-in segment physics engine:
- Breguet range equation per segment
- ISA atmosphere model
- Parabolic drag polar
- No external dependencies
- Automatically selected when Aviary isn't installed or the geometry is
  incompatible (e.g., small/non-transport aircraft)
