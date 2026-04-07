# Aircraft Analysis Pipeline

A shared-CPACS aircraft analysis pipeline built on four independent
**Model Context Protocol (MCP)** servers. A single CPACS XML file serves as
the central data backbone; each MCP reads its inputs from and writes its
results back into that file, enabling flexible, version-tracked multidisciplinary
analysis.

## Architecture

```
                ┌──────────────┐
                │  CPACS XML   │  ← single source of truth
                │  (versioned) │
                └──────┬───────┘
       ┌───────────────┼───────────────┐───────────────┐
       ▼               ▼               ▼               ▼
  ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐
  │ TiGL    │    │ SU2     │    │ pyCycle  │    │ Mission  │
  │ MCP     │    │ MCP     │    │ MCP      │    │ MCP      │
  │ v0.3.0  │    │ v0.3.0  │    │ v0.3.0   │    │ v0.2.0   │
  └────┬────┘    └────┬────┘    └────┬─────┘    └────┬─────┘
       │              │              │               │
  Geometry       Aerodynamics    Engine Cycle    Mission/Traj
  (wings,        (CL, CD,       (TSFC, Fn,     (fuel burn,
   fuselages,     L/D)           OPR, BPR)      GTOW, range)
   STEP export)
```

## CPACS Versioning

Each time an MCP adapter completes, it commits a new numbered version of the
CPACS XML. This gives a full audit trail:

| Version | Author | Contents |
|---------|--------|----------|
| v0 | `file_load` | Original CPACS as loaded from disk |
| v1 | `tigl-mcp` | + geometry analysis results (wing/fuselage counts, bounding boxes) |
| v2 | `su2-mcp` | + aerodynamic coefficients (CL, CD, L/D) |
| v3 | `pycycle-mcp` | + engine performance (TSFC, thrust, OPR, BPR) |
| v4 | `mission-mcp` | + mission results (fuel burn, GTOW, trajectory) |

All version snapshots are saved as `cpacs_v0.xml`, `cpacs_v1.xml`, etc., so
you can compare or restore any previous state.

## XPath Ownership

Each MCP reads from and writes to designated sections of the CPACS tree:

| MCP | Reads | Writes |
|-----|-------|--------|
| **TiGL** | `.//vehicles/aircraft/model`, `.//vehicles/profiles` | `.//analysisResults/tigl` |
| **SU2** | `.//vehicles/aircraft/model/reference`, `.//analysisResults/tigl` | `.//analysisResults/aero` |
| **pyCycle** | `.//vehicles/engines` | `.//vehicles/engines/engine/analysis/mcpResults` |
| **Mission** | `.//reference`, `.//analysisResults/aero`, `.//mcpResults` | `.//analysisResults/mission` |

No two MCPs write to the same XPath, preventing conflicts.

## Quick Start

### Prerequisites

| Dependency | Install |
|-----------|---------|
| Python >= 3.12 | — |
| SU2_CFD | `~/.local/su2/bin/SU2_CFD` |
| Gmsh | `brew install gmsh` or `pip install gmsh` |
| OpenMDAO + pyCycle | `pip install openmdao==3.36.0 om-pycycle` |
| Aviary (optional) | `pip install aviary==0.9.10 dymos==1.13.1` |

> **Critical**: Aviary requires `openmdao==3.36.0` and `dymos==1.13.1`.
> Newer versions cause unit-compatibility errors.

### Install MCPs

```bash
pip install -e tigl-mcp/
pip install -e su2-mcp/
pip install -e pycycle-mcp/
pip install -e mission-mcp/
pip install -e "mission-mcp/[aviary]"  # optional
```

### Run the Pipeline

```bash
# All four MCPs on D150
./run_pipeline.sh d150

# Specific MCPs only
./run_pipeline.sh d150 --mcps tigl su2

# Custom flight conditions
./run_pipeline.sh d150 --mach 0.85 --aoa 3.0

# Other examples
./run_pipeline.sh canards
./run_pipeline.sh dlrf25
```

### Run Tests & OVS

```bash
./run_pipeline.sh --test       # Pipeline integration tests
./run_pipeline.sh --ovs        # OVS validation suite
./run_pipeline.sh --test-all   # Everything
```

## MCP Repositories

| MCP | GitHub | Version | Description |
|-----|--------|---------|-------------|
| TiGL | [cmudrc/tigl-mcp](https://github.com/cmudrc/tigl-mcp) | 0.3.0 | CPACS geometry parsing, STEP export |
| SU2 | [cmudrc/su2-mcp](https://github.com/cmudrc/su2-mcp) | 0.3.0 | CFD aerodynamic analysis (Euler) |
| pyCycle | [cmudrc/pycycle-mcp](https://github.com/cmudrc/pycycle-mcp) | 0.3.0 | Turbofan engine cycle analysis |
| Mission | [cmudrc/mission-mcp](https://github.com/cmudrc/mission-mcp) | 0.2.0 | Mission analysis (Aviary + NSEG) |

## Directory Structure

```
aircraft-analysis/
├── README.md              ← This file
├── run_pipeline.sh        ← Convenience runner
├── shared_cpacs/          ← CPACSManager + XPathRegistry
│   ├── __init__.py
│   ├── manager.py
│   └── xpath_registry.py
├── ovs/                   ← Output Verification System
│   └── validator.py
├── pipeline/              ← Orchestrator
│   └── shared_cpacs_orchestrator.py
├── examples/              ← Sample CPACS files
│   ├── D150_v30.xml
│   ├── canards.xml
│   └── DLR-F25_simple.xml
└── docs/
    ├── architecture.md
    └── parameters.md
```

## Output Verification System (OVS)

The OVS validates that each MCP's output meets structural and plausibility
requirements:

- **Structural checks**: Required XPaths exist and are properly nested
- **Range checks**: Numerical values fall within physically plausible bounds
  (e.g., CL ∈ [-2, 3], TSFC ∈ [0, 5])
- **Cross-MCP checks**: Later MCPs can verify their inputs from earlier MCPs

OVS runs as a CI check (`.github/workflows/ovs.yml`) on every MCP repository.

## Example Results (D150)

| Domain | Key Results |
|--------|-------------|
| **TiGL** | 5 wings, 2 fuselages, STEP exported |
| **SU2** | CL=0.074, CD=0.021, L/D=3.48, Euler solver |
| **pyCycle** | TSFC=0.885, Fn=26528 N, OPR=30.6, BPR=1.5 |
| **Mission (Aviary)** | Fuel=5812 kg, GTOW=62732 kg, Converged |

## License

[MIT](LICENSE)
