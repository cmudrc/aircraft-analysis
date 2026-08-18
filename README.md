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
| v4 | `nseg-mcp` or `aviary-cpacs-mcp` | + mission results (fuel burn, GTOW, trajectory) |

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
| Python >= 3.12 (3.13 OK) | macOS: `brew install python@3.13`. Linux: `apt-get install python3.13 python3.13-venv`. Windows: from https://python.org, or `winget install Python.Python.3.13`. |
| SU2_CFD | Linux/macOS: run `bash su2-mcp/scripts/install_su2.sh` (conda preferred, falls back to binary download). Windows: install via WSL2 (`wsl --install`), then run the same script inside WSL. |
| Gmsh | `brew install gmsh` (macOS), `apt-get install gmsh` (Linux), or `pip install gmsh` (all platforms; bundles a private binary). |
| OpenMDAO + pyCycle | `pip install openmdao==3.36.0 om-pycycle` |
| Aviary (optional) | `pip install aviary==0.9.10 dymos==1.13.1` |
| Ollama (for the agent layer) | macOS: `brew install ollama`. Linux: `curl -fsSL https://ollama.com/install.sh \| sh`. Windows: `winget install Ollama.Ollama`. |

> **Critical**: Aviary requires `openmdao==3.36.0` and `dymos==1.13.1`.
> Newer versions cause unit-compatibility errors.

> **One-command shortcut**: run `bash bootstrap.sh` (POSIX) or
> `pwsh bootstrap.ps1` (Windows) at the project root and the script
> will install everything in the table above, pull the Gemma model,
> and launch the agent. See [`cmudrc/agent-mcp`](https://github.com/cmudrc/agent-mcp).

### Install MCPs

```bash
pip install -e tigl-mcp/
pip install -e su2-mcp/
pip install -e pycycle-mcp/
pip install -e nseg-mcp/
pip install -e aviary-cpacs-mcp/   # optional, trajectory-level missions
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
```

### Run Tests

```bash
./run_pipeline.sh --test       # Skill-harness tests
```

## Deterministic skill harnesses (`scripts/`)

The agent's *iterative skills* — judgment loops that call a tool repeatedly
until a numerical condition is met — each ship a no-LLM Python harness so they
can be reproduced (and unit-tested) without an agent. They wrap the **real**
SU2 / pyCycle / NSEG adapters; a missing solver is a loud, structured error,
never a fabricated result.

| Harness | What it converges | Disciplines |
|---------|-------------------|-------------|
| [`scripts/run_converged_su2.py`](scripts/run_converged_su2.py) | mesh density until CL/CD plateau | SU2 |
| [`scripts/run_aoa_sweep.py`](scripts/run_aoa_sweep.py) | best-L/D + trim angle for a target CL | SU2 |
| [`scripts/run_engine_resize.py`](scripts/run_engine_resize.py) | smallest engine that closes the mission at top of climb | pyCycle ↔ NSEG |
| [`scripts/run_cruise_match.py`](scripts/run_cruise_match.py) | cruise point where thrust = drag, with weight/fuel closure | SU2 ↔ pyCycle ↔ NSEG |

```bash
# Example: size the engine so a 3000 km mission closes with a 5% climb margin
python scripts/run_engine_resize.py --cpacs examples/D150_v30.xml \
  --mach 0.78 --altitude 35000 --weight 70000 --range-km 3000 \
  --target-margin-frac 0.05
```

Unit tests for the loop logic live in `scripts/tests/` and run on monkeypatched
adapters (`pytest scripts/tests`). The agent-side specs are in
[`cmudrc/agent-mcp`](https://github.com/cmudrc/agent-mcp) under `skills/`.

## MCP Repositories

| MCP | GitHub | Version | Description |
|-----|--------|---------|-------------|
| TiGL | [cmudrc/tigl-mcp](https://github.com/cmudrc/tigl-mcp) | 0.3.0 | CPACS geometry parsing, STEP export |
| SU2 | [cmudrc/su2-mcp](https://github.com/cmudrc/su2-mcp) | 0.3.0 | CFD aerodynamic analysis (Euler) |
| pyCycle | [cmudrc/pycycle-mcp](https://github.com/cmudrc/pycycle-mcp) | 0.3.0 | Turbofan engine cycle analysis |
| Mission (segment/Breguet) | [cmudrc/nseg-mcp](https://github.com/cmudrc/nseg-mcp) | Mission analysis, low fidelity |
| Mission (trajectory) | [cmudrc/aviary-cpacs-mcp](https://github.com/cmudrc/aviary-cpacs-mcp) | Mission analysis via NASA Aviary |

## Directory Structure

```
aircraft-analysis/
├── README.md              ← This file
├── run_pipeline.sh        ← Convenience runner
├── shared_cpacs/          ← CPACSManager + XPathRegistry
│   ├── __init__.py
│   ├── manager.py
│   └── xpath_registry.py
├── pipeline/              ← Orchestrator
│   └── shared_cpacs_orchestrator.py
├── scripts/               ← Deterministic skill harnesses (+ tests/)
│   ├── run_converged_su2.py
│   ├── run_aoa_sweep.py
│   ├── run_engine_resize.py
│   └── run_cruise_match.py
├── examples/              ← Sample CPACS files
│   ├── D150_v30.xml
│   └── canards.xml
└── docs/
    ├── architecture.md
    └── parameters.md
```

## Change-control gate (CI)

Every repository runs a GitHub Actions gate on each push to `main` and each
pull request, across Ubuntu, Windows, and macOS on Python 3.12 and 3.13. It
checks that the code is clean and that its declared dependencies resolve:
the package installs from its own manifest, `ruff` lints it, `mypy` type-checks
it in strict mode, and FastMCP smoke-imports confirm the tool surface builds.
Solver-integration tests are gated to Ubuntu, since they need native Linux
toolchains.

This fires on code changes only, never during a pipeline run or an agent
session. Reproducing a *result* is a separate thing, and is what the
deterministic harnesses in `scripts/` are for: they re-derive any reported
number without running an AI.

## Example Results (D150)

| Domain | Key Results |
|--------|-------------|
| **TiGL** | 3 wings, 1 fuselage, STEP exported (3.2 MB, real TiGL) |
| **SU2** | CL=0.074, CD=0.021, L/D=3.48, Euler — *demo point, not cruise* |
| **pyCycle** | TSFC=0.885, Fn=26528 N, OPR=30.6, BPR=1.5 |
| **Mission (Aviary)** | Fuel=5812 kg, GTOW=62732 kg, Converged |

**Read the SU2 row carefully.** `L/D = 3.48` is not a cruise value and should
not be compared with one. It is a coarse-mesh, low-angle-of-attack smoke test
whose purpose is to show the pipeline runs end to end. Two things make it low:
the angle of attack is well below the cruise trim point, so lift is small, and
the mesh is the ~50k-cell preset. On a converged ~2M-cell mesh at
alpha = 2.5 deg the same aircraft gives CL = 0.348 and L/D = 16.4. Euler still
sits below the RANS band of roughly 20-27 for this class, because inviscid flow
has no skin friction.

The counts come from a parser scoped to the aircraft model subtree. An earlier
version searched the whole document and reported 5 wings and 2 fuselages for
the D150, having also matched vendor blocks under `toolspecific/`. If you see
those numbers in older output, that is why.

## License

[MIT](LICENSE)

## Maintainers

Mayank Dixit ([@Kugel-Blitz-13](https://github.com/Kugel-Blitz-13)), Carnegie
Mellon University — mayankd@cmu.edu
Christopher McComb, Carnegie Mellon University — Design Research Collective
