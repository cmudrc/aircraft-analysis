#!/usr/bin/env bash
# ============================================================================
# Shared-CPACS Pipeline Runner (Real Solvers)
# ============================================================================
#
# Quick-start:
#   ./run_pipeline.sh                          # Run D150 with all MCPs
#   ./run_pipeline.sh canards                  # Run canards example
#   ./run_pipeline.sh dlrf25                   # Run DLR-F25 example
#   ./run_pipeline.sh D150_v30.xml             # Run a specific CPACS file
#   ./run_pipeline.sh d150 --mcps tigl su2     # Run only TiGL and SU2
#   ./run_pipeline.sh d150 --mach 0.85         # Override flight conditions
#
# Testing:
#   ./run_pipeline.sh --test                   # Run all pipeline tests
#   ./run_pipeline.sh --ovs                    # Run OVS validation suite
#   ./run_pipeline.sh --test-all               # Run everything (tests + OVS)
#
# Requirements:
#   - SU2_CFD on PATH (installed at ~/.local/su2/bin)
#   - Gmsh on PATH (brew install gmsh or pip install gmsh)
#   - OpenMDAO + pyCycle (pip install openmdao om-pycycle)
#   - Existing STEP/mesh files from prior TiGL MCP exports
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Ensure SU2 is on PATH
SU2_BIN="$HOME/.local/su2/bin"
if [ -d "$SU2_BIN" ] && [[ ":$PATH:" != *":$SU2_BIN:"* ]]; then
    export PATH="$SU2_BIN:$PATH"
fi

# Ensure PYTHONPATH covers all packages
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/tigl-mcp/src:${SCRIPT_DIR}/su2-mcp/src:${SCRIPT_DIR}/pycycle-mcp/src:${SCRIPT_DIR}/mission-mcp/src:${SCRIPT_DIR}/pipeline:${PYTHONPATH:-}"

# ── Handle test modes ──
case "${1:-}" in
    --test)
        echo "Running pipeline integration tests..."
        python -m pytest pipeline/test_shared_pipeline.py -v --tb=short
        exit $?
        ;;
    --ovs)
        echo "Running OVS validation suite..."
        python -m pytest ovs/test_ovs.py -v --tb=short
        exit $?
        ;;
    --test-all)
        echo "Running all tests + OVS..."
        python -m pytest pipeline/test_shared_pipeline.py ovs/test_ovs.py -v --tb=short
        exit $?
        ;;
esac

# ── Resolve CPACS file and associated artifacts ──
INPUT="${1:-d150}"
shift 2>/dev/null || true

case "$INPUT" in
    d150|D150)
        CPACS="$SCRIPT_DIR/D150_v30.xml"
        STEP="$SCRIPT_DIR/pipeline/d150_final/aircraft_fused.step"
        MESH="$SCRIPT_DIR/pipeline/d150_final/aircraft_volume.su2"
        ;;
    canards)
        CPACS="$SCRIPT_DIR/canards.xml"
        STEP="$SCRIPT_DIR/pipeline/canards_run/aircraft_fused.step"
        MESH="$SCRIPT_DIR/pipeline/canards_run/aircraft_volume.su2"
        ;;
    dlrf25|f25)
        CPACS="$SCRIPT_DIR/pipeline/dlr_f25_run/DLR-F25_simple.xml"
        STEP="$SCRIPT_DIR/pipeline/dlr_f25_run/aircraft_simple.step"
        MESH="$SCRIPT_DIR/pipeline/dlr_f25_run/aircraft_volume.su2"
        ;;
    *)
        if [ -f "$INPUT" ]; then
            CPACS="$INPUT"
        elif [ -f "$SCRIPT_DIR/$INPUT" ]; then
            CPACS="$SCRIPT_DIR/$INPUT"
        else
            echo "Error: Cannot find CPACS file '$INPUT'"
            echo "Available shortcuts: d150, canards, dlrf25"
            exit 1
        fi
        STEP=""
        MESH=""
        ;;
esac

echo "CPACS file: $CPACS"

# Build CLI args for step/mesh if they exist
EXTRA_ARGS=()
if [ -n "${STEP:-}" ] && [ -f "$STEP" ]; then
    echo "STEP file:  $STEP"
    EXTRA_ARGS+=(--step "$STEP")
fi
if [ -n "${MESH:-}" ] && [ -f "$MESH" ]; then
    echo "Mesh file:  $MESH"
    EXTRA_ARGS+=(--mesh "$MESH")
fi

python pipeline/shared_cpacs_orchestrator.py "$CPACS" "${EXTRA_ARGS[@]}" "$@"
