#!/usr/bin/env python3
"""Open-ended SU2 mesh refinement until CL/CD plateau (no LLM needed).

This is the deterministic counterpart to the agent-driven
[`SKILL_OPEN_ENDED_MESH.md`](../agent-mcp/skills/SKILL_OPEN_ENDED_MESH.md)
skill. A third-party user who wants a *converged* SU2 result without
running a local LLM can invoke this script and get exactly the same
refinement loop the agent would have executed:

    python scripts/run_converged_su2.py \\
        --cpacs D150_v30.xml \\
        --step pipeline/d150_final/aircraft_fused.step \\
        --start-density 30 --growth 2.0 \\
        --max-rungs 5 --max-wall-seconds 7200 \\
        --mach 0.78 --aoa 2.0 --altitude 35000

The script wraps the real ``su2_mcp.cpacs_adapter.run_adapter`` call
(via the new ``surface_density`` override added 2026-06-21). No fake CL
or "plateau" guess is ever emitted: when SU2 fails the script prints
the structured adapter error and exits non-zero, per the project's
no-stubs rule.

Output: ``<output_root>/convergence_history.json`` plus a per-rung
sub-directory holding the SU2 working files. The history JSON is the
single artifact the paper / PPT pulls numbers from.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Auto-relaunch under the project's .venv so that gmsh / pyvista / the
# real SU2 binary live on PATH (mirrors gemma_agent.py's behaviour).
_VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"
if (
    _VENV_PY.exists()
    and not os.environ.get("CONVERGED_SU2_RESPAWNED")
    and Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve()
):
    os.environ["CONVERGED_SU2_RESPAWNED"] = "1"
    os.execv(
        str(_VENV_PY),
        [str(_VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

_SU2_BIN = Path.home() / ".local" / "su2" / "bin"
if _SU2_BIN.is_dir() and str(_SU2_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_SU2_BIN}:{os.environ.get('PATH', '')}"

# Make su2-mcp importable from the source tree even before pip-install.
for sub in ("su2-mcp/src",):
    p = PROJECT_ROOT / sub
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Open-ended SU2 mesh refinement: doubles surface_density until "
            "CL and CD plateau within --plateau-tol AND SU2's inner Cauchy "
            "criterion fires. Honours a hard wall-clock + cell-count budget."
        )
    )
    p.add_argument("--cpacs", required=True, help="Path to the CPACS XML file.")
    p.add_argument(
        "--step",
        default=None,
        help=(
            "Path to the STEP geometry to remesh on each rung. Required "
            "unless --mesh is given for the first rung (rare)."
        ),
    )
    p.add_argument(
        "--mesh",
        default=None,
        help=(
            "Path to an existing .su2 mesh to use for the FIRST rung only. "
            "Subsequent rungs always remesh from --step at the new density."
        ),
    )
    p.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "converged_su2_runs"),
        help="Root directory for per-rung output (default: ./converged_su2_runs).",
    )

    # Refinement ladder
    p.add_argument("--start-density", type=int, default=30,
                   help="surface_density for rung 1 (default 30).")
    p.add_argument("--growth", type=float, default=2.0,
                   help="Multiplier applied to surface_density each rung (default 2.0).")
    p.add_argument("--max-rungs", type=int, default=5,
                   help="Maximum number of refinement rungs (default 5).")
    p.add_argument("--max-wall-seconds", type=int, default=7200,
                   help="Total wall-clock budget across rungs (default 7200 s).")
    p.add_argument("--max-n-elem", type=int, default=5_000_000,
                   help="Skip next rung if estimated cells exceed this (default 5e6).")
    p.add_argument("--plateau-tol", type=float, default=0.01,
                   help="Outer plateau tolerance on CL and CD (default 0.01 = 1%%).")

    # Flight condition
    p.add_argument("--mach", type=float, default=0.78)
    p.add_argument("--aoa", type=float, default=2.0)
    p.add_argument("--altitude", type=float, default=35000.0,
                   help="Altitude in feet (default 35000).")

    # SU2 budget per rung
    p.add_argument("--iter-cap", type=int, default=800,
                   help="SU2 inner iteration cap per rung (default 800).")
    p.add_argument("--cl-eps", type=float, default=1e-4,
                   help="Cauchy convergence tolerance on LIFT (default 1e-4).")
    p.add_argument("--per-rung-timeout", type=int, default=7200,
                   help="Per-rung wall timeout for SU2_CFD (default 7200 s).")

    return p.parse_args(argv)


def _next_density(current: int, growth: float) -> int:
    """Compute the next rung's surface_density.

    Always strictly greater than the current value to avoid pathological
    stalls when growth is set very close to 1.0.
    """
    return max(int(round(current * growth)), current + 10)


def _projected_n_elem(history: list[dict[str, Any]], next_density: int) -> int | None:
    """Project the next rung's cell count from the most recent rung.

    Surface elements scale ~quadratically with density; volume cells
    scale ~cubically. We use the cubic estimate (the conservative one
    for memory) when the latest rung returned ``mesh_n_elem``.
    """
    if not history:
        return None
    last = history[-1]
    last_n = last.get("mesh_n_elem")
    last_d = last.get("surface_density")
    if not isinstance(last_n, int) or not isinstance(last_d, int) or last_d <= 0:
        return None
    ratio = next_density / last_d
    return int(last_n * (ratio ** 3))


def _plateaued(prev: dict[str, Any], last: dict[str, Any], tol: float) -> tuple[bool, float, float]:
    cl_last, cl_prev = last.get("CL"), prev.get("CL")
    cd_last, cd_prev = last.get("CD"), prev.get("CD")
    if cl_last is None or cl_prev is None or cd_last is None or cd_prev is None:
        return False, float("nan"), float("nan")
    d_cl = abs(cl_last - cl_prev) / max(abs(cl_last), 1e-9)
    d_cd = abs(cd_last - cd_prev) / max(abs(cd_last), 1e-9)
    return (
        d_cl < tol and d_cd < tol and bool(last.get("cauchy_triggered")),
        d_cl,
        d_cd,
    )


def _print_rung(rec: dict[str, Any]) -> None:
    print(
        f"  rung {rec['rung']}: density={rec['surface_density']:>5d}  "
        f"n_elem={rec.get('mesh_n_elem')!s:>9}  "
        f"CL={rec.get('CL')!s:>8}  CD={rec.get('CD')!s:>8}  "
        f"L/D={rec.get('L_over_D')!s:>7}  "
        f"cauchy={rec.get('cauchy_triggered')!s:<5}  "
        f"t={rec.get('runtime_seconds')!s}s",
        flush=True,
    )


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the open-ended refinement loop and return the history dict."""
    from su2_mcp import cpacs_adapter as a

    cpacs_path = Path(args.cpacs).resolve()
    if not cpacs_path.exists():
        raise FileNotFoundError(f"CPACS file not found: {cpacs_path}")

    step_path = Path(args.step).resolve() if args.step else None
    if step_path is not None and not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")
    mesh_path = Path(args.mesh).resolve() if args.mesh else None
    if mesh_path is not None and not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    if step_path is None and mesh_path is None:
        raise ValueError("Provide --step (preferred) or --mesh for the first rung.")

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    xml = cpacs_path.read_text(encoding="utf-8")
    fc = {"mach": args.mach, "aoa": args.aoa, "altitude_ft": args.altitude}

    history: list[dict[str, Any]] = []
    density = int(args.start_density)
    wall_start = time.time()
    status = "budget_exhausted"
    stop_reason = "max_rungs"

    print(
        f"[converge] start: density={density}, growth={args.growth}, "
        f"max_rungs={args.max_rungs}, max_wall={args.max_wall_seconds}s, "
        f"plateau_tol={args.plateau_tol}",
        flush=True,
    )

    for rung_idx in range(1, int(args.max_rungs) + 1):
        rung_dir = out_root / f"rung_{rung_idx:02d}_density_{density}"
        rung_dir.mkdir(parents=True, exist_ok=True)
        rung_start = time.time()

        # First rung may reuse an existing mesh; subsequent rungs always
        # remesh from STEP at the new density.
        use_step = str(step_path) if step_path else None
        use_mesh = str(mesh_path) if (mesh_path and rung_idx == 1 and step_path is None) else None

        try:
            _new_xml, summary = a.run_adapter(
                xml,
                flight_conditions=fc,
                step_path=use_step,
                mesh_path=use_mesh,
                output_dir=str(rung_dir),
                preset="industry",
                surface_density=density,
                iter_cap=int(args.iter_cap),
                cl_convergence_eps=float(args.cl_eps),
                wall_timeout_seconds=int(args.per_rung_timeout),
            )
        except Exception as exc:
            rec = {
                "rung": rung_idx,
                "surface_density": density,
                "error": {"type": "adapter_exception", "message": str(exc)},
                "runtime_seconds": round(time.time() - rung_start, 2),
            }
            history.append(rec)
            _print_rung(rec)
            status = "error"
            stop_reason = "adapter_exception"
            break

        rec = {
            "rung": rung_idx,
            "surface_density": density,
            "mesh_n_elem": summary.get("mesh_n_elem"),
            "CL": summary.get("CL"),
            "CD": summary.get("CD"),
            "L_over_D": summary.get("L_over_D"),
            "cauchy_triggered": summary.get("cauchy_triggered"),
            "runtime_seconds": summary.get("runtime_seconds"),
            "preset_label": summary.get("preset_label"),
            "error": summary.get("error"),
            "output_dir": summary.get("output_dir"),
        }
        history.append(rec)
        _print_rung(rec)

        if rec.get("error"):
            status = "error"
            stop_reason = rec["error"].get("type", "unknown")
            break

        if len(history) >= 2:
            ok, d_cl, d_cd = _plateaued(history[-2], history[-1], args.plateau_tol)
            print(
                f"    Δ vs prev: ΔCL/CL={d_cl:.4f}  ΔCD/CD={d_cd:.4f}  "
                f"cauchy={rec.get('cauchy_triggered')}",
                flush=True,
            )
            if ok:
                status = "plateaued"
                stop_reason = "outer_plateau_and_inner_cauchy"
                break

        elapsed = time.time() - wall_start
        if elapsed >= args.max_wall_seconds:
            status = "budget_exhausted"
            stop_reason = "max_wall_seconds"
            break

        next_density = _next_density(density, args.growth)
        projected = _projected_n_elem(history, next_density)
        if projected is not None and projected > args.max_n_elem:
            status = "budget_exhausted"
            stop_reason = "max_n_elem_projection"
            print(
                f"  next rung would project {projected:,} cells > "
                f"{args.max_n_elem:,}; stopping.",
                flush=True,
            )
            break

        density = next_density

    history_doc = {
        "status": status,
        "stop_reason": stop_reason,
        "final": history[-1] if history else None,
        "history": history,
        "config": {
            "cpacs": str(cpacs_path),
            "step": str(step_path) if step_path else None,
            "mesh": str(mesh_path) if mesh_path else None,
            "mach": args.mach,
            "aoa_deg": args.aoa,
            "altitude_ft": args.altitude,
            "plateau_tol": args.plateau_tol,
            "growth": args.growth,
            "start_density": int(args.start_density),
            "max_rungs": int(args.max_rungs),
            "max_wall_seconds": int(args.max_wall_seconds),
            "max_n_elem": int(args.max_n_elem),
            "iter_cap": int(args.iter_cap),
            "cl_convergence_eps": float(args.cl_eps),
            "per_rung_timeout": int(args.per_rung_timeout),
        },
        "elapsed_wall_seconds": round(time.time() - wall_start, 2),
    }

    out_file = out_root / "convergence_history.json"
    out_file.write_text(json.dumps(history_doc, indent=2), encoding="utf-8")
    print(f"[converge] status={status} ({stop_reason}); wrote {out_file}", flush=True)
    return history_doc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        doc = run_loop(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if doc.get("status") == "plateaued":
        return 0
    if doc.get("status") == "budget_exhausted":
        return 0  # honest no-plateau result is still a successful run
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
