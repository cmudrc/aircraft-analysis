#!/usr/bin/env python3
"""Open-ended SU2 angle-of-attack sweep / trim search (no LLM needed).

This is the deterministic counterpart to the agent-driven
[`SKILL_AOA_SWEEP.md`](../agent-mcp/skills/SKILL_AOA_SWEEP.md) skill. It
is the second iterative skill in the family started by
[`run_converged_su2.py`](run_converged_su2.py): instead of escalating
mesh density until CL/CD plateau, it holds the mesh fixed and sweeps the
angle of attack, characterising the lift/drag polar and reporting two
operating points the agent (or a human) usually wants:

  * the **max-L/D** angle, and
  * the **trim** angle that hits a requested target lift coefficient
    (``--target-cl``), found by linear interpolation between the two
    bracketing sweep points.

Example::

    python scripts/run_aoa_sweep.py \\
        --cpacs D150_v30.xml \\
        --mesh pipeline/d150_final/aircraft_volume.su2 \\
        --mach 0.78 --altitude 35000 \\
        --aoa-list 0,1,2,3,4 --target-cl 0.5

Because angle of attack does not change the geometry, the mesh is built
**once** (or supplied via ``--mesh``) and reused for every point, so an
N-point sweep costs one mesh + N flow solves rather than N meshes.

The script wraps the real ``su2_mcp.cpacs_adapter.run_adapter`` call. No
fake CL/CD or interpolated "trim" is ever emitted from missing data:
when SU2 fails for a point the structured adapter error is recorded and
the point is skipped; if no point succeeds the script exits non-zero,
per the project's no-stubs rule.

Output: ``<output_root>/aoa_sweep_history.json`` plus a per-point
sub-directory holding the SU2 working files.
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
# real SU2 binary live on PATH (mirrors run_converged_su2.py).
_VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"
if (
    _VENV_PY.exists()
    and not os.environ.get("AOA_SWEEP_RESPAWNED")
    and Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve()
):
    os.environ["AOA_SWEEP_RESPAWNED"] = "1"
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
            "Sweep SU2 over a set of angles of attack on a fixed mesh, then "
            "report the max-L/D point and (optionally) the trim angle for a "
            "target CL by linear interpolation. Honours a wall-clock budget."
        )
    )
    p.add_argument("--cpacs", required=True, help="Path to the CPACS XML file.")
    p.add_argument(
        "--mesh",
        default=None,
        help=(
            "Path to an existing .su2 volume mesh, reused for every angle. "
            "Preferred: a sweep then costs zero meshing."
        ),
    )
    p.add_argument(
        "--step",
        default=None,
        help=(
            "Path to STEP geometry. Used only if --mesh is absent: the first "
            "angle meshes from STEP and the generated mesh is reused for the "
            "rest of the sweep."
        ),
    )
    p.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "aoa_sweep_runs"),
        help="Root directory for per-point output (default: ./aoa_sweep_runs).",
    )

    # Angle set: either an explicit list or a min/max/step range.
    p.add_argument(
        "--aoa-list",
        default=None,
        help="Comma-separated angles in degrees, e.g. '0,1,2,3,4'.",
    )
    p.add_argument("--aoa-min", type=float, default=None,
                   help="Range start (deg); used if --aoa-list is absent.")
    p.add_argument("--aoa-max", type=float, default=None,
                   help="Range end (deg, inclusive); used with --aoa-min.")
    p.add_argument("--aoa-step", type=float, default=1.0,
                   help="Range step (deg, default 1.0); used with --aoa-min/--aoa-max.")

    p.add_argument(
        "--target-cl",
        type=float,
        default=None,
        help="Optional target lift coefficient; the trim angle is interpolated.",
    )

    # Fixed flight condition (everything except AoA)
    p.add_argument("--mach", type=float, default=0.78)
    p.add_argument("--altitude", type=float, default=35000.0,
                   help="Altitude in feet (default 35000).")

    # Mesh fidelity (constant across the sweep)
    p.add_argument("--preset", default="workstation",
                   choices=["laptop", "workstation", "industry"],
                   help="SU2 mesh/solver preset (default workstation).")
    p.add_argument("--surface-density", type=int, default=None,
                   help="Optional explicit Gmsh surface density override.")

    # Budget
    p.add_argument("--max-wall-seconds", type=int, default=14400,
                   help="Total wall-clock budget across all angles (default 14400 s).")
    p.add_argument("--iter-cap", type=int, default=800,
                   help="SU2 inner iteration cap per point (default 800).")
    p.add_argument("--cl-eps", type=float, default=1e-4,
                   help="Cauchy convergence tolerance on LIFT (default 1e-4).")
    p.add_argument("--per-point-timeout", type=int, default=7200,
                   help="Per-point wall timeout for SU2_CFD (default 7200 s).")

    return p.parse_args(argv)


def aoa_values(args: argparse.Namespace) -> list[float]:
    """Resolve the ordered list of angles from --aoa-list or the range flags."""
    if args.aoa_list:
        vals = [float(x) for x in str(args.aoa_list).split(",") if x.strip() != ""]
    elif args.aoa_min is not None and args.aoa_max is not None:
        if args.aoa_step <= 0:
            raise ValueError("--aoa-step must be positive.")
        if args.aoa_max < args.aoa_min:
            raise ValueError("--aoa-max must be >= --aoa-min.")
        vals = []
        v = args.aoa_min
        # Guard against float drift; include the endpoint within half a step.
        while v <= args.aoa_max + args.aoa_step * 0.5:
            vals.append(round(v, 6))
            v += args.aoa_step
    else:
        raise ValueError("Provide --aoa-list or both --aoa-min and --aoa-max.")
    if not vals:
        raise ValueError("Resolved angle list is empty.")
    # De-duplicate while preserving order, then sort ascending for interpolation.
    seen: set[float] = set()
    uniq = [x for x in vals if not (x in seen or seen.add(x))]
    return sorted(uniq)


def best_ld(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the successful point with the highest L/D, or None."""
    usable = [
        pt for pt in points
        if pt.get("error") is None and isinstance(pt.get("L_over_D"), (int, float))
    ]
    if not usable:
        return None
    return max(usable, key=lambda pt: pt["L_over_D"])


def interp_trim(points: list[dict[str, Any]], target_cl: float) -> dict[str, Any] | None:
    """Linearly interpolate the AoA that yields ``target_cl``.

    Scans consecutive (by angle) successful points for the first pair that
    brackets ``target_cl`` and interpolates the trim angle and the drag
    there. Returns None when no pair brackets the target (the honest
    answer: the sweep did not span the requested lift).
    """
    usable = sorted(
        (
            pt for pt in points
            if pt.get("error") is None
            and isinstance(pt.get("CL"), (int, float))
            and isinstance(pt.get("aoa"), (int, float))
        ),
        key=lambda pt: pt["aoa"],
    )
    for lo, hi in zip(usable, usable[1:]):
        cl_lo, cl_hi = lo["CL"], hi["CL"]
        if cl_lo == cl_hi:
            continue
        # Bracketed when target sits between the two CL values (either order).
        if (cl_lo - target_cl) * (cl_hi - target_cl) <= 0:
            t = (target_cl - cl_lo) / (cl_hi - cl_lo)
            aoa_trim = lo["aoa"] + t * (hi["aoa"] - lo["aoa"])
            cd_trim = None
            ld_trim = None
            if isinstance(lo.get("CD"), (int, float)) and isinstance(hi.get("CD"), (int, float)):
                cd_trim = lo["CD"] + t * (hi["CD"] - lo["CD"])
                if cd_trim not in (None, 0):
                    ld_trim = target_cl / cd_trim
            return {
                "target_cl": target_cl,
                "aoa_trim_deg": round(aoa_trim, 4),
                "cd_trim": round(cd_trim, 6) if cd_trim is not None else None,
                "ld_trim": round(ld_trim, 4) if ld_trim is not None else None,
                "bracket_aoa_deg": [lo["aoa"], hi["aoa"]],
                "bracket_cl": [cl_lo, cl_hi],
            }
    return None


def _print_point(rec: dict[str, Any]) -> None:
    print(
        f"  aoa={rec['aoa']!s:>6}deg  "
        f"CL={rec.get('CL')!s:>8}  CD={rec.get('CD')!s:>8}  "
        f"L/D={rec.get('L_over_D')!s:>7}  "
        f"t={rec.get('runtime_seconds')!s}s"
        + ("  ERROR" if rec.get("error") else ""),
        flush=True,
    )


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the AoA sweep and return the history document."""
    from su2_mcp import cpacs_adapter as a

    cpacs_path = Path(args.cpacs).resolve()
    if not cpacs_path.exists():
        raise FileNotFoundError(f"CPACS file not found: {cpacs_path}")

    mesh_path = Path(args.mesh).resolve() if args.mesh else None
    if mesh_path is not None and not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    step_path = Path(args.step).resolve() if args.step else None
    if step_path is not None and not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")
    if mesh_path is None and step_path is None:
        raise ValueError("Provide --mesh (preferred) or --step for meshing.")

    angles = aoa_values(args)
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    xml = cpacs_path.read_text(encoding="utf-8")

    # Mesh once, reuse everywhere: start from the supplied mesh if given,
    # otherwise mesh from STEP on the first point and reuse the result.
    reuse_mesh: Path | None = mesh_path
    points: list[dict[str, Any]] = []
    wall_start = time.time()
    status = "completed"
    stop_reason = "all_points_done"

    print(
        f"[aoa-sweep] angles={angles}  mach={args.mach}  alt={args.altitude}ft  "
        f"preset={args.preset}  target_cl={args.target_cl}",
        flush=True,
    )

    for idx, aoa in enumerate(angles, start=1):
        pt_dir = out_root / f"point_{idx:02d}_aoa_{str(aoa).replace('.', 'p')}"
        pt_dir.mkdir(parents=True, exist_ok=True)
        pt_start = time.time()

        fc = {"mach": args.mach, "aoa": aoa, "altitude_ft": args.altitude}

        # Use the reusable mesh when we have one; only the first point may
        # need to mesh from STEP.
        use_mesh = str(reuse_mesh) if reuse_mesh is not None else None
        use_step = str(step_path) if (reuse_mesh is None and step_path) else None

        try:
            _new_xml, summary = a.run_adapter(
                xml,
                flight_conditions=fc,
                step_path=use_step,
                mesh_path=use_mesh,
                output_dir=str(pt_dir),
                preset=args.preset,
                surface_density=args.surface_density,
                iter_cap=int(args.iter_cap),
                cl_convergence_eps=float(args.cl_eps),
                wall_timeout_seconds=int(args.per_point_timeout),
            )
        except Exception as exc:  # noqa: BLE001 - recorded structurally below
            rec = {
                "aoa": aoa,
                "error": {"type": "adapter_exception", "message": str(exc)},
                "runtime_seconds": round(time.time() - pt_start, 2),
            }
            points.append(rec)
            _print_point(rec)
            continue

        rec = {
            "aoa": aoa,
            "CL": summary.get("CL"),
            "CD": summary.get("CD"),
            "L_over_D": summary.get("L_over_D"),
            "cauchy_triggered": summary.get("cauchy_triggered"),
            "mesh_n_elem": summary.get("mesh_n_elem"),
            "runtime_seconds": summary.get("runtime_seconds"),
            "error": summary.get("error"),
            "output_dir": summary.get("output_dir"),
        }
        points.append(rec)
        _print_point(rec)

        # After the first successful mesh-from-STEP point, reuse its mesh.
        if reuse_mesh is None and rec.get("error") is None:
            generated = pt_dir / "aircraft_volume.su2"
            if not generated.exists():
                candidates = sorted(pt_dir.glob("*.su2"))
                generated = candidates[0] if candidates else generated
            if generated.exists():
                reuse_mesh = generated.resolve()
                print(f"    reusing generated mesh for remaining points: {reuse_mesh}",
                      flush=True)

        if time.time() - wall_start >= args.max_wall_seconds:
            status = "budget_exhausted"
            stop_reason = "max_wall_seconds"
            print("  wall-clock budget exhausted; stopping sweep.", flush=True)
            break

    successes = [pt for pt in points if pt.get("error") is None and pt.get("CL") is not None]
    if not successes:
        status = "error"
        stop_reason = "no_successful_points"

    summary_best_ld = best_ld(points)
    summary_trim = (
        interp_trim(points, args.target_cl) if args.target_cl is not None else None
    )

    history_doc = {
        "status": status,
        "stop_reason": stop_reason,
        "best_ld": summary_best_ld,
        "trim": summary_trim,
        "n_points": len(points),
        "n_success": len(successes),
        "points": points,
        "config": {
            "cpacs": str(cpacs_path),
            "mesh": str(mesh_path) if mesh_path else None,
            "step": str(step_path) if step_path else None,
            "angles_deg": angles,
            "mach": args.mach,
            "altitude_ft": args.altitude,
            "preset": args.preset,
            "surface_density": args.surface_density,
            "target_cl": args.target_cl,
            "max_wall_seconds": int(args.max_wall_seconds),
            "iter_cap": int(args.iter_cap),
            "cl_convergence_eps": float(args.cl_eps),
            "per_point_timeout": int(args.per_point_timeout),
        },
        "elapsed_wall_seconds": round(time.time() - wall_start, 2),
    }

    out_file = out_root / "aoa_sweep_history.json"
    out_file.write_text(json.dumps(history_doc, indent=2), encoding="utf-8")

    if summary_best_ld is not None:
        print(
            f"[aoa-sweep] best L/D = {summary_best_ld.get('L_over_D')} "
            f"at aoa={summary_best_ld.get('aoa')}deg",
            flush=True,
        )
    if summary_trim is not None:
        print(
            f"[aoa-sweep] trim aoa for CL={args.target_cl} ~ "
            f"{summary_trim['aoa_trim_deg']}deg (L/D~{summary_trim['ld_trim']})",
            flush=True,
        )
    elif args.target_cl is not None:
        print(
            f"[aoa-sweep] target CL={args.target_cl} not bracketed by the swept "
            f"angles; widen the range to trim it.",
            flush=True,
        )
    print(f"[aoa-sweep] status={status} ({stop_reason}); wrote {out_file}", flush=True)
    return history_doc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        doc = run_loop(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # An honest sweep that completed (even if a target CL wasn't bracketed)
    # is still a successful run; only a total solver failure is non-zero.
    return 0 if doc.get("status") in ("completed", "budget_exhausted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
