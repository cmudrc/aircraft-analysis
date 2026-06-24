#!/usr/bin/env python3
"""Open-ended engine resizing until the mission closes (no LLM needed).

This is the deterministic counterpart to the agent-driven
[`SKILL_ENGINE_RESIZE.md`](../agent-mcp/skills/SKILL_ENGINE_RESIZE.md) skill
and the third member of the iterative-skill family that began with
[`run_converged_su2.py`](run_converged_su2.py) (mesh escalation) and
[`run_aoa_sweep.py`](run_aoa_sweep.py) (angle sweep / trim).

Where those two stay inside the aerodynamics tool, this skill couples **two**
disciplines: it resizes the engine (pyCycle) and re-flies the mission (NSEG)
in a loop until the engine is just large enough to close the mission at the
binding sizing point — top of climb.

The physics of "closing the mission"
------------------------------------
NSEG's segment integrators assume thrust is always available, so on their own
they never tell you whether an engine is big enough.  This harness relies on
the *thrust-closure* block that ``nseg`` now reports: at the cruise ceiling the
engine must deliver the cruise drag **plus** enough excess thrust for a
residual rate of climb (~300 ft/min). That required thrust is compared against
the engine's installed thrust (``Fn_N`` from pyCycle). A negative margin means
the engine is too small; the mission does not close.

The loop
--------
1. Run **pyCycle** at the cruise design point with a trial design thrust
   (``Fn_DES``) → installed thrust ``Fn_N`` and ``TSFC``.
2. Run **NSEG** with that engine → top-of-climb thrust margin + block fuel.
3. Newton-correct the design thrust toward the requested margin
   (``margin → target``). Because pyCycle sizes the cycle so that the achieved
   net thrust equals ``Fn_DES`` at the design point, the margin responds ~1:1
   to the design-thrust change, so a single Newton gain converges in a few
   iterations.

It converges on the *smallest* engine that meets the target margin — i.e. the
most fuel-efficient engine that still closes the mission — rather than just
"some engine that works".

No fake numbers are ever emitted: both ``pyCycle`` and ``NSEG`` are the real
adapters. If pyCycle/OpenMDAO is missing, or a solver errors, the structured
error is recorded and the script exits non-zero, per the project no-stubs rule.

Example::

    python scripts/run_engine_resize.py \\
        --cpacs D150_v30.xml \\
        --mach 0.78 --altitude 35000 \\
        --weight 70000 --range-km 3000 \\
        --target-margin-frac 0.05

Output: ``<output_root>/engine_resize_history.json`` and the final updated
CPACS XML carrying the converged engine + mission results.
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

# Auto-relaunch under the project's .venv so pyCycle / OpenMDAO are importable
# (mirrors run_converged_su2.py / run_aoa_sweep.py).
_VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"
if (
    _VENV_PY.exists()
    and not os.environ.get("ENGINE_RESIZE_RESPAWNED")
    and Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve()
):
    os.environ["ENGINE_RESIZE_RESPAWNED"] = "1"
    os.execv(
        str(_VENV_PY),
        [str(_VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

# Make the source trees importable even before pip-install.
for sub in ("pycycle-mcp/src", "nseg-mcp/src"):
    p = PROJECT_ROOT / sub
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

N_TO_LBF = 0.224809
LBF_TO_N = 4.44822
FT_TO_M = 0.3048


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Resize the engine (pyCycle) until the mission (NSEG) closes at the "
            "top-of-climb sizing point, converging on the smallest engine that "
            "meets the requested thrust margin."
        )
    )
    p.add_argument("--cpacs", required=True, help="Path to the CPACS XML file.")
    p.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "engine_resize_runs"),
        help="Directory for history + updated CPACS (default ./engine_resize_runs).",
    )

    # Cruise design point (shared by pyCycle and NSEG).
    p.add_argument("--mach", type=float, default=0.78, help="Cruise Mach (default 0.78).")
    p.add_argument("--altitude", type=float, default=35000.0,
                   help="Cruise altitude in feet (default 35000).")

    # Mission inputs.
    p.add_argument("--weight", type=float, default=78000.0,
                   help="Takeoff gross weight [kg] (default 78000).")
    p.add_argument("--range-km", type=float, default=3000.0,
                   help="Cruise range [km] (default 3000).")

    # Sizing target.
    p.add_argument("--target-margin-frac", type=float, default=0.0,
                   help="Target top-of-climb thrust margin as a fraction of "
                        "required thrust (default 0.0 = engine just closes).")
    p.add_argument("--target-margin-n", type=float, default=None,
                   help="Absolute target margin [N]; overrides --target-margin-frac.")
    p.add_argument("--tol-frac", type=float, default=0.01,
                   help="Convergence tolerance on margin as a fraction of "
                        "required thrust (default 0.01).")
    p.add_argument("--tol-n", type=float, default=200.0,
                   help="Absolute convergence tolerance floor [N] (default 200).")

    # Search controls.
    p.add_argument("--start-thrust-lbf", type=float, default=None,
                   help="Initial design thrust Fn_DES [lbf]; default lets pyCycle "
                        "pick from CPACS aero drag.")
    p.add_argument("--min-thrust-lbf", type=float, default=1000.0,
                   help="Lower bound on design thrust [lbf] (default 1000).")
    p.add_argument("--max-thrust-lbf", type=float, default=60000.0,
                   help="Upper bound on design thrust [lbf] (default 60000).")
    p.add_argument("--max-iters", type=int, default=12,
                   help="Maximum resize iterations (default 12).")
    p.add_argument("--gain", type=float, default=1.0,
                   help="Newton damping on the design-thrust update (default 1.0).")
    p.add_argument("--max-wall-seconds", type=int, default=14400,
                   help="Total wall-clock budget (default 14400 s).")

    return p.parse_args(argv)


def newton_step(
    design_thrust_lbf: float,
    margin_n: float,
    target_n: float,
    gain: float,
    min_lbf: float,
    max_lbf: float,
) -> float:
    """Next design thrust: move the achieved margin toward the target.

    pyCycle sizes the cycle so the achieved net thrust matches ``Fn_DES`` at the
    design point, so d(margin)/d(Fn_DES) ≈ 1. We therefore shift the design
    thrust by the margin error (converted lbf), damped by ``gain``, and clamp to
    the search bounds.
    """
    step_lbf = gain * (target_n - margin_n) * N_TO_LBF
    return max(min_lbf, min(max_lbf, design_thrust_lbf + step_lbf))


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the resize loop and return the history document."""
    import nseg_mcp.cpacs_adapter as nseg
    import pycycle_mcp.cpacs_adapter as pyc

    cpacs_path = Path(args.cpacs).resolve()
    if not cpacs_path.exists():
        raise FileNotFoundError(f"CPACS file not found: {cpacs_path}")
    if args.range_km <= 0:
        raise ValueError("--range-km must be positive.")
    if args.max_iters < 1:
        raise ValueError("--max-iters must be >= 1.")
    if args.min_thrust_lbf <= 0 or args.max_thrust_lbf <= args.min_thrust_lbf:
        raise ValueError("Require 0 < --min-thrust-lbf < --max-thrust-lbf.")

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    xml = cpacs_path.read_text(encoding="utf-8")

    fc = {"mach": args.mach, "altitude_ft": args.altitude}
    mission_profile = {
        "weight_kg": args.weight,
        "cruise_mach": args.mach,
        "cruise_altitude_m": args.altitude * FT_TO_M,
        "range_m": args.range_km * 1000.0,
    }

    print(
        f"[engine-resize] mach={args.mach} alt={args.altitude}ft "
        f"weight={args.weight}kg range={args.range_km}km "
        f"target_margin_frac={args.target_margin_frac}",
        flush=True,
    )

    design_thrust_lbf = args.start_thrust_lbf
    iterations: list[dict[str, Any]] = []
    wall_start = time.time()
    status = "did_not_converge"
    stop_reason = "max_iters"
    final_xml = xml
    converged_record: dict[str, Any] | None = None

    for it in range(1, args.max_iters + 1):
        # 1. Engine — real pyCycle at the cruise design point.
        try:
            eng_xml, eng = pyc.run_adapter(
                xml, flight_conditions=fc, design_thrust_lbf=design_thrust_lbf
            )
        except Exception as exc:  # noqa: BLE001 - recorded structurally
            return _abort(out_root, iterations, args,
                          {"type": "adapter_exception", "stage": "pycycle", "message": str(exc)})
        if eng.get("error"):
            return _abort(out_root, iterations, args,
                          {"stage": "pycycle", **_as_error(eng["error"])})

        fn_n = float(eng.get("Fn_N") or 0.0)
        tsfc = eng.get("TSFC_1_per_s")
        fn_des_used = eng.get("Fn_DES_lbf")
        if fn_n <= 0.0:
            return _abort(out_root, iterations, args,
                          {"stage": "pycycle", "type": "invalid_thrust",
                           "message": f"pyCycle returned non-positive Fn_N={fn_n}"})

        # 2. Mission — real NSEG with that engine.
        try:
            mis_xml, mission = nseg.run_adapter(eng_xml, mission_profile=mission_profile)
        except Exception as exc:  # noqa: BLE001
            return _abort(out_root, iterations, args,
                          {"type": "adapter_exception", "stage": "nseg", "message": str(exc)})
        if mission.get("error") or not mission.get("success"):
            err = mission.get("error") or {"type": "mission_failed", "message": "NSEG did not succeed"}
            return _abort(out_root, iterations, args, {"stage": "nseg", **_as_error(err)})

        tc = mission.get("thrust_closure")
        if tc is None:
            return _abort(out_root, iterations, args,
                          {"stage": "nseg", "type": "no_thrust_closure",
                           "message": "NSEG returned no thrust_closure block; "
                                      "cannot size the engine. Ensure a cruise segment "
                                      "and engine thrust are present."})

        t_req = float(tc["thrust_required_n"])
        margin = float(tc["thrust_margin_n"])
        target_n = (
            float(args.target_margin_n)
            if args.target_margin_n is not None
            else args.target_margin_frac * t_req
        )
        tol_n = max(args.tol_frac * t_req, args.tol_n)

        rec = {
            "iter": it,
            "design_thrust_lbf": round(float(fn_des_used), 2) if fn_des_used is not None else None,
            "Fn_N": round(fn_n, 2),
            "Fn_lbf": eng.get("Fn_lbf"),
            "TSFC_1_per_s": tsfc,
            "thrust_required_n": round(t_req, 2),
            "thrust_margin_n": round(margin, 2),
            "target_margin_n": round(target_n, 2),
            "tol_n": round(tol_n, 2),
            "thrust_limited": bool(tc.get("thrust_limited")),
            "block_fuel_kg": round(float(mission.get("total_fuel_burned_kg", 0.0)), 2),
            "elapsed_s": round(time.time() - wall_start, 2),
        }
        iterations.append(rec)
        print(
            f"  it={it:>2} Fn={fn_n:>10.1f}N  T_req={t_req:>10.1f}N  "
            f"margin={margin:>+10.1f}N (target {target_n:>+.1f}±{tol_n:.0f})  "
            f"fuel={rec['block_fuel_kg']}kg",
            flush=True,
        )

        if abs(margin - target_n) <= tol_n:
            status = "converged"
            stop_reason = "margin_within_tolerance"
            final_xml = mis_xml
            converged_record = rec
            break

        # 3. Newton update on the design thrust.
        base_lbf = (
            float(fn_des_used)
            if fn_des_used is not None
            else fn_n * N_TO_LBF
        )
        next_thrust = newton_step(
            base_lbf, margin, target_n, args.gain,
            args.min_thrust_lbf, args.max_thrust_lbf,
        )
        if next_thrust in (args.min_thrust_lbf, args.max_thrust_lbf) and (
            abs(next_thrust - base_lbf) < 1e-6
        ):
            status = "did_not_converge"
            stop_reason = "thrust_bound_reached"
            final_xml = mis_xml
            print(f"  design thrust hit bound at {next_thrust} lbf; stopping.", flush=True)
            break
        design_thrust_lbf = next_thrust

        if time.time() - wall_start >= args.max_wall_seconds:
            status = "did_not_converge"
            stop_reason = "max_wall_seconds"
            final_xml = mis_xml
            print("  wall-clock budget exhausted; stopping.", flush=True)
            break

    out_xml_path = out_root / "engine_resize_final.xml"
    out_xml_path.write_text(final_xml, encoding="utf-8")

    history_doc = {
        "status": status,
        "stop_reason": stop_reason,
        "converged": converged_record,
        "n_iters": len(iterations),
        "iterations": iterations,
        "final_cpacs": str(out_xml_path),
        "config": {
            "cpacs": str(cpacs_path),
            "mach": args.mach,
            "altitude_ft": args.altitude,
            "weight_kg": args.weight,
            "range_km": args.range_km,
            "target_margin_frac": args.target_margin_frac,
            "target_margin_n": args.target_margin_n,
            "tol_frac": args.tol_frac,
            "tol_n": args.tol_n,
            "start_thrust_lbf": args.start_thrust_lbf,
            "min_thrust_lbf": args.min_thrust_lbf,
            "max_thrust_lbf": args.max_thrust_lbf,
            "max_iters": args.max_iters,
            "gain": args.gain,
        },
        "elapsed_wall_seconds": round(time.time() - wall_start, 2),
    }
    out_file = out_root / "engine_resize_history.json"
    out_file.write_text(json.dumps(history_doc, indent=2), encoding="utf-8")

    if converged_record is not None:
        print(
            f"[engine-resize] CONVERGED: Fn={converged_record['Fn_N']}N "
            f"(Fn_DES={converged_record['design_thrust_lbf']}lbf), "
            f"margin={converged_record['thrust_margin_n']}N, "
            f"block fuel={converged_record['block_fuel_kg']}kg",
            flush=True,
        )
    print(f"[engine-resize] status={status} ({stop_reason}); wrote {out_file}", flush=True)
    return history_doc


def _as_error(err: Any) -> dict[str, Any]:
    if isinstance(err, dict):
        return {"type": err.get("type", "error"), "message": err.get("message", str(err))}
    return {"type": "error", "message": str(err)}


def _abort(
    out_root: Path,
    iterations: list[dict[str, Any]],
    args: argparse.Namespace,
    error: dict[str, Any],
) -> dict[str, Any]:
    """Write an honest failure document and return it (no fake convergence)."""
    doc = {
        "status": "error",
        "stop_reason": "solver_error",
        "error": error,
        "n_iters": len(iterations),
        "iterations": iterations,
        "config": {"cpacs": str(Path(args.cpacs).resolve())},
    }
    out_file = out_root / "engine_resize_history.json"
    out_file.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"[engine-resize] ERROR ({error.get('stage', '?')}): {error.get('message')}",
          file=sys.stderr, flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        doc = run_loop(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0 if doc.get("status") == "converged" else 1


if __name__ == "__main__":
    raise SystemExit(main())
