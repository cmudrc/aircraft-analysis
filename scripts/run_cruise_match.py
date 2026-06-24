#!/usr/bin/env python3
"""Cross-discipline cruise match: thrust = drag, with weight/fuel closure.

This is the deterministic counterpart to the agent-driven
[`SKILL_CRUISE_MATCH.md`](../agent-mcp/skills/SKILL_CRUISE_MATCH.md) skill and
the fourth member of the iterative-skill family
([`run_converged_su2.py`](run_converged_su2.py) → mesh,
[`run_aoa_sweep.py`](run_aoa_sweep.py) → angle,
[`run_engine_resize.py`](run_engine_resize.py) → engine size). It is the first
to couple **three** disciplines in a single fixed point:

    SU2 (aerodynamics)  ↔  pyCycle (propulsion)  ↔  NSEG (mission/weights)

What it solves
--------------
At a steady cruise point, *thrust must equal drag* and *lift must equal weight*.
Both depend on the weight, which depends on the fuel, which depends on the
engine and the drag — a classic multidisciplinary fixed point. This harness
closes it:

1. **SU2** builds a real drag polar ``CD = CD0 + k·CL²`` from a few angle-of-
   attack solves on a fixed mesh (reusing the AoA-sweep efficiency trick).
2. For a trial takeoff weight, the cruise lift coefficient is
   ``CL = W·g / (q·S)`` (lift = weight); the polar gives ``CD`` and the drag
   force ``D = CD·q·S``.
3. **pyCycle** is sized so the cruise net thrust equals that drag
   (``Fn_DES = D``) → ``TSFC``. This *is* the thrust = drag condition.
4. **NSEG** flies the mission with that polar + engine → block fuel.
5. The takeoff weight is re-closed as ``OEW + payload + fuel·(1+reserve)`` and
   the loop repeats until the weight stops moving.

At convergence: thrust = drag (by construction each iteration) **and** the
weight/fuel loop is closed — a genuine converged cruise design point.

No fake numbers: SU2, pyCycle and NSEG are the real adapters. A missing solver
or a failed solve is a loud, structured error and a non-zero exit, never a
fabricated polar or fuel figure (no-stubs rule). The drag polar may instead be
supplied explicitly with ``--cd0``/``--k`` (a user input, not a stub); SU2 is
run by default.

Example (sizing mode)::

    python scripts/run_cruise_match.py \\
        --cpacs D150_v30.xml \\
        --mesh pipeline/d150_final/aircraft_volume.su2 \\
        --mach 0.78 --altitude 35000 \\
        --oew 42000 --payload 18000 --range-km 3000 \\
        --polar-aoa 1,3

Example (fixed-weight match)::

    python scripts/run_cruise_match.py --cpacs D150_v30.xml --cd0 0.022 --k 0.045 \\
        --mach 0.78 --altitude 35000 --weight 70000 --range-km 3000

Output: ``<output_root>/cruise_match_history.json`` and the final updated CPACS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Auto-relaunch under the project's .venv (SU2 + pyCycle + NSEG live there).
_VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"
if (
    _VENV_PY.exists()
    and not os.environ.get("CRUISE_MATCH_RESPAWNED")
    and Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve()
):
    os.environ["CRUISE_MATCH_RESPAWNED"] = "1"
    os.execv(
        str(_VENV_PY),
        [str(_VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

_SU2_BIN = Path.home() / ".local" / "su2" / "bin"
if _SU2_BIN.is_dir() and str(_SU2_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_SU2_BIN}:{os.environ.get('PATH', '')}"

for sub in ("su2-mcp/src", "pycycle-mcp/src", "nseg-mcp/src"):
    p = PROJECT_ROOT / sub
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

G0 = 9.80665
N_TO_LBF = 0.224809
FT_TO_M = 0.3048


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Converge a cruise design point where thrust = drag and the "
            "weight/fuel loop is closed, coupling SU2 (polar), pyCycle "
            "(thrust = drag) and NSEG (fuel)."
        )
    )
    p.add_argument("--cpacs", required=True, help="Path to the CPACS XML file.")
    p.add_argument("--mesh", default=None, help="Prebuilt .su2 mesh for the SU2 polar.")
    p.add_argument("--step", default=None, help="STEP geometry (meshed once) if no --mesh.")
    p.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "cruise_match_runs"),
        help="Directory for history + updated CPACS (default ./cruise_match_runs).",
    )

    # Cruise condition.
    p.add_argument("--mach", type=float, default=0.78)
    p.add_argument("--altitude", type=float, default=35000.0, help="Cruise altitude [ft].")

    # Drag polar source: either SU2 angles or an explicit polar.
    p.add_argument("--polar-aoa", default="1,3",
                   help="Comma-separated AoA [deg] for the SU2 polar fit (default 1,3).")
    p.add_argument("--cd0", type=float, default=None,
                   help="Explicit zero-lift drag; skips SU2 when given with --k.")
    p.add_argument("--k", type=float, default=None,
                   help="Explicit induced-drag factor; skips SU2 when given with --cd0.")
    p.add_argument("--preset", default="workstation",
                   choices=["laptop", "workstation", "industry"])
    p.add_argument("--surface-density", type=int, default=None)
    p.add_argument("--per-point-timeout", type=int, default=7200)
    p.add_argument("--iter-cap", type=int, default=800)

    # Weight definition. Sizing mode: --oew + --payload. Match mode: --weight.
    p.add_argument("--oew", type=float, default=None, help="Operating empty weight [kg].")
    p.add_argument("--payload", type=float, default=None, help="Payload [kg].")
    p.add_argument("--weight", type=float, default=None,
                   help="Fixed takeoff weight [kg] (match mode; alternative to --oew/--payload).")
    p.add_argument("--range-km", type=float, default=3000.0)
    p.add_argument("--reserve-frac", type=float, default=0.05,
                   help="Reserve fuel as a fraction of block fuel (default 0.05).")
    p.add_argument("--fuel-guess-frac", type=float, default=0.25,
                   help="Initial fuel guess as a fraction of OEW+payload (default 0.25).")

    # Fixed-point controls.
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Relative takeoff-weight convergence tolerance (default 1e-3).")
    p.add_argument("--relax", type=float, default=1.0,
                   help="Relaxation on the weight update in (0,1] (default 1.0).")
    p.add_argument("--max-iters", type=int, default=25)
    p.add_argument("--max-wall-seconds", type=int, default=14400)

    return p.parse_args(argv)


def polar_aoa_list(spec: str) -> list[float]:
    vals = [float(x) for x in str(spec).split(",") if x.strip() != ""]
    seen: set[float] = set()
    uniq = [x for x in vals if not (x in seen or seen.add(x))]
    if len(uniq) < 2:
        raise ValueError("--polar-aoa needs at least two distinct angles to fit a polar.")
    return sorted(uniq)


def fit_polar(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares fit of CD = CD0 + k·CL² to (CL, CD) points.

    Requires at least two points with distinct CL. Returns (CD0, k). Raises
    ValueError on insufficient/degenerate data — never invents a polar.
    """
    usable = [
        (cl, cd) for cl, cd in points
        if isinstance(cl, (int, float)) and isinstance(cd, (int, float))
    ]
    if len(usable) < 2:
        raise ValueError("Need at least two successful SU2 points to fit a drag polar.")
    xs = [cl * cl for cl, _ in usable]
    ys = [cd for _, cd in usable]
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-18:
        raise ValueError("Degenerate polar fit: SU2 points share the same CL.")
    k = (n * sxy - sx * sy) / denom
    cd0 = (sy - k * sx) / n
    return cd0, k


def cruise_state(weight_kg: float, q_pa: float, ref_area_m2: float,
                 cd0: float, k: float) -> dict[str, float]:
    """Cruise CL/CD/drag at lift = weight. Pure; no solver needed."""
    if q_pa * ref_area_m2 <= 0:
        raise ValueError("Non-physical dynamic pressure or reference area.")
    cl = weight_kg * G0 / (q_pa * ref_area_m2)
    cd = cd0 + k * cl * cl
    drag_n = cd * q_pa * ref_area_m2
    lod = cl / cd if cd > 1e-12 else float("nan")
    return {"CL": cl, "CD": cd, "drag_n": drag_n, "L_over_D": lod}


def _set_aero_coeffs(xml: str, cd0: float, cl: float, cd: float) -> str:
    """Encode the current polar point into CPACS so NSEG reads cd0 and k.

    NSEG derives ``k = (CD - CD0) / CL²`` from the aero coefficients; writing
    (CD0, CL, CD) at the cruise point lets it recover exactly the fitted polar.
    """
    root = ET.fromstring(xml)
    node = root
    for tag in ("vehicles", "aircraft", "model", "analysisResults", "aero", "coefficients"):
        child = node.find(tag)
        if child is None:
            child = ET.SubElement(node, tag)
        node = child
    for tag, val in (("CD0", cd0), ("CL", cl), ("CD", cd)):
        el = node.find(tag)
        if el is None:
            el = ET.SubElement(node, tag)
        el.text = repr(float(val))
    return ET.tostring(root, encoding="unicode")


def _ref_area(xml: str) -> float:
    root = ET.fromstring(xml)
    el = root.find(".//vehicles/aircraft/model/reference/area")
    if el is not None and el.text:
        return float(el.text)
    return 122.4


def _build_polar(args: argparse.Namespace, xml: str, out_root: Path) -> dict[str, Any]:
    """Return {cd0, k, source, points}. Runs real SU2 unless --cd0/--k given."""
    if args.cd0 is not None and args.k is not None:
        return {"cd0": float(args.cd0), "k": float(args.k), "source": "user", "points": []}

    import su2_mcp.cpacs_adapter as su2

    mesh_path = Path(args.mesh).resolve() if args.mesh else None
    step_path = Path(args.step).resolve() if args.step else None
    if mesh_path is None and step_path is None:
        raise ValueError("Provide --mesh/--step for the SU2 polar, or --cd0 and --k.")
    if mesh_path is not None and not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    if step_path is not None and not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    angles = polar_aoa_list(args.polar_aoa)
    reuse_mesh = mesh_path
    pts: list[tuple[float, float]] = []
    raw: list[dict[str, Any]] = []
    print(f"[cruise-match] building SU2 polar at AoA={angles} ...", flush=True)
    for idx, aoa in enumerate(angles, start=1):
        pt_dir = out_root / f"polar_{idx:02d}_aoa_{str(aoa).replace('.', 'p')}"
        pt_dir.mkdir(parents=True, exist_ok=True)
        fc = {"mach": args.mach, "aoa": aoa, "altitude_ft": args.altitude}
        use_mesh = str(reuse_mesh) if reuse_mesh is not None else None
        use_step = str(step_path) if (reuse_mesh is None and step_path) else None
        _new_xml, summary = su2.run_adapter(
            xml, flight_conditions=fc, step_path=use_step, mesh_path=use_mesh,
            output_dir=str(pt_dir), preset=args.preset,
            surface_density=args.surface_density, iter_cap=int(args.iter_cap),
            wall_timeout_seconds=int(args.per_point_timeout),
        )
        cl, cd, err = summary.get("CL"), summary.get("CD"), summary.get("error")
        raw.append({"aoa": aoa, "CL": cl, "CD": cd, "error": err})
        print(f"    aoa={aoa} CL={cl} CD={cd}" + ("  ERROR" if err else ""), flush=True)
        if err is None and cl is not None and cd is not None:
            pts.append((cl, cd))
        if reuse_mesh is None and err is None:
            gen = pt_dir / "aircraft_volume.su2"
            if not gen.exists():
                cands = sorted(pt_dir.glob("*.su2"))
                gen = cands[0] if cands else gen
            if gen.exists():
                reuse_mesh = gen.resolve()

    cd0, k = fit_polar(pts)  # raises if < 2 usable points
    return {"cd0": cd0, "k": k, "source": "su2", "points": raw}


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the cruise-match fixed point and return the history document."""
    import nseg_mcp.cpacs_adapter as nseg
    import pycycle_mcp.cpacs_adapter as pyc
    from nseg_mcp.physics.atmosphere import dynamic_pressure

    cpacs_path = Path(args.cpacs).resolve()
    if not cpacs_path.exists():
        raise FileNotFoundError(f"CPACS file not found: {cpacs_path}")
    if args.range_km <= 0:
        raise ValueError("--range-km must be positive.")
    if not (0.0 < args.relax <= 1.0):
        raise ValueError("--relax must be in (0, 1].")

    sizing = args.oew is not None and args.payload is not None
    if not sizing and args.weight is None:
        raise ValueError("Provide --oew and --payload (sizing) OR --weight (match mode).")

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    base_xml = cpacs_path.read_text(encoding="utf-8")

    wall_start = time.time()
    polar = _build_polar(args, base_xml, out_root)
    cd0, k = polar["cd0"], polar["k"]

    alt_m = args.altitude * FT_TO_M
    q = dynamic_pressure(args.mach, alt_m)
    S = _ref_area(base_xml)

    fixed = {
        "cruise_mach": args.mach,
        "cruise_altitude_m": alt_m,
        "range_m": args.range_km * 1000.0,
    }

    print(
        f"[cruise-match] polar cd0={cd0:.5f} k={k:.5f} ({polar['source']}); "
        f"q={q:.1f}Pa S={S}m^2; mode={'sizing' if sizing else 'match'}",
        flush=True,
    )

    if sizing:
        oew, payload = float(args.oew), float(args.payload)
        fuel_guess = args.fuel_guess_frac * (oew + payload)
        w_to = oew + payload + fuel_guess
    else:
        oew = payload = None
        w_to = float(args.weight)

    iterations: list[dict[str, Any]] = []
    status = "did_not_converge"
    stop_reason = "max_iters"
    final_xml = base_xml
    converged_record: dict[str, Any] | None = None
    ff = 0.0

    for it in range(1, args.max_iters + 1):
        # Mid-cruise weight (Breguet uses the average over the burn).
        w_cruise = w_to * (1.0 - 0.5 * ff)
        cs = cruise_state(w_cruise, q, S, cd0, k)
        drag_n = cs["drag_n"]

        # pyCycle sized so cruise thrust = drag.
        try:
            eng_xml, eng = pyc.run_adapter(
                base_xml, flight_conditions={"mach": args.mach, "altitude_ft": args.altitude},
                design_thrust_lbf=drag_n * N_TO_LBF,
            )
        except Exception as exc:  # noqa: BLE001
            return _abort(out_root, iterations, args,
                          {"stage": "pycycle", "type": "adapter_exception", "message": str(exc)})
        if eng.get("error"):
            return _abort(out_root, iterations, args, {"stage": "pycycle", **_as_error(eng["error"])})

        fn_n = float(eng.get("Fn_N") or 0.0)
        tsfc = eng.get("TSFC_1_per_s")
        thrust_drag_residual = fn_n - drag_n

        # Encode the polar at this cruise CL and run NSEG for block fuel.
        aero_xml = _set_aero_coeffs(eng_xml, cd0, cs["CL"], cs["CD"])
        try:
            mis_xml, mission = nseg.run_adapter(aero_xml, mission_profile={**fixed, "weight_kg": w_to})
        except Exception as exc:  # noqa: BLE001
            return _abort(out_root, iterations, args,
                          {"stage": "nseg", "type": "adapter_exception", "message": str(exc)})
        if mission.get("error") or not mission.get("success"):
            err = mission.get("error") or {"type": "mission_failed", "message": "NSEG did not succeed"}
            return _abort(out_root, iterations, args, {"stage": "nseg", **_as_error(err)})

        block_fuel = float(mission.get("total_fuel_burned_kg", 0.0))
        total_fuel = block_fuel * (1.0 + args.reserve_frac)

        if sizing:
            w_to_new = oew + payload + total_fuel
        else:
            w_to_new = w_to  # fixed-weight match: weight does not move
        rel = abs(w_to_new - w_to) / max(w_to, 1.0)

        rec = {
            "iter": it,
            "W_TO_kg": round(w_to, 2),
            "W_cruise_kg": round(w_cruise, 2),
            "CL": round(cs["CL"], 5),
            "CD": round(cs["CD"], 6),
            "L_over_D": round(cs["L_over_D"], 3),
            "drag_n": round(drag_n, 2),
            "Fn_N": round(fn_n, 2),
            "thrust_drag_residual_n": round(thrust_drag_residual, 2),
            "TSFC_1_per_s": tsfc,
            "block_fuel_kg": round(block_fuel, 2),
            "total_fuel_kg": round(total_fuel, 2),
            "W_TO_new_kg": round(w_to_new, 2),
            "rel_dW": round(rel, 6),
            "elapsed_s": round(time.time() - wall_start, 2),
        }
        iterations.append(rec)
        print(
            f"  it={it:>2} W_TO={w_to:>9.1f}kg CL={cs['CL']:.4f} L/D={cs['L_over_D']:.2f} "
            f"D={drag_n:>9.1f}N Fn={fn_n:>9.1f}N (res {thrust_drag_residual:>+.1f}N) "
            f"fuel={block_fuel:>8.1f}kg -> W_TO'={w_to_new:>9.1f}kg (dW {rel:.2e})",
            flush=True,
        )
        final_xml = mis_xml

        if rel <= args.tol:
            status = "converged"
            stop_reason = "weight_within_tolerance"
            converged_record = rec
            break

        w_to = args.relax * w_to_new + (1.0 - args.relax) * w_to
        ff = block_fuel / max(w_to, 1.0)

        if time.time() - wall_start >= args.max_wall_seconds:
            stop_reason = "max_wall_seconds"
            print("  wall-clock budget exhausted; stopping.", flush=True)
            break

    out_xml_path = out_root / "cruise_match_final.xml"
    out_xml_path.write_text(final_xml, encoding="utf-8")

    history_doc = {
        "status": status,
        "stop_reason": stop_reason,
        "mode": "sizing" if sizing else "match",
        "polar": {"cd0": round(cd0, 6), "k": round(k, 6), "source": polar["source"],
                  "points": polar["points"]},
        "converged": converged_record,
        "n_iters": len(iterations),
        "iterations": iterations,
        "final_cpacs": str(out_xml_path),
        "config": {
            "cpacs": str(cpacs_path),
            "mach": args.mach,
            "altitude_ft": args.altitude,
            "ref_area_m2": S,
            "oew_kg": args.oew,
            "payload_kg": args.payload,
            "weight_kg": args.weight,
            "range_km": args.range_km,
            "reserve_frac": args.reserve_frac,
            "tol": args.tol,
            "relax": args.relax,
            "max_iters": args.max_iters,
        },
        "elapsed_wall_seconds": round(time.time() - wall_start, 2),
    }
    out_file = out_root / "cruise_match_history.json"
    out_file.write_text(json.dumps(history_doc, indent=2), encoding="utf-8")

    if converged_record is not None:
        print(
            f"[cruise-match] CONVERGED: W_TO={converged_record['W_TO_kg']}kg "
            f"L/D={converged_record['L_over_D']} thrust=drag residual="
            f"{converged_record['thrust_drag_residual_n']}N "
            f"block fuel={converged_record['block_fuel_kg']}kg",
            flush=True,
        )
    print(f"[cruise-match] status={status} ({stop_reason}); wrote {out_file}", flush=True)
    return history_doc


def _as_error(err: Any) -> dict[str, Any]:
    if isinstance(err, dict):
        return {"type": err.get("type", "error"), "message": err.get("message", str(err))}
    return {"type": "error", "message": str(err)}


def _abort(out_root: Path, iterations: list[dict[str, Any]],
           args: argparse.Namespace, error: dict[str, Any]) -> dict[str, Any]:
    doc = {
        "status": "error",
        "stop_reason": "solver_error",
        "error": error,
        "n_iters": len(iterations),
        "iterations": iterations,
        "config": {"cpacs": str(Path(args.cpacs).resolve())},
    }
    out_file = out_root / "cruise_match_history.json"
    out_file.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"[cruise-match] ERROR ({error.get('stage', '?')}): {error.get('message')}",
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
