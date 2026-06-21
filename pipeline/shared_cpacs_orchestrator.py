#!/usr/bin/env python3
"""Shared-CPACS orchestrator for the MCP pipeline.

Loads a CPACS XML file, runs each MCP adapter in sequence, and tracks
every mutation as a versioned snapshot.  Each MCP reads from and writes
back to the same CPACS document.

Intermediate artifacts (STEP files, meshes) are managed in a shared
working directory and passed between adapters automatically.

Usage:
    python shared_cpacs_orchestrator.py <cpacs_file> [options]

Examples:
    python shared_cpacs_orchestrator.py ../D150_v30.xml
    python shared_cpacs_orchestrator.py ../D150_v30.xml --mcps tigl su2 pycycle mission
    python shared_cpacs_orchestrator.py ../pipeline/dlr_f25_run/DLR-F25_simple.xml --mach 0.82 --aoa 3.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project roots are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
for sub in (
    "tigl-mcp/src",
    "su2-mcp/src",
    "pycycle-mcp/src",
    "nseg-mcp/src",
    "aviary-cpacs-mcp/src",
    "mission-mcp/src",  # legacy, kept for back-compat with --mcps mission
):
    p = _PROJECT_ROOT / sub
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared_cpacs.manager import CPACSManager


def _find_existing_artifacts(cpacs_path: str) -> dict[str, str | None]:
    """Look for STEP and mesh files near the CPACS file from prior runs."""
    cpacs_dir = Path(cpacs_path).parent
    artifacts: dict[str, str | None] = {
        "step_path": None,
        "mesh_path": None,
    }

    for candidate in [
        cpacs_dir / "aircraft_fused.step",
        cpacs_dir / "aircraft.step",
        cpacs_dir / "aircraft_simple.step",
    ]:
        if candidate.exists() and candidate.stat().st_size > 100:
            artifacts["step_path"] = str(candidate)
            break

    for candidate in [
        cpacs_dir / "aircraft_volume.su2",
    ]:
        if candidate.exists() and candidate.stat().st_size > 100:
            artifacts["mesh_path"] = str(candidate)
            break

    return artifacts


def _load_converge_harness():
    """Locate and import the open-ended SU2 refinement harness.

    The script lives at ``scripts/run_converged_su2.py`` next to either
    this orchestrator (workspace layout) or the aircraft-analysis repo
    root. We resolve it via importlib so the orchestrator does not need
    the harness to be installed as a package.
    """
    import importlib.util
    from pathlib import Path

    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "scripts" / "run_converged_su2.py",          # workspace root layout
        here.parent.parent / "aircraft-analysis" / "scripts" / "run_converged_su2.py",
        here.parent / "scripts" / "run_converged_su2.py",
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("_converge_harness", p)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod.run_loop
    raise FileNotFoundError(
        "Could not locate scripts/run_converged_su2.py. Searched: "
        + ", ".join(str(c) for c in candidates)
    )


def _import_adapter(domain: str):
    """Lazily import the adapter for a domain."""
    if domain == "tigl":
        from tigl_mcp import cpacs_adapter
        return cpacs_adapter
    elif domain == "su2":
        from su2_mcp import cpacs_adapter
        return cpacs_adapter
    elif domain == "pycycle":
        from pycycle_mcp import cpacs_adapter
        return cpacs_adapter
    elif domain == "nseg":
        from nseg_mcp import cpacs_adapter
        return cpacs_adapter
    elif domain == "aviary":
        from aviary_cpacs_mcp import cpacs_adapter
        return cpacs_adapter
    elif domain == "mission":
        # Legacy path: the old combined mission-mcp (kept for back-compat).
        from mission_mcp import cpacs_adapter
        return cpacs_adapter
    else:
        raise ValueError(f"Unknown MCP domain: {domain}")


def run_pipeline(
    cpacs_path: str,
    mcps: list[str] | None = None,
    flight_conditions: dict | None = None,
    mission_profile: dict | None = None,
    output_dir: str | None = None,
    verbose: bool = True,
    extra_artifacts: dict | None = None,
    su2_preset: str | None = None,
    su2_surface_density: int | None = None,
    su2_farfield_factor: float | None = None,
    su2_converge: bool = False,
    su2_converge_kwargs: dict | None = None,
) -> dict:
    """Run the shared-CPACS pipeline end to end.

    Parameters
    ----------
    cpacs_path : str
        Path to the input CPACS XML file.
    mcps : list[str] | None
        Ordered list of MCP domains to run.  Defaults to all four.
    flight_conditions : dict | None
        Override flight conditions (mach, aoa, altitude_ft).
    mission_profile : dict | None
        Override mission parameters (weight_kg, range_m, segments, etc.).
    output_dir : str | None
        Directory for output files.  Defaults to ``./pipeline_output``.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict
        Pipeline summary with per-MCP results and version history.
    """
    if mcps is None:
        mcps = ["tigl", "su2", "pycycle", "nseg"]

    out = Path(output_dir or "pipeline_output")
    out.mkdir(parents=True, exist_ok=True)

    manager = CPACSManager()
    manager.load_file(cpacs_path)

    existing = _find_existing_artifacts(cpacs_path)
    if extra_artifacts:
        for k, v in extra_artifacts.items():
            if v:
                existing[k] = v

    if verbose:
        ref = manager.extract_reference_data()
        print(f"\n{'='*60}")
        print(f"  Shared-CPACS Pipeline (Real Solvers)")
        print(f"{'='*60}")
        print(f"  Source:     {cpacs_path}")
        print(f"  Aircraft:   {ref.get('name', 'N/A')}")
        print(f"  Ref area:   {ref.get('ref_area_m2', 'N/A')} m²")
        print(f"  Ref length: {ref.get('ref_length_m', 'N/A')} m")
        print(f"  MCPs:       {', '.join(mcps)}")
        print(f"  Output:     {out}")
        if existing["step_path"]:
            print(f"  STEP found: {existing['step_path']}")
        if existing["mesh_path"]:
            print(f"  Mesh found: {existing['mesh_path']}")
        print(f"{'='*60}\n")

    summaries: dict[str, dict] = {}
    total_start = time.time()

    # Track intermediate artifacts across MCP runs
    shared_artifacts: dict[str, Any] = {
        "step_bytes": None,
        "step_path": existing.get("step_path"),
        "mesh_path": existing.get("mesh_path"),
    }

    for i, domain in enumerate(mcps, 1):
        if verbose:
            print(f"[{i}/{len(mcps)}] Running {domain.upper()} adapter...")

        t0 = time.time()
        adapter = _import_adapter(domain)

        current_xml = manager.get_current_xml()

        if domain == "tigl":
            updated_xml, summary = adapter.run_adapter(
                current_xml,
                output_dir=str(out),
                existing_step_path=shared_artifacts.get("step_path"),
            )
            if summary.get("step_bytes"):
                shared_artifacts["step_bytes"] = summary["step_bytes"]
            if summary.get("step_path"):
                shared_artifacts["step_path"] = summary["step_path"]
            summary.pop("step_bytes", None)

        elif domain == "su2":
            su2_mesh_path = shared_artifacts.get("mesh_path")
            # Custom density / preset / converge mode all need a fresh
            # mesh, so drop any inherited cached mesh.
            if su2_preset or su2_surface_density is not None or su2_converge:
                su2_mesh_path = None

            if su2_converge:
                # Open-ended refinement: delegate to the deterministic
                # harness (which loops run_adapter internally) and slot
                # the final rung's results back into the per-MCP summary
                # so downstream stages (pyCycle, NSEG, Aviary) see a
                # converged CL/CD/L/D in CPACS.
                _converge_run_loop = _load_converge_harness()
                import argparse as _ap

                step_for_converge = (
                    shared_artifacts.get("step_path")
                    or (extra_artifacts or {}).get("step_path")
                )
                if not step_for_converge:
                    raise ValueError(
                        "--su2-converge requires a STEP file (pass --step "
                        "or run tigl first)."
                    )

                converge_defaults = {
                    "cpacs": cpacs_path,
                    "step": step_for_converge,
                    "mesh": None,
                    "output_root": str(out / "su2_converge"),
                    "start_density": 30,
                    "growth": 2.0,
                    "max_rungs": 5,
                    "max_wall_seconds": 7200,
                    "max_n_elem": 5_000_000,
                    "plateau_tol": 0.01,
                    "mach": (flight_conditions or {}).get("mach", 0.78),
                    "aoa": (flight_conditions or {}).get("aoa", 2.0),
                    "altitude": (flight_conditions or {}).get("altitude_ft", 35000.0),
                    "iter_cap": 800,
                    "cl_eps": 1e-4,
                    "per_rung_timeout": 7200,
                }
                converge_defaults.update(su2_converge_kwargs or {})
                converge_args = _ap.Namespace(**converge_defaults)
                converge_doc = _converge_run_loop(converge_args)

                final = converge_doc.get("final") or {}
                # Promote the converged rung into a real run_adapter call
                # against the produced mesh so the CPACS write-back path
                # is exercised identically to a preset run.
                final_mesh = None
                fd = final.get("output_dir")
                if fd:
                    mp = Path(fd) / "aircraft_volume.su2"
                    if mp.exists():
                        final_mesh = str(mp)
                updated_xml, summary = adapter.run_adapter(
                    current_xml,
                    flight_conditions=flight_conditions,
                    mesh_path=final_mesh,
                    output_dir=str(out / "su2_run"),
                    preset="industry",
                    iter_cap=int(converge_defaults["iter_cap"]),
                    cl_convergence_eps=float(converge_defaults["cl_eps"]),
                    wall_timeout_seconds=int(converge_defaults["per_rung_timeout"]),
                )
                summary["converge_history"] = converge_doc
            else:
                updated_xml, summary = adapter.run_adapter(
                    current_xml,
                    flight_conditions=flight_conditions,
                    step_bytes=shared_artifacts.get("step_bytes"),
                    step_path=shared_artifacts.get("step_path"),
                    mesh_path=su2_mesh_path,
                    output_dir=str(out / "su2_run"),
                    preset=su2_preset,
                    surface_density=su2_surface_density,
                    farfield_factor=su2_farfield_factor,
                )

        elif domain == "pycycle":
            updated_xml, summary = adapter.run_adapter(
                current_xml,
                flight_conditions=flight_conditions,
            )

        elif domain in ("nseg", "mission"):
            updated_xml, summary = adapter.run_adapter(
                current_xml,
                mission_profile=mission_profile,
            )

        elif domain == "aviary":
            updated_xml, summary = adapter.run_adapter(
                current_xml,
                mission_profile=mission_profile,
            )
        else:
            updated_xml, summary = adapter.run_adapter(current_xml)

        # Update the manager's tree in-place (preserving version history)
        from xml.etree import ElementTree as ET
        manager._root = ET.fromstring(updated_xml)
        manager._tree = ET.ElementTree(manager._root)
        manager.commit(author=f"{domain}-mcp", description=f"{domain.upper()} analysis complete")

        elapsed = time.time() - t0
        summary["_elapsed_s"] = round(elapsed, 3)
        summaries[domain] = summary

        if verbose:
            _print_summary(domain, summary, elapsed)

    total_elapsed = time.time() - total_start

    final_cpacs_path = out / "cpacs_final.xml"
    manager.save(final_cpacs_path)

    for v_idx in range(len(manager.version_history())):
        ver = manager.get_version(v_idx)
        ver_path = out / f"cpacs_v{ver.version_id}.xml"
        ver_path.write_text(ver.xml_string, encoding="utf-8")

    results_path = out / "pipeline_results.json"
    pipeline_result = {
        "source": cpacs_path,
        "mcps_run": mcps,
        "flight_conditions": flight_conditions,
        "mission_profile": mission_profile,
        "total_elapsed_s": round(total_elapsed, 3),
        "version_count": len(manager.version_history()),
        "versions": manager.version_history(),
        "summaries": _sanitize_for_json(summaries),
    }

    with open(results_path, "w") as f:
        json.dump(pipeline_result, f, indent=2, default=str)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Pipeline Complete")
        print(f"{'='*60}")
        print(f"  Versions created: {len(manager.version_history())}")
        print(f"  Total time:       {total_elapsed:.1f}s")
        print(f"  Final CPACS:      {final_cpacs_path}")
        print(f"  Results JSON:     {results_path}")
        print(f"{'='*60}\n")

    return pipeline_result


def _sanitize_for_json(obj):
    """Recursively convert non-serializable values."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        return obj
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    return obj


def _print_summary(domain: str, summary: dict, elapsed: float) -> None:
    """Print a concise summary for a single MCP run."""
    if summary.get("error"):
        err = summary["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        print(f"      ERROR ({elapsed:.2f}s): {msg}")
        return

    print(f"      Done in {elapsed:.2f}s")

    if domain == "tigl":
        print(f"      Wings: {summary.get('wing_count', '?')}, "
              f"Fuselages: {summary.get('fuselage_count', '?')}, "
              f"Components: {len(summary.get('components', []))}")
        print(f"      STEP export: {summary.get('step_source', 'N/A')}")

    elif domain == "su2":
        if summary.get("converged"):
            print(f"      CL={summary.get('CL', '?')}, CD={summary.get('CD', '?')}, "
                  f"L/D={summary.get('L_over_D', '?')}")
            if summary.get("runtime_seconds"):
                print(f"      SU2 runtime: {summary['runtime_seconds']:.1f}s")
        else:
            print(f"      Mesh source: {summary.get('mesh_source', 'N/A')}")

    elif domain == "pycycle":
        if not summary.get("error"):
            print(f"      TSFC={summary.get('TSFC_lb_lbf_hr', '?')} lb/(lbf·hr), "
                  f"Fn={summary.get('Fn_N', '?')} N, "
                  f"OPR={summary.get('OPR', '?')}, BPR={summary.get('BPR', '?')}")

    elif domain in ("nseg", "mission"):
        if summary.get("success"):
            fbk = summary.get('total_fuel_burned_kg') or summary.get('fuel_burned_kg', 0)
            if fbk:
                print(f"      Fuel burned: {fbk:.1f} kg")
            dnm = summary.get('total_distance_nm', 0)
            thr = summary.get('total_time_hr', 0)
            ff = summary.get('fuel_fraction', 0)
            print(f"      Range: {dnm:.1f} nm, Time: {thr:.2f} hr, Fuel fraction: {ff:.3f}")

    elif domain == "aviary":
        if summary.get("success"):
            fbk = summary.get('total_fuel_burned_kg') or summary.get('fuel_burned_kg')
            gtow = summary.get('gtow_kg')
            wm = summary.get('wing_mass_kg')
            conv = summary.get('converged', False)
            rt = summary.get('runtime_seconds')
            if fbk:
                print(f"      Fuel burned: {fbk:.1f} kg")
            if gtow:
                print(f"      GTOW: {gtow:.1f} kg")
            if wm:
                print(f"      Wing mass: {wm:.1f} kg")
            print(f"      Converged: {conv}")
            if rt:
                print(f"      Runtime: {rt:.1f}s")
        elif summary.get("error"):
            err = summary["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            print(f"      Aviary error: {msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Shared-CPACS MCP Pipeline Orchestrator (Real Solvers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("cpacs", help="Path to CPACS XML file")
    parser.add_argument("--mcps", nargs="+", default=None,
                        choices=["tigl", "su2", "pycycle", "nseg", "aviary", "mission"],
                        help=("MCPs to run (default: tigl su2 pycycle nseg). "
                              "Pick exactly one mission MCP per run -- 'nseg' (fast) or "
                              "'aviary' (trajectory-coupled). 'mission' is legacy."))
    parser.add_argument("--mach", type=float, default=None)
    parser.add_argument("--aoa", type=float, default=None)
    parser.add_argument("--altitude", type=float, default=None, help="Altitude in feet")
    parser.add_argument("--weight", type=float, default=None, help="Takeoff weight in kg")
    parser.add_argument("--range", type=float, default=None, dest="range_m",
                        help="Cruise range in metres")
    parser.add_argument("--step", default=None, help="Path to existing STEP file for SU2 meshing")
    parser.add_argument("--mesh", default=None, help="Path to existing .su2 mesh file")
    parser.add_argument("--su2-preset", default=None,
                        choices=["laptop", "workstation", "industry"],
                        help=("SU2 mesh/iteration fidelity preset. When set, any existing "
                              "mesh is ignored and a fresh mesh is generated at that density."))
    parser.add_argument("--su2-density", type=int, default=None, dest="su2_density",
                        help=("Open-ended SU2 surface_density override (any positive integer). "
                              "Overrides the preset's mesh density only; iter/timeout still "
                              "follow the preset. Forces a fresh mesh."))
    parser.add_argument("--su2-farfield-factor", type=float, default=None,
                        dest="su2_farfield_factor",
                        help="Optional override of the Gmsh farfield-box / span ratio.")
    parser.add_argument("--su2-converge", action="store_true", dest="su2_converge",
                        help=("Run the open-ended SU2 mesh-refinement loop "
                              "(scripts/run_converged_su2.py) before the rest of the pipeline. "
                              "Returns a converged CL/CD/L/D and records the rung table in the "
                              "pipeline summary."))
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    fc: dict | None = None
    if any(v is not None for v in [args.mach, args.aoa, args.altitude]):
        fc = {}
        if args.mach is not None:
            fc["mach"] = args.mach
        if args.aoa is not None:
            fc["aoa"] = args.aoa
        if args.altitude is not None:
            fc["altitude_ft"] = args.altitude

    mp: dict | None = None
    if any(v is not None for v in [args.weight, args.range_m]):
        mp = {}
        if args.weight is not None:
            mp["weight_kg"] = args.weight
        if args.range_m is not None:
            mp["range_m"] = args.range_m

    # Ensure SU2 is on PATH
    su2_bin = Path.home() / ".local" / "su2" / "bin"
    if su2_bin.is_dir() and str(su2_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{su2_bin}:{os.environ.get('PATH', '')}"

    extra_artifacts = {}
    if args.step:
        extra_artifacts["step_path"] = args.step
    if args.mesh:
        extra_artifacts["mesh_path"] = args.mesh

    run_pipeline(
        cpacs_path=args.cpacs,
        mcps=args.mcps,
        flight_conditions=fc,
        mission_profile=mp,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        extra_artifacts=extra_artifacts,
        su2_preset=args.su2_preset,
        su2_surface_density=args.su2_density,
        su2_farfield_factor=args.su2_farfield_factor,
        su2_converge=args.su2_converge,
    )


if __name__ == "__main__":
    main()
