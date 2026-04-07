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
for sub in ("tigl-mcp/src", "su2-mcp/src", "pycycle-mcp/src", "mission-mcp/src"):
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
    elif domain == "mission":
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
        mcps = ["tigl", "su2", "pycycle", "mission"]

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
            updated_xml, summary = adapter.run_adapter(
                current_xml,
                flight_conditions=flight_conditions,
                step_bytes=shared_artifacts.get("step_bytes"),
                step_path=shared_artifacts.get("step_path"),
                mesh_path=shared_artifacts.get("mesh_path"),
                output_dir=str(out / "su2_run"),
            )

        elif domain == "pycycle":
            updated_xml, summary = adapter.run_adapter(
                current_xml,
                flight_conditions=flight_conditions,
            )

        elif domain == "mission":
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

    elif domain == "mission":
        if summary.get("success"):
            backend = summary.get("backend", "nseg")
            fbk = summary.get('total_fuel_burned_kg') or summary.get('fuel_burned_kg', 0)
            print(f"      Backend: {backend}")
            if fbk:
                print(f"      Fuel burned: {fbk:.1f} kg")
            if backend == "aviary":
                gtow = summary.get('gtow_kg')
                if gtow:
                    print(f"      GTOW: {gtow:.1f} kg")
                conv = summary.get('converged', False)
                print(f"      Converged: {conv}")
                rt = summary.get('runtime_seconds')
                if rt:
                    print(f"      Runtime: {rt:.1f}s")
            else:
                dnm = summary.get('total_distance_nm', 0)
                thr = summary.get('total_time_hr', 0)
                print(f"      Range: {dnm:.1f} nm, Time: {thr:.2f} hr")


def main():
    parser = argparse.ArgumentParser(
        description="Shared-CPACS MCP Pipeline Orchestrator (Real Solvers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("cpacs", help="Path to CPACS XML file")
    parser.add_argument("--mcps", nargs="+", default=None,
                        choices=["tigl", "su2", "pycycle", "mission"],
                        help="MCPs to run (default: all four in order)")
    parser.add_argument("--mach", type=float, default=None)
    parser.add_argument("--aoa", type=float, default=None)
    parser.add_argument("--altitude", type=float, default=None, help="Altitude in feet")
    parser.add_argument("--weight", type=float, default=None, help="Takeoff weight in kg")
    parser.add_argument("--range", type=float, default=None, dest="range_m",
                        help="Cruise range in metres")
    parser.add_argument("--step", default=None, help="Path to existing STEP file for SU2 meshing")
    parser.add_argument("--mesh", default=None, help="Path to existing .su2 mesh file")
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
    )


if __name__ == "__main__":
    main()
