#!/usr/bin/env python3
"""Wetted area from the surface patches of an SU2 mesh.

Ron Engelbeck, 2026-09 (email): "It should be a fairly simple matter to ...
sum up the surface area of the surface patches of your CFD mesh." This does
exactly that and nothing else: it reads the boundary-marker elements of a
native ``.su2`` mesh, sums the areas of the triangles and quadrilaterals on
the wall markers, and reports the result together with everything needed to
judge it (element counts, bounding box for the unit sanity check, per-marker
areas). No smoothing, no correction factors. The number is the area of the
faceted surface the CFD actually ran on; on a coarse mesh that is a lower
bound on the true smooth-surface area, and the way to see how far off it is
is to run this on meshes of increasing density.

Usage:
    wetted_area_from_su2.py MESH.su2 [--wall WALL] [--half-model] [--json]

``--half-model`` doubles the area for a mesh that carries only one side of a
symmetric aircraft. It is never assumed: a full-span mesh with no symmetry
marker is the default, and the bounding box in the report shows which case
you have.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

TRIANGLE = 5
QUADRILATERAL = 9
SURFACE_TYPES = {TRIANGLE: 3, QUADRILATERAL: 4}


def _tri_area(a, b, c) -> float:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def read_su2_surface(path: Path) -> dict:
    """Return points and the boundary elements of every marker.

    Only the parts of the file this needs are parsed: NDIME, NPOIN and the
    NMARK block. The volume elements are skipped by count, not by reading them.
    """
    points: list[tuple[float, float, float]] = []
    markers: dict[str, list[tuple[int, ...]]] = {}
    ndime = None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        line_iter = iter(fh)
        for raw in line_iter:
            line = raw.split("%", 1)[0].strip()
            if not line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key == "NDIME":
                ndime = int(val)
                if ndime != 3:
                    raise ValueError(f"{path}: NDIME={ndime}; only 3-D meshes have a wetted area")
            elif key == "NELEM":
                n = int(val)
                skipped = 0
                while skipped < n:
                    if next(line_iter).strip():
                        skipped += 1
            elif key == "NPOIN":
                n = int(val)
                got = 0
                while got < n:
                    parts = next(line_iter).split()
                    if not parts:
                        continue
                    points.append((float(parts[0]), float(parts[1]), float(parts[2])))
                    got += 1
            elif key == "MARKER_TAG":
                tag = val.strip()
                nxt = next(line_iter)
                k2, _, v2 = nxt.partition("=")
                if k2.strip() != "MARKER_ELEMS":
                    raise ValueError(f"{path}: MARKER_TAG {tag} not followed by MARKER_ELEMS")
                m = int(v2)
                elems: list[tuple[int, ...]] = []
                while len(elems) < m:
                    parts = next(line_iter).split()
                    if not parts:
                        continue
                    etype = int(parts[0])
                    nn = SURFACE_TYPES.get(etype)
                    if nn is None:
                        raise ValueError(f"{path}: marker {tag} has element type {etype}; only triangles (5) and quads (9) are surface elements")
                    elems.append(tuple(int(x) for x in parts[1 : 1 + nn]))
                markers[tag] = elems
    if ndime is None or not points or not markers:
        raise ValueError(f"{path}: missing NDIME, NPOIN or NMARK block")
    return {"points": points, "markers": markers}


def marker_area(points, elems) -> float:
    total = 0.0
    for e in elems:
        p = [points[i] for i in e]
        if len(p) == 3:
            total += _tri_area(*p)
        else:  # quad: split along one diagonal; exact for planar quads
            total += _tri_area(p[0], p[1], p[2]) + _tri_area(p[0], p[2], p[3])
    return total


def wetted_area(path: Path, wall_markers: list[str], half_model: bool) -> dict:
    mesh = read_su2_surface(path)
    pts = mesh["points"]
    per_marker = {tag: marker_area(pts, el) for tag, el in mesh["markers"].items()}
    missing = [w for w in wall_markers if w not in per_marker]
    if missing:
        raise KeyError(f"{path}: wall marker(s) {missing} not in mesh; markers present: {sorted(per_marker)}")
    wall_pts = {i for w in wall_markers for e in mesh["markers"][w] for i in e}
    xs = [pts[i][0] for i in wall_pts]
    ys = [pts[i][1] for i in wall_pts]
    zs = [pts[i][2] for i in wall_pts]
    area = sum(per_marker[w] for w in wall_markers)
    return {
        "mesh": str(path),
        "wall_markers": wall_markers,
        "wall_elements": sum(len(mesh["markers"][w]) for w in wall_markers),
        "wall_points": len(wall_pts),
        "wall_bbox_min": [min(xs), min(ys), min(zs)],
        "wall_bbox_max": [max(xs), max(ys), max(zs)],
        "half_model": half_model,
        "per_marker_area": per_marker,
        "wetted_area_mesh_units_sq": area * (2.0 if half_model else 1.0),
        "note": "area of the faceted wall surface as meshed; a lower bound on the smooth surface, tightening with mesh density",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mesh", type=Path)
    p.add_argument("--wall", action="append", default=None, help="wall marker name (repeatable). Default: WALL")
    p.add_argument("--half-model", action="store_true", help="mesh carries one side only; double the area")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    a = p.parse_args(argv)
    if not a.mesh.exists():
        print(json.dumps({"error": {"type": "missing_input", "message": f"mesh not found: {a.mesh}"}}))
        return 2
    try:
        r = wetted_area(a.mesh, a.wall or ["WALL"], a.half_model)
    except (ValueError, KeyError) as exc:
        print(json.dumps({"error": {"type": "bad_mesh", "message": str(exc)}}))
        return 1
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        bb = [hi - lo for lo, hi in zip(r["wall_bbox_min"], r["wall_bbox_max"])]
        print(f"{r['mesh']}")
        print(f"  wall elements {r['wall_elements']:,}  points {r['wall_points']:,}  extent (x,y,z) = {bb[0]:.2f} x {bb[1]:.2f} x {bb[2]:.2f} mesh units")
        for tag, ar in r["per_marker_area"].items():
            print(f"  marker {tag:<10} area {ar:,.2f}")
        print(f"  wetted area ({'half model x2' if r['half_model'] else 'full model'}): {r['wetted_area_mesh_units_sq']:,.2f} mesh-units^2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
