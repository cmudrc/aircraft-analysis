"""Tests for scripts/wetted_area_from_su2.py on a unit cube (area 6)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "wetted_area_from_su2.py"


def _load():
    spec = importlib.util.spec_from_file_location("wetted_area_from_su2", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


PTS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
QUADS = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def _mesh(quads: bool = False, wall: str = "WALL") -> str:
    faces = QUADS if quads else [t for q in QUADS for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
    code = 9 if quads else 5
    lines = ["NDIME= 3", "NELEM= 1", "10 0 1 2 4 0", f"NPOIN= {len(PTS)}"]
    lines += [f"{x} {y} {z} {i}" for i, (x, y, z) in enumerate(PTS)]
    lines += ["NMARK= 1", f"MARKER_TAG= {wall}", f"MARKER_ELEMS= {len(faces)}"]
    lines += [f"{code} " + " ".join(map(str, f)) for f in faces]
    return "\n".join(lines) + "\n"


def test_cube_triangles(tmp_path: Path) -> None:
    mod = _load()
    m = tmp_path / "c.su2"
    m.write_text(_mesh())
    r = mod.wetted_area(m, ["WALL"], half_model=False)
    assert r["wetted_area_mesh_units_sq"] == pytest.approx(6.0)
    assert r["wall_elements"] == 12
    assert r["wall_bbox_max"] == [1.0, 1.0, 1.0]


def test_cube_quads_and_half_model(tmp_path: Path) -> None:
    mod = _load()
    m = tmp_path / "q.su2"
    m.write_text(_mesh(quads=True))
    assert mod.wetted_area(m, ["WALL"], False)["wetted_area_mesh_units_sq"] == pytest.approx(6.0)
    assert mod.wetted_area(m, ["WALL"], True)["wetted_area_mesh_units_sq"] == pytest.approx(12.0)


def test_missing_marker_is_a_structured_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load()
    m = tmp_path / "n.su2"
    m.write_text(_mesh(wall="BODY"))
    assert mod.main([str(m)]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["type"] == "bad_mesh" and "BODY" in err["message"]
    assert mod.main([str(tmp_path / "nope.su2")]) == 2
