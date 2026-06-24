"""Unit tests for the AoA sweep / trim harness.

The real SU2 binary is *not* required: we monkeypatch
``su2_mcp.cpacs_adapter.run_adapter`` with a deterministic stub so we can
exercise the sweep logic (angle resolution, best-L/D selection, trim
interpolation, budget handling) end-to-end. This is the "adapter logic
test" exemption in the no-stubs rule: the production adapter path still
calls the real solver; here we test the harness *around* it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_HARNESS_PATH = Path(__file__).resolve().parent.parent / "run_aoa_sweep.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("run_aoa_sweep", _HARNESS_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CPACS = (
    "<?xml version='1.0'?>"
    "<cpacs><vehicles><aircraft><model>"
    "<reference><area>122.4</area><length>4.2</length></reference>"
    "</model></aircraft></vehicles></cpacs>"
)


def _make_fake_adapter(by_aoa):
    """Return a stub ``run_adapter`` keyed on the requested AoA."""

    def fake_run_adapter(xml, **kwargs):  # type: ignore[no-untyped-def]
        aoa = kwargs["flight_conditions"]["aoa"]
        summary = dict(by_aoa[aoa])
        summary.setdefault("output_dir", kwargs.get("output_dir"))
        return xml, summary

    return fake_run_adapter


def _install_fake(monkeypatch, by_aoa):
    fake_module = type(sys)("su2_mcp.cpacs_adapter")
    fake_module.run_adapter = _make_fake_adapter(by_aoa)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "su2_mcp.cpacs_adapter", fake_module)
    fake_pkg = type(sys)("su2_mcp")
    fake_pkg.cpacs_adapter = fake_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "su2_mcp", fake_pkg)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    cpacs = tmp_path / "fake.xml"
    cpacs.write_text(_CPACS, encoding="utf-8")
    mesh = tmp_path / "fake.su2"
    mesh.write_text("NDIME= 3\nNELEM= 0\n", encoding="utf-8")
    return cpacs, mesh


# --------------------------------------------------------------------------
# Pure helper tests
# --------------------------------------------------------------------------

def test_aoa_values_from_list():
    h = _load_harness()
    args = h.parse_args(["--cpacs", "x", "--mesh", "m", "--aoa-list", "4,0,2,2,1"])
    # De-duplicated and sorted ascending.
    assert h.aoa_values(args) == [0.0, 1.0, 2.0, 4.0]


def test_aoa_values_from_range_includes_endpoint():
    h = _load_harness()
    args = h.parse_args(
        ["--cpacs", "x", "--mesh", "m", "--aoa-min", "0", "--aoa-max", "3", "--aoa-step", "1"]
    )
    assert h.aoa_values(args) == [0.0, 1.0, 2.0, 3.0]


def test_aoa_values_requires_a_spec():
    h = _load_harness()
    args = h.parse_args(["--cpacs", "x", "--mesh", "m"])
    with pytest.raises(ValueError):
        h.aoa_values(args)


def test_best_ld_ignores_errored_points():
    h = _load_harness()
    pts = [
        {"aoa": 0.0, "L_over_D": 10.0, "error": None},
        {"aoa": 1.0, "L_over_D": 99.0, "error": {"type": "boom"}},  # excluded
        {"aoa": 2.0, "L_over_D": 16.0, "error": None},
    ]
    assert h.best_ld(pts)["aoa"] == 2.0
    assert h.best_ld([]) is None


def test_interp_trim_brackets_and_interpolates():
    h = _load_harness()
    pts = [
        {"aoa": 2.0, "CL": 0.40, "CD": 0.020, "error": None},
        {"aoa": 4.0, "CL": 0.60, "CD": 0.040, "error": None},
    ]
    trim = h.interp_trim(pts, 0.50)
    assert trim is not None
    assert trim["aoa_trim_deg"] == pytest.approx(3.0)
    assert trim["cd_trim"] == pytest.approx(0.030)
    assert trim["ld_trim"] == pytest.approx(0.50 / 0.030, abs=1e-3)


def test_interp_trim_returns_none_when_not_bracketed():
    h = _load_harness()
    pts = [
        {"aoa": 0.0, "CL": 0.10, "CD": 0.01, "error": None},
        {"aoa": 2.0, "CL": 0.30, "CD": 0.02, "error": None},
    ]
    assert h.interp_trim(pts, 0.9) is None  # target above swept range


# --------------------------------------------------------------------------
# Integration tests through run_loop with a fake adapter
# --------------------------------------------------------------------------

def test_sweep_reports_best_ld_and_trim(tmp_path, monkeypatch):
    h = _load_harness()
    cpacs, mesh = _write_inputs(tmp_path)
    by_aoa = {
        0.0: {"CL": 0.10, "CD": 0.012, "L_over_D": 8.3, "cauchy_triggered": True, "runtime_seconds": 5.0},
        2.0: {"CL": 0.40, "CD": 0.020, "L_over_D": 20.0, "cauchy_triggered": True, "runtime_seconds": 6.0},
        4.0: {"CL": 0.60, "CD": 0.040, "L_over_D": 15.0, "cauchy_triggered": True, "runtime_seconds": 7.0},
    }
    _install_fake(monkeypatch, by_aoa)

    args = h.parse_args([
        "--cpacs", str(cpacs),
        "--mesh", str(mesh),
        "--output-root", str(tmp_path / "out"),
        "--aoa-list", "0,2,4",
        "--target-cl", "0.5",
    ])
    doc = h.run_loop(args)
    assert doc["status"] == "completed"
    assert doc["n_success"] == 3
    assert doc["best_ld"]["aoa"] == 2.0
    assert doc["trim"]["aoa_trim_deg"] == pytest.approx(3.0)
    out_json = tmp_path / "out" / "aoa_sweep_history.json"
    assert out_json.exists()
    assert json.loads(out_json.read_text())["best_ld"]["aoa"] == 2.0


def test_sweep_survives_single_point_failure(tmp_path, monkeypatch):
    h = _load_harness()
    cpacs, mesh = _write_inputs(tmp_path)
    by_aoa = {
        0.0: {"CL": 0.10, "CD": 0.012, "L_over_D": 8.3, "runtime_seconds": 5.0},
        2.0: {"error": {"type": "su2_failure", "message": "diverged"}},
        4.0: {"CL": 0.60, "CD": 0.040, "L_over_D": 15.0, "runtime_seconds": 7.0},
    }
    _install_fake(monkeypatch, by_aoa)

    args = h.parse_args([
        "--cpacs", str(cpacs),
        "--mesh", str(mesh),
        "--output-root", str(tmp_path / "out"),
        "--aoa-list", "0,2,4",
    ])
    doc = h.run_loop(args)
    assert doc["status"] == "completed"
    assert doc["n_success"] == 2
    assert doc["best_ld"]["aoa"] == 4.0


def test_sweep_all_fail_is_error(tmp_path, monkeypatch):
    h = _load_harness()
    cpacs, mesh = _write_inputs(tmp_path)
    by_aoa = {
        0.0: {"error": {"type": "su2_failure", "message": "boom"}},
        2.0: {"error": {"type": "su2_failure", "message": "boom"}},
    }
    _install_fake(monkeypatch, by_aoa)

    args = h.parse_args([
        "--cpacs", str(cpacs),
        "--mesh", str(mesh),
        "--output-root", str(tmp_path / "out"),
        "--aoa-list", "0,2",
    ])
    doc = h.run_loop(args)
    assert doc["status"] == "error"
    assert doc["stop_reason"] == "no_successful_points"


def test_missing_inputs_rejected(tmp_path):
    h = _load_harness()
    args = h.parse_args([
        "--cpacs", str(tmp_path / "nope.xml"),
        "--mesh", str(tmp_path / "nope.su2"),
        "--output-root", str(tmp_path / "out"),
        "--aoa-list", "0,2",
    ])
    with pytest.raises(FileNotFoundError):
        h.run_loop(args)


def test_missing_mesh_and_step_rejected(tmp_path):
    h = _load_harness()
    cpacs = tmp_path / "fake.xml"
    cpacs.write_text(_CPACS, encoding="utf-8")
    args = h.parse_args([
        "--cpacs", str(cpacs),
        "--output-root", str(tmp_path / "out"),
        "--aoa-list", "0,2",
    ])
    with pytest.raises(ValueError):
        h.run_loop(args)
