"""Unit tests for the open-ended SU2 refinement harness.

The real SU2 binary is *not* required: we monkeypatch
``su2_mcp.cpacs_adapter.run_adapter`` with a deterministic stub so we
can exercise the loop logic (plateau detection, cell-count cap, wall
budget) end-to-end. This is the "adapter logic test" exemption in the
no-stubs rule: the production adapter path still calls the real solver,
but here we are testing the harness *around* it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

_HARNESS_PATH = Path(__file__).resolve().parent.parent / "run_converged_su2.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("run_converged_su2", _HARNESS_PATH)
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


def _make_fake_adapter(answers):
    """Return a stub ``run_adapter`` that yields the prepared records."""
    seq = iter(answers)

    def fake_run_adapter(xml, **kwargs):  # type: ignore[no-untyped-def]
        summary = next(seq)
        summary = dict(summary)
        summary.setdefault("output_dir", kwargs.get("output_dir"))
        # Echo the requested density so the harness records it.
        summary.setdefault("preset_label", "test-fake")
        return xml, summary

    return fake_run_adapter


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    cpacs = tmp_path / "fake.xml"
    cpacs.write_text(_CPACS, encoding="utf-8")
    step = tmp_path / "fake.step"
    step.write_text("ISO-10303-21;\nENDSEC;\nEND-ISO-10303-21;\n", encoding="utf-8")
    return cpacs, step


def test_plateau_triggers_on_second_rung(tmp_path, monkeypatch):
    harness = _load_harness()
    cpacs, step = _write_inputs(tmp_path)

    # Two rungs with CL/CD identical and Cauchy on -> plateau on rung 2.
    answers = [
        {"CL": 0.3000, "CD": 0.0200, "L_over_D": 15.0, "mesh_n_elem": 50_000, "cauchy_triggered": True, "runtime_seconds": 5.0},
        {"CL": 0.3010, "CD": 0.0201, "L_over_D": 15.0, "mesh_n_elem": 100_000, "cauchy_triggered": True, "runtime_seconds": 12.0},
    ]
    fake_module = type(sys)("su2_mcp.cpacs_adapter")
    fake_module.run_adapter = _make_fake_adapter(answers)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "su2_mcp.cpacs_adapter", fake_module)
    fake_pkg = type(sys)("su2_mcp")
    fake_pkg.cpacs_adapter = fake_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "su2_mcp", fake_pkg)

    args = harness.parse_args([
        "--cpacs", str(cpacs),
        "--step", str(step),
        "--output-root", str(tmp_path / "out"),
        "--start-density", "30",
        "--growth", "2.0",
        "--max-rungs", "4",
        "--max-wall-seconds", "60",
        "--max-n-elem", "10000000",
        "--plateau-tol", "0.01",
    ])
    doc = harness.run_loop(args)
    assert doc["status"] == "plateaued"
    assert doc["stop_reason"] == "outer_plateau_and_inner_cauchy"
    assert len(doc["history"]) == 2
    out_json = tmp_path / "out" / "convergence_history.json"
    assert out_json.exists()
    assert json.loads(out_json.read_text())["status"] == "plateaued"


def test_plateau_requires_cauchy(tmp_path, monkeypatch):
    """Even with CL/CD identical, plateau must NOT fire when Cauchy didn't trigger."""
    harness = _load_harness()
    cpacs, step = _write_inputs(tmp_path)

    answers = [
        {"CL": 0.30, "CD": 0.02, "mesh_n_elem": 50_000, "cauchy_triggered": False, "runtime_seconds": 5.0},
        {"CL": 0.30, "CD": 0.02, "mesh_n_elem": 100_000, "cauchy_triggered": False, "runtime_seconds": 6.0},
        {"CL": 0.30, "CD": 0.02, "mesh_n_elem": 200_000, "cauchy_triggered": False, "runtime_seconds": 7.0},
    ]
    fake_module = type(sys)("su2_mcp.cpacs_adapter")
    fake_module.run_adapter = _make_fake_adapter(answers)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "su2_mcp.cpacs_adapter", fake_module)
    fake_pkg = type(sys)("su2_mcp")
    fake_pkg.cpacs_adapter = fake_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "su2_mcp", fake_pkg)

    args = harness.parse_args([
        "--cpacs", str(cpacs),
        "--step", str(step),
        "--output-root", str(tmp_path / "out"),
        "--max-rungs", "3",
    ])
    doc = harness.run_loop(args)
    assert doc["status"] == "budget_exhausted"
    assert doc["stop_reason"] == "max_rungs"


def test_cell_count_cap_stops_loop(tmp_path, monkeypatch):
    harness = _load_harness()
    cpacs, step = _write_inputs(tmp_path)

    # Rung 1 has 1M cells; growth=2.0 -> next projected = 1M * 2^3 = 8M > 5M cap.
    answers = [
        {"CL": 0.3, "CD": 0.02, "mesh_n_elem": 1_000_000, "cauchy_triggered": True, "runtime_seconds": 100.0},
    ]
    fake_module = type(sys)("su2_mcp.cpacs_adapter")
    fake_module.run_adapter = _make_fake_adapter(answers)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "su2_mcp.cpacs_adapter", fake_module)
    fake_pkg = type(sys)("su2_mcp")
    fake_pkg.cpacs_adapter = fake_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "su2_mcp", fake_pkg)

    args = harness.parse_args([
        "--cpacs", str(cpacs),
        "--step", str(step),
        "--output-root", str(tmp_path / "out"),
        "--max-rungs", "5",
        "--growth", "2.0",
        "--max-n-elem", "5000000",
    ])
    doc = harness.run_loop(args)
    assert doc["status"] == "budget_exhausted"
    assert doc["stop_reason"] == "max_n_elem_projection"
    assert len(doc["history"]) == 1


def test_missing_inputs_rejected(tmp_path):
    harness = _load_harness()
    args = harness.parse_args([
        "--cpacs", str(tmp_path / "nope.xml"),
        "--step", str(tmp_path / "also_nope.step"),
        "--output-root", str(tmp_path / "out"),
    ])
    with pytest.raises(FileNotFoundError):
        harness.run_loop(args)


def test_next_density_strictly_increases():
    harness = _load_harness()
    assert harness._next_density(30, 2.0) == 60
    assert harness._next_density(60, 1.5) == 90
    # Pathological growth=1.0 should still strictly increase.
    assert harness._next_density(50, 1.0) == 60


def test_projected_n_elem_uses_cubic_scaling():
    harness = _load_harness()
    hist = [{"mesh_n_elem": 100_000, "surface_density": 30}]
    assert harness._projected_n_elem(hist, 60) == int(100_000 * (2 ** 3))
    assert harness._projected_n_elem([], 60) is None
    assert harness._projected_n_elem([{"mesh_n_elem": None, "surface_density": 30}], 60) is None
