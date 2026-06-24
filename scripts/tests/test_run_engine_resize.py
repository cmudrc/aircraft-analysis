"""Unit tests for the engine-resize harness.

Neither OpenMDAO/pyCycle nor NSEG's real solver is required: we monkeypatch
both ``pycycle_mcp.cpacs_adapter.run_adapter`` and
``nseg_mcp.cpacs_adapter.run_adapter`` with deterministic stubs that emulate
the *relationship* the real solvers obey (pyCycle design mode sizes the cycle
so achieved net thrust ``Fn`` equals the requested ``Fn_DES``; NSEG reports a
top-of-climb thrust margin = available − required). This is the "adapter logic
test" exemption in the no-stubs rule: the production path still calls the real
solvers; here we test the Newton resize loop *around* them.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_HARNESS_PATH = Path(__file__).resolve().parent.parent / "run_engine_resize.py"

LBF_TO_N = 4.44822


def _load_harness():
    spec = importlib.util.spec_from_file_location("run_engine_resize", _HARNESS_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CPACS = (
    "<?xml version='1.0'?>"
    "<cpacs><vehicles><aircraft><model>"
    "<reference><area>122.4</area></reference>"
    "</model></aircraft></vehicles></cpacs>"
)


def _install_fakes(monkeypatch, *, t_req_n, pyc_error=None, drop_thrust_closure=False,
                   default_start_lbf=12000.0, fuel_kg=8000.0):
    """Wire fake pyCycle + NSEG modules into sys.modules.

    The fakes emulate: Fn_N == Fn_DES (design mode) and
    margin == Fn_N - t_req_n (top-of-climb closure).
    """

    def fake_pyc(xml, flight_conditions=None, design_thrust_lbf=None):
        if pyc_error is not None:
            return xml, {"error": pyc_error, "solver": "pycycle_openmdao"}
        fn_des = float(design_thrust_lbf) if design_thrust_lbf is not None else default_start_lbf
        fn_n = fn_des * LBF_TO_N
        eng = {
            "Fn_DES_lbf": round(fn_des, 2),
            "Fn_N": round(fn_n, 2),
            "Fn_lbf": round(fn_des, 2),
            "TSFC_1_per_s": 1.7e-5,
        }
        return xml + f"<fnmark>{fn_n}</fnmark>", eng

    def fake_nseg(xml, mission_profile=None):
        m = re.search(r"<fnmark>([0-9.eE+-]+)</fnmark>", xml)
        fn_n = float(m.group(1)) if m else 0.0
        result = {"success": True, "total_fuel_burned_kg": fuel_kg}
        if not drop_thrust_closure:
            margin = fn_n - t_req_n
            result["thrust_closure"] = {
                "thrust_required_n": t_req_n,
                "thrust_margin_n": margin,
                "thrust_limited": margin < 0,
            }
            result["thrust_limited"] = margin < 0
        return xml, result

    pyc_mod = type(sys)("pycycle_mcp.cpacs_adapter")
    pyc_mod.run_adapter = fake_pyc  # type: ignore[attr-defined]
    pyc_pkg = type(sys)("pycycle_mcp")
    pyc_pkg.cpacs_adapter = pyc_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pycycle_mcp", pyc_pkg)
    monkeypatch.setitem(sys.modules, "pycycle_mcp.cpacs_adapter", pyc_mod)

    nseg_mod = type(sys)("nseg_mcp.cpacs_adapter")
    nseg_mod.run_adapter = fake_nseg  # type: ignore[attr-defined]
    nseg_pkg = type(sys)("nseg_mcp")
    nseg_pkg.cpacs_adapter = nseg_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nseg_mcp", nseg_pkg)
    monkeypatch.setitem(sys.modules, "nseg_mcp.cpacs_adapter", nseg_mod)


def _write_cpacs(tmp_path: Path) -> Path:
    cpacs = tmp_path / "fake.xml"
    cpacs.write_text(_CPACS, encoding="utf-8")
    return cpacs


# --------------------------------------------------------------------------
# Pure helper tests
# --------------------------------------------------------------------------

def test_newton_step_moves_toward_target():
    h = _load_harness()
    # margin below target -> design thrust should increase.
    nxt = h.newton_step(10000.0, margin_n=-5000.0, target_n=0.0,
                        gain=1.0, min_lbf=1000.0, max_lbf=60000.0)
    assert nxt > 10000.0
    # margin above target -> design thrust should decrease.
    nxt2 = h.newton_step(10000.0, margin_n=5000.0, target_n=0.0,
                         gain=1.0, min_lbf=1000.0, max_lbf=60000.0)
    assert nxt2 < 10000.0


def test_newton_step_clamps_to_bounds():
    h = _load_harness()
    hi = h.newton_step(59000.0, margin_n=-1e7, target_n=0.0,
                       gain=1.0, min_lbf=1000.0, max_lbf=60000.0)
    assert hi == 60000.0
    lo = h.newton_step(2000.0, margin_n=1e7, target_n=0.0,
                       gain=1.0, min_lbf=1000.0, max_lbf=60000.0)
    assert lo == 1000.0


# --------------------------------------------------------------------------
# Integration tests through run_loop with fake adapters
# --------------------------------------------------------------------------

def test_resize_converges_to_zero_margin(tmp_path, monkeypatch):
    h = _load_harness()
    cpacs = _write_cpacs(tmp_path)
    t_req = 80000.0  # N
    _install_fakes(monkeypatch, t_req_n=t_req)

    args = h.parse_args([
        "--cpacs", str(cpacs),
        "--output-root", str(tmp_path / "out"),
        "--target-margin-frac", "0.0",
        "--tol-n", "100",
    ])
    doc = h.run_loop(args)
    assert doc["status"] == "converged"
    conv = doc["converged"]
    # Engine sized so installed thrust ~ required thrust (margin ~ 0).
    assert conv["Fn_N"] == pytest.approx(t_req, abs=200.0)
    assert abs(conv["thrust_margin_n"]) <= 200.0
    out_json = tmp_path / "out" / "engine_resize_history.json"
    assert out_json.exists()
    assert json.loads(out_json.read_text())["status"] == "converged"


def test_resize_hits_positive_margin_fraction(tmp_path, monkeypatch):
    h = _load_harness()
    cpacs = _write_cpacs(tmp_path)
    t_req = 100000.0
    _install_fakes(monkeypatch, t_req_n=t_req)

    args = h.parse_args([
        "--cpacs", str(cpacs),
        "--output-root", str(tmp_path / "out"),
        "--target-margin-frac", "0.05",
        "--tol-n", "100",
    ])
    doc = h.run_loop(args)
    assert doc["status"] == "converged"
    # Margin should land near 5% of required thrust.
    assert doc["converged"]["thrust_margin_n"] == pytest.approx(0.05 * t_req, abs=200.0)


def test_pycycle_error_aborts_without_fake_convergence(tmp_path, monkeypatch):
    h = _load_harness()
    cpacs = _write_cpacs(tmp_path)
    _install_fakes(
        monkeypatch, t_req_n=80000.0,
        pyc_error={"type": "missing_dependency", "message": "OpenMDAO not installed"},
    )
    args = h.parse_args([
        "--cpacs", str(cpacs),
        "--output-root", str(tmp_path / "out"),
    ])
    doc = h.run_loop(args)
    assert doc["status"] == "error"
    assert doc["error"]["stage"] == "pycycle"
    assert doc.get("converged") is None


def test_no_thrust_closure_is_error(tmp_path, monkeypatch):
    h = _load_harness()
    cpacs = _write_cpacs(tmp_path)
    _install_fakes(monkeypatch, t_req_n=80000.0, drop_thrust_closure=True)
    args = h.parse_args([
        "--cpacs", str(cpacs),
        "--output-root", str(tmp_path / "out"),
    ])
    doc = h.run_loop(args)
    assert doc["status"] == "error"
    assert doc["error"]["type"] == "no_thrust_closure"


def test_thrust_bound_reached_does_not_claim_convergence(tmp_path, monkeypatch):
    h = _load_harness()
    cpacs = _write_cpacs(tmp_path)
    # Required thrust far exceeds the max engine the search permits.
    _install_fakes(monkeypatch, t_req_n=5.0e6)
    args = h.parse_args([
        "--cpacs", str(cpacs),
        "--output-root", str(tmp_path / "out"),
        "--max-thrust-lbf", "60000",
        "--max-iters", "8",
    ])
    doc = h.run_loop(args)
    assert doc["status"] == "did_not_converge"
    assert doc["stop_reason"] == "thrust_bound_reached"
    assert doc["converged"] is None


def test_missing_cpacs_rejected(tmp_path):
    h = _load_harness()
    args = h.parse_args([
        "--cpacs", str(tmp_path / "nope.xml"),
        "--output-root", str(tmp_path / "out"),
    ])
    with pytest.raises(FileNotFoundError):
        h.run_loop(args)
