"""Unit tests for the cross-discipline cruise-match harness.

No real solver is required: SU2 and pyCycle are replaced by deterministic stubs
and NSEG's ``cpacs_adapter.run_adapter`` is monkeypatched (the real
``nseg_mcp.physics.atmosphere`` is kept so dynamic pressure stays honest). This
is the "adapter logic test" exemption in the no-stubs rule — the production path
still calls the real SU2/pyCycle/NSEG; here we test the fixed-point loop.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

_HARNESS_PATH = Path(__file__).resolve().parent.parent / "run_cruise_match.py"

LBF_TO_N = 4.44822


def _load_harness():
    spec = importlib.util.spec_from_file_location("run_cruise_match", _HARNESS_PATH)
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


def _write_cpacs(tmp_path: Path) -> Path:
    cpacs = tmp_path / "fake.xml"
    cpacs.write_text(_CPACS, encoding="utf-8")
    return cpacs


def _install_su2(monkeypatch, by_aoa):
    def fake_run_adapter(xml, **kwargs):  # type: ignore[no-untyped-def]
        aoa = kwargs["flight_conditions"]["aoa"]
        return xml, dict(by_aoa[aoa])

    mod = type(sys)("su2_mcp.cpacs_adapter")
    mod.run_adapter = fake_run_adapter  # type: ignore[attr-defined]
    pkg = type(sys)("su2_mcp")
    pkg.cpacs_adapter = mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "su2_mcp", pkg)
    monkeypatch.setitem(sys.modules, "su2_mcp.cpacs_adapter", mod)


def _install_pyc(monkeypatch, *, error=None, tsfc=1.7e-5):
    def fake_run_adapter(xml, flight_conditions=None, design_thrust_lbf=None):
        if error is not None:
            return xml, {"error": error}
        fn_des = float(design_thrust_lbf) if design_thrust_lbf is not None else 12000.0
        fn_n = fn_des * LBF_TO_N  # design mode: Fn == Fn_DES => thrust == drag
        return xml, {"Fn_DES_lbf": round(fn_des, 2), "Fn_N": round(fn_n, 2),
                     "Fn_lbf": round(fn_des, 2), "TSFC_1_per_s": tsfc}

    mod = type(sys)("pycycle_mcp.cpacs_adapter")
    mod.run_adapter = fake_run_adapter  # type: ignore[attr-defined]
    pkg = type(sys)("pycycle_mcp")
    pkg.cpacs_adapter = mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pycycle_mcp", pkg)
    monkeypatch.setitem(sys.modules, "pycycle_mcp.cpacs_adapter", mod)


def _install_nseg(monkeypatch, *, fuel_frac=0.15, error=None):
    """Fake only nseg_mcp.cpacs_adapter; keep real package for atmosphere."""
    import nseg_mcp.cpacs_adapter as real_nseg  # noqa: F401  (ensures real pkg loaded)

    def fake_run_adapter(xml, mission_profile=None):
        if error is not None:
            return xml, {"error": error}
        w = float((mission_profile or {}).get("weight_kg", 0.0))
        return xml, {"success": True, "total_fuel_burned_kg": w * fuel_frac}

    monkeypatch.setattr(real_nseg, "run_adapter", fake_run_adapter)


# --------------------------------------------------------------------------
# Pure helper tests
# --------------------------------------------------------------------------

def test_polar_aoa_list_dedupes_and_requires_two():
    h = _load_harness()
    assert h.polar_aoa_list("3,1,1,2") == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        h.polar_aoa_list("2")


def test_fit_polar_two_points_exact():
    h = _load_harness()
    cd0, k = h.fit_polar([(0.4, 0.026), (0.6, 0.034)])
    # k = (0.034-0.026)/(0.36-0.16) = 0.04 ; cd0 = 0.026 - 0.04*0.16
    assert k == pytest.approx(0.04, abs=1e-9)
    assert cd0 == pytest.approx(0.0196, abs=1e-9)


def test_fit_polar_least_squares_three_points():
    h = _load_harness()
    # Points generated from cd0=0.02, k=0.05 exactly -> recovered exactly.
    pts = [(cl, 0.02 + 0.05 * cl * cl) for cl in (0.3, 0.5, 0.7)]
    cd0, k = h.fit_polar(pts)
    assert cd0 == pytest.approx(0.02, abs=1e-9)
    assert k == pytest.approx(0.05, abs=1e-9)


def test_fit_polar_rejects_insufficient_and_degenerate():
    h = _load_harness()
    with pytest.raises(ValueError):
        h.fit_polar([(0.4, 0.026)])
    with pytest.raises(ValueError):
        h.fit_polar([(0.4, 0.026), (0.4, 0.030)])  # same CL


def test_cruise_state_math():
    h = _load_harness()
    cs = h.cruise_state(70000.0, q_pa=12000.0, ref_area_m2=122.4, cd0=0.02, k=0.04)
    expected_cl = 70000.0 * 9.80665 / (12000.0 * 122.4)
    assert cs["CL"] == pytest.approx(expected_cl)
    assert cs["CD"] == pytest.approx(0.02 + 0.04 * expected_cl ** 2)
    assert cs["drag_n"] == pytest.approx(cs["CD"] * 12000.0 * 122.4)
    assert cs["L_over_D"] == pytest.approx(cs["CL"] / cs["CD"])


def test_set_aero_coeffs_lets_nseg_recover_k():
    h = _load_harness()
    cd0, k, cl = 0.02, 0.045, 0.5
    cd = cd0 + k * cl * cl
    xml = h._set_aero_coeffs(_CPACS, cd0, cl, cd)
    root = ET.fromstring(xml)
    coeff = root.find(".//analysisResults/aero/coefficients")
    got_cd0 = float(coeff.find("CD0").text)
    got_cl = float(coeff.find("CL").text)
    got_cd = float(coeff.find("CD").text)
    recovered_k = (got_cd - got_cd0) / (got_cl * got_cl)
    assert recovered_k == pytest.approx(k, abs=1e-9)


# --------------------------------------------------------------------------
# Integration tests through run_loop with fakes
# --------------------------------------------------------------------------

def test_sizing_mode_converges_weight(tmp_path, monkeypatch):
    h = _load_harness()
    cpacs = _write_cpacs(tmp_path)
    mesh = tmp_path / "m.su2"
    mesh.write_text("NDIME= 3\n", encoding="utf-8")
    _install_su2(monkeypatch, {
        1.0: {"CL": 0.40, "CD": 0.026, "error": None},
        3.0: {"CL": 0.60, "CD": 0.034, "error": None},
    })
    _install_pyc(monkeypatch)
    _install_nseg(monkeypatch, fuel_frac=0.15)

    args = h.parse_args([
        "--cpacs", str(cpacs), "--mesh", str(mesh),
        "--output-root", str(tmp_path / "out"),
        "--oew", "42000", "--payload", "18000", "--range-km", "3000",
        "--polar-aoa", "1,3", "--reserve-frac", "0.05",
    ])
    doc = h.run_loop(args)
    assert doc["status"] == "converged"
    # Closed-form fixed point: W = (oew+payload)/(1 - 0.15*1.05).
    expected_w = 60000.0 / (1.0 - 0.15 * 1.05)
    assert doc["converged"]["W_TO_kg"] == pytest.approx(expected_w, rel=2e-3)
    # Thrust was sized to drag each iteration.
    assert abs(doc["converged"]["thrust_drag_residual_n"]) < 1.0
    assert doc["polar"]["source"] == "su2"
    assert (tmp_path / "out" / "cruise_match_history.json").exists()


def test_match_mode_single_pass_with_user_polar(tmp_path, monkeypatch):
    h = _load_harness()
    cpacs = _write_cpacs(tmp_path)
    _install_pyc(monkeypatch)
    _install_nseg(monkeypatch, fuel_frac=0.15)

    args = h.parse_args([
        "--cpacs", str(cpacs),
        "--output-root", str(tmp_path / "out"),
        "--cd0", "0.022", "--k", "0.045",
        "--weight", "70000", "--range-km", "3000",
    ])
    doc = h.run_loop(args)
    assert doc["status"] == "converged"
    assert doc["mode"] == "match"
    assert doc["n_iters"] == 1
    assert doc["polar"]["source"] == "user"
    assert abs(doc["converged"]["thrust_drag_residual_n"]) < 1.0


def test_pycycle_error_aborts(tmp_path, monkeypatch):
    h = _load_harness()
    cpacs = _write_cpacs(tmp_path)
    _install_pyc(monkeypatch, error={"type": "missing_dependency", "message": "no OpenMDAO"})
    _install_nseg(monkeypatch)

    args = h.parse_args([
        "--cpacs", str(cpacs), "--output-root", str(tmp_path / "out"),
        "--cd0", "0.022", "--k", "0.045", "--weight", "70000",
    ])
    doc = h.run_loop(args)
    assert doc["status"] == "error"
    assert doc["error"]["stage"] == "pycycle"
    assert doc.get("converged") is None


def test_polar_needs_two_successful_points(tmp_path, monkeypatch):
    h = _load_harness()
    cpacs = _write_cpacs(tmp_path)
    mesh = tmp_path / "m.su2"
    mesh.write_text("NDIME= 3\n", encoding="utf-8")
    _install_su2(monkeypatch, {
        1.0: {"CL": None, "CD": None, "error": {"type": "su2_failure", "message": "diverged"}},
        3.0: {"CL": 0.60, "CD": 0.034, "error": None},
    })
    _install_pyc(monkeypatch)
    _install_nseg(monkeypatch)

    args = h.parse_args([
        "--cpacs", str(cpacs), "--mesh", str(mesh),
        "--output-root", str(tmp_path / "out"),
        "--oew", "42000", "--payload", "18000", "--polar-aoa", "1,3",
    ])
    with pytest.raises(ValueError):
        h.run_loop(args)


def test_missing_cpacs_rejected(tmp_path):
    h = _load_harness()
    args = h.parse_args([
        "--cpacs", str(tmp_path / "nope.xml"),
        "--output-root", str(tmp_path / "out"),
        "--cd0", "0.02", "--k", "0.04", "--weight", "70000",
    ])
    with pytest.raises(FileNotFoundError):
        h.run_loop(args)
