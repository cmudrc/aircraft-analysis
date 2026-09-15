"""Tests for scripts/estimate_oew_from_wetted_area.py on synthetic CPACS strings.

No solver is involved: the script only reads a wetted area that the SU2 or
TiGL server recorded and multiplies it by K. The tests check the lookup order,
the refusal when nothing was recorded, calibration on a reference file, the
nacelle caveat, and the CPACS write.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "estimate_oew_from_wetted_area.py"

LB_PER_KG = 1.0 / 0.45359237
FT2_PER_M2 = 1.0 / 0.3048**2


def _load():
    spec = importlib.util.spec_from_file_location("estimate_oew_from_wetted_area", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _cpacs(*, aero_m2=None, aero_source=None, tigl_m2=None, oem_kg=None, engine=False) -> str:
    """Minimal CPACS with the nodes the servers write, in the shapes they write them."""
    aero = ""
    if aero_m2 is not None:
        aero = f"<aero><solver>su2_cfd</solver><wettedAreaM2>{aero_m2}</wettedAreaM2>"
        if aero_source is not None:
            aero += f"<wettedAreaSource>{aero_source}</wettedAreaSource>"
        aero += "<coefficients><CL>0.5</CL></coefficients></aero>"
    comps = (
        "<component><uid>wing1</uid><name>wing1</name><type>Wing</type></component>"
        "<component><uid>fus1</uid><name>fus1</name><type>Fuselage</type></component>"
    )
    if engine:
        comps += "<component><uid>eng1</uid><name>eng1</name><type>Engine</type></component>"
    fused = f"<fusedBody><surfaceAreaM2>{tigl_m2:.3f}</surfaceAreaM2><nacellesPylonsIncluded>false</nacellesPylonsIncluded></fusedBody>" if tigl_m2 is not None else ""
    tigl = f"<tigl><wingCount>1</wingCount><engineCount>{1 if engine else 0}</engineCount>{fused}<components>{comps}</components></tigl>"
    analyses = ""
    if oem_kg is not None:
        analyses = f"<analyses><massBreakdown><mOEM><massDescription><mass>{oem_kg}</mass></massDescription></mOEM></massBreakdown></analyses>"
    return (
        "<?xml version='1.0'?>"
        "<cpacs><vehicles><aircraft><model>"
        "<reference><area>122.4</area></reference>"
        f"{analyses}<analysisResults>{aero}{tigl}</analysisResults>"
        "</model></aircraft></vehicles></cpacs>"
    )


def _write(tmp_path: Path, name: str, xml: str) -> Path:
    p = tmp_path / name
    p.write_text(xml, encoding="utf-8")
    return p


def test_aero_node_is_used_first_with_its_stated_source(tmp_path: Path) -> None:
    mod = _load()
    src = "sum of the wall-marker faces of the SU2 mesh the run used"
    p = _write(tmp_path, "a.xml", _cpacs(aero_m2=718.15, aero_source=src, tigl_m2=700.0))
    est = mod.estimate_oew(p, k_lb_ft2=12.0)
    assert est["wetted_area_m2"] == 718.15
    assert est["wetted_area_source"] == src
    assert est["wetted_area_node"].endswith("analysisResults/aero/wettedAreaM2")
    expected_kg = 12.0 * 718.15 * FT2_PER_M2 / LB_PER_KG
    assert est["oew_kg"] == pytest.approx(expected_kg)
    assert est["oew_kg"] == pytest.approx(42076.0, abs=1.0)  # the D150 number from the journal
    assert est["oew_lb"] == pytest.approx(est["oew_kg"] * LB_PER_KG)
    assert est["k_lb_per_ft2"] == 12.0
    assert "--k-lb-ft2" in est["k_source"]
    assert est["method"] == "OEW = K * A_wet (rule of thumb, Ron Engelbeck, Boeing, 2026-09)"


def test_tigl_fused_body_is_the_fallback(tmp_path: Path) -> None:
    mod = _load()
    p = _write(tmp_path, "t.xml", _cpacs(tigl_m2=650.123))
    est = mod.estimate_oew(p, k_lb_ft2=10.0)
    assert est["wetted_area_m2"] == pytest.approx(650.123)
    assert est["wetted_area_node"].endswith("analysisResults/tigl/fusedBody/surfaceAreaM2")
    assert "TiGL" in est["wetted_area_source"]
    assert "nacellesPylonsIncluded=false" in est["wetted_area_source"]
    assert est["oew_kg"] == pytest.approx(10.0 * 650.123 * FT2_PER_M2 / LB_PER_KG)


def test_no_wetted_area_is_a_structured_error_not_a_formula(tmp_path: Path, capsys) -> None:
    mod = _load()
    p = _write(tmp_path, "n.xml", _cpacs())
    with pytest.raises(mod.EstimateError) as info:
        mod.estimate_oew(p, k_lb_ft2=12.0)
    assert info.value.error_type == "missing_input"
    assert "wettedAreaM2" in str(info.value) and "surfaceAreaM2" in str(info.value)
    assert isinstance(info.value, ValueError)
    # CLI prints the structured error and exits 2.
    assert mod.main([str(p), "--k-lb-ft2", "12"]) == 2
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["type"] == "missing_input"


def test_k_has_no_default_and_cannot_be_given_twice(tmp_path: Path) -> None:
    mod = _load()
    p = _write(tmp_path, "a.xml", _cpacs(aero_m2=718.15))
    ref = _write(tmp_path, "ref.xml", _cpacs(aero_m2=700.0, oem_kg=40000.0))
    with pytest.raises(mod.EstimateError) as info:
        mod.estimate_oew(p)
    assert info.value.error_type == "missing_input"
    with pytest.raises(mod.EstimateError) as info:
        mod.estimate_oew(p, k_lb_ft2=12.0, calibrate_from=ref)
    assert info.value.error_type == "conflicting_input"
    with pytest.raises(mod.EstimateError) as info:
        mod.estimate_oew(p, k_lb_ft2=0.0)
    assert info.value.error_type == "bad_input"


def test_calibration_from_a_reference_scales_oem_by_area_ratio(tmp_path: Path) -> None:
    mod = _load()
    ref = _write(tmp_path, "ref.xml", _cpacs(aero_m2=700.0, oem_kg=40000.0))
    target = _write(tmp_path, "target.xml", _cpacs(aero_m2=770.0))
    est = mod.estimate_oew(target, calibrate_from=ref)
    # K = OEM_ref / A_ref, so OEW_target = OEM_ref * A_target / A_ref, unit constants cancel.
    assert est["oew_kg"] == pytest.approx(40000.0 * 770.0 / 700.0)
    assert est["k_lb_per_ft2"] == pytest.approx(40000.0 * LB_PER_KG / (700.0 * FT2_PER_M2))
    assert "calibrated on" in est["k_source"] and "ref.xml" in est["k_source"]
    cal = est["calibration"]
    assert cal["reference_oem_kg"] == 40000.0
    assert cal["reference_wetted_area_m2"] == 700.0
    assert cal["reference_oem_node"].endswith("massBreakdown/mOEM/massDescription/mass")


def test_calibration_reference_must_state_oem_and_area(tmp_path: Path) -> None:
    mod = _load()
    target = _write(tmp_path, "target.xml", _cpacs(aero_m2=770.0))
    no_oem = _write(tmp_path, "no_oem.xml", _cpacs(aero_m2=700.0))
    with pytest.raises(mod.EstimateError, match="mOEM") as info:
        mod.estimate_oew(target, calibrate_from=no_oem)
    assert info.value.error_type == "missing_input"
    no_area = _write(tmp_path, "no_area.xml", _cpacs(oem_kg=40000.0))
    with pytest.raises(mod.EstimateError, match="reference no_area.xml") as info:
        mod.estimate_oew(target, calibrate_from=no_area)
    assert info.value.error_type == "missing_input"


def test_nacelle_caveat_present_without_engines_and_absent_with_them(tmp_path: Path) -> None:
    mod = _load()
    without = _write(tmp_path, "w.xml", _cpacs(aero_m2=718.15))
    assert mod.NO_NACELLE_CAVEAT in mod.estimate_oew(without, k_lb_ft2=12.0)["caveats"]
    with_engine = _write(tmp_path, "e.xml", _cpacs(aero_m2=718.15, engine=True))
    assert mod.NO_NACELLE_CAVEAT not in mod.estimate_oew(with_engine, k_lb_ft2=12.0)["caveats"]
    # A file with no TiGL component list at all cannot vouch for nacelles either.
    bare = _write(tmp_path, "b.xml",
                  "<cpacs><vehicles><aircraft><model><analysisResults><aero>"
                  "<wettedAreaM2>500</wettedAreaM2></aero></analysisResults>"
                  "</model></aircraft></vehicles></cpacs>")
    assert mod.NO_NACELLE_CAVEAT in mod.estimate_oew(bare, k_lb_ft2=12.0)["caveats"]


def test_write_records_estimate_and_touches_nothing_else(tmp_path: Path, capsys) -> None:
    mod = _load()
    src = "sum of the wall-marker faces of the SU2 mesh the run used"
    p = _write(tmp_path, "w.xml", _cpacs(aero_m2=718.15, aero_source=src))
    before = ET.fromstring(p.read_text(encoding="utf-8"))

    assert mod.main([str(p), "--k-lb-ft2", "12", "--write"]) == 0
    out = capsys.readouterr().out
    assert f"written to   {p}" in out

    root = ET.fromstring(p.read_text(encoding="utf-8"))
    node = root.find("vehicles/aircraft/model/analysisResults/massProperties/oewEstimate")
    assert node is not None
    assert float(node.findtext("oewKg")) == pytest.approx(12.0 * 718.15 * FT2_PER_M2 / LB_PER_KG)
    assert node.findtext("method") == mod.METHOD
    assert float(node.findtext("kLbPerFt2")) == 12.0
    assert "--k-lb-ft2" in node.findtext("kSource")
    assert float(node.findtext("wettedAreaM2")) == 718.15
    assert node.findtext("wettedAreaSource") == src
    caveats = [c.text for c in node.find("caveats")]
    assert mod.NO_NACELLE_CAVEAT in caveats

    # Everything that was there before is still there, unchanged.
    for path in ("vehicles/aircraft/model/reference/area",
                 "vehicles/aircraft/model/analysisResults/aero/wettedAreaM2",
                 "vehicles/aircraft/model/analysisResults/aero/coefficients/CL",
                 "vehicles/aircraft/model/analysisResults/tigl/components"):
        assert ET.tostring(root.find(path)) == ET.tostring(before.find(path))

    # Re-running replaces the estimate rather than adding a second one.
    assert mod.main([str(p), "--k-lb-ft2", "11", "--write"]) == 0
    root = ET.fromstring(p.read_text(encoding="utf-8"))
    nodes = root.findall("vehicles/aircraft/model/analysisResults/massProperties/oewEstimate")
    assert len(nodes) == 1 and float(nodes[0].findtext("kLbPerFt2")) == 11.0


def test_write_to_a_separate_path_leaves_the_input_alone(tmp_path: Path) -> None:
    mod = _load()
    p = _write(tmp_path, "in.xml", _cpacs(aero_m2=718.15))
    original = p.read_text(encoding="utf-8")
    out = tmp_path / "out.xml"
    assert mod.main([str(p), "--k-lb-ft2", "12", "--out", str(out), "--json"]) == 0
    assert p.read_text(encoding="utf-8") == original
    root = ET.fromstring(out.read_text(encoding="utf-8"))
    assert root.find("vehicles/aircraft/model/analysisResults/massProperties/oewEstimate") is not None


def test_missing_file_is_a_structured_error(tmp_path: Path, capsys) -> None:
    mod = _load()
    assert mod.main([str(tmp_path / "nope.xml"), "--k-lb-ft2", "12"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "missing_input"
