"""Tests for scripts/estimate_oew_flops_aviary.py.

The real path runs Aviary's FLOPS geometry and mass groups on a synthetic
CPACS file shaped like the D150 (wing with a positioning chain, horizontal and
vertical tails, a fuselage built from scaled circle profiles). Those tests skip
with a reason when Aviary is not importable. The geometry readers and the
structured-error paths need no solver; the missing-dependency path is
exercised by monkeypatching the import.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "estimate_oew_flops_aviary.py"
_D150 = Path(__file__).resolve().parents[3] / "D150_v30.xml"

os.environ.setdefault("OPENMDAO_REPORTS", "0")

try:
    import aviary  # noqa: F401
    import openmdao.api  # noqa: F401

    _HAVE_AVIARY = True
except ImportError:  # pragma: no cover
    _HAVE_AVIARY = False

needs_aviary = pytest.mark.skipif(not _HAVE_AVIARY, reason="NASA Aviary is not importable in this environment")

# The CLI set the D150 run in the journal used; the file states none of these.
D150_CLI = dict(
    design_gross_weight_kg=73500.0, passengers=150, design_range_km=5000.0, num_engines=2,
    engine_sls_thrust_kn=120.0, engine_mass_kg=2380.0, nacelle_diameter_m=2.1, nacelle_length_m=4.4,
    engine_spanwise_fraction=0.34, passenger_compartment_length_m=27.5, max_mach=0.82,
)


def _load():
    spec = importlib.util.spec_from_file_location("estimate_oew_flops_aviary", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _naca_like(n: int = 40, tc: float = 0.12) -> tuple[str, str]:
    """Closed airfoil loop trailing edge -> upper -> leading edge -> lower -> trailing edge."""
    xs, zs = [], []
    for i in range(n + 1):
        x = 1.0 - i / n
        xs.append(x)
        zs.append(0.5 * tc * math.sin(math.pi * x))
    for i in range(1, n + 1):
        x = i / n
        xs.append(x)
        zs.append(-0.5 * tc * math.sin(math.pi * x))
    return ";".join(f"{v:.6f}" for v in xs), ";".join(f"{v:.6f}" for v in zs)


def _circle(n: int = 36) -> tuple[str, str]:
    ang = [2 * math.pi * i / n for i in range(n + 1)]
    return ";".join(f"{math.cos(a):.6f}" for a in ang), ";".join(f"{math.sin(a):.6f}" for a in ang)


def _wing(uid: str, airfoil: str, chords: list[float], positionings: list[tuple[float, float, float]],
          *, symmetric: bool = True, translation=(0.0, 0.0, 0.0), rot_x: float = 0.0) -> str:
    """A CPACS wing whose sections sit on a positioning chain (length, sweep, dihedral)."""
    secs, poss = "", ""
    prev = None
    for i, (chord, (length, sweep, dih)) in enumerate(zip(chords, positionings), start=1):
        sid = f"{uid}_sec{i}"
        secs += (
            f"<section uID='{sid}'><transformation><scaling><x>1</x><y>1</y><z>1</z></scaling>"
            "<rotation><x>0</x><y>0</y><z>0</z></rotation><translation><x>0</x><y>0</y><z>0</z></translation>"
            f"</transformation><elements><element uID='{sid}_el'><airfoilUID>{airfoil}</airfoilUID>"
            f"<transformation><scaling><x>{chord}</x><y>{chord}</y><z>{chord}</z></scaling>"
            "<rotation><x>0</x><y>0</y><z>0</z></rotation><translation><x>0</x><y>0</y><z>0</z></translation>"
            "</transformation></element></elements></section>"
        )
        frm = f"<fromSectionUID>{prev}</fromSectionUID>" if prev else ""
        poss += (
            f"<positioning uID='{sid}_pos'><length>{length}</length><sweepAngle>{sweep}</sweepAngle>"
            f"<dihedralAngle>{dih}</dihedralAngle>{frm}<toSectionUID>{sid}</toSectionUID></positioning>"
        )
        prev = sid
    sym = " symmetry='x-z-plane'" if symmetric else ""
    tx, ty, tz = translation
    return (
        f"<wing uID='{uid}'{sym}><name>{uid}</name><transformation>"
        "<scaling><x>1</x><y>1</y><z>1</z></scaling>"
        f"<rotation><x>{rot_x}</x><y>0</y><z>0</z></rotation>"
        f"<translation><x>{tx}</x><y>{ty}</y><z>{tz}</z></translation></transformation>"
        f"<sections>{secs}</sections><positionings>{poss}</positionings></wing>"
    )


def _fuselage(uid: str, stations: list[tuple[float, float, float]]) -> str:
    """Fuselage sections placed along x by positionings of sweep 90 with (dx, half-width, half-height)."""
    secs, poss = "", ""
    prev = None
    for i, (dx, ry, rz) in enumerate(stations, start=1):
        sid = f"{uid}_s{i}"
        secs += (
            f"<section uID='{sid}'><transformation><scaling><x>1</x><y>1</y><z>1</z></scaling>"
            "<rotation><x>0</x><y>0</y><z>0</z></rotation><translation><x>0</x><y>0</y><z>0</z></translation>"
            f"</transformation><elements><element uID='{sid}_el'><profileUID>circle</profileUID>"
            f"<transformation><scaling><x>1</x><y>{ry}</y><z>{rz}</z></scaling>"
            "<rotation><x>0</x><y>0</y><z>0</z></rotation><translation><x>0</x><y>0</y><z>0</z></translation>"
            "</transformation></element></elements></section>"
        )
        frm = f"<fromSectionUID>{prev}</fromSectionUID>" if prev else ""
        poss += (
            f"<positioning uID='{sid}_pos'><length>{dx}</length><sweepAngle>90</sweepAngle>"
            f"<dihedralAngle>0</dihedralAngle>{frm}<toSectionUID>{sid}</toSectionUID></positioning>"
        )
        prev = sid
    return (
        f"<fuselage uID='{uid}'><name>{uid}</name><transformation><scaling><x>1</x><y>1</y><z>1</z></scaling>"
        "<rotation><x>0</x><y>0</y><z>0</z></rotation><translation><x>0</x><y>0</y><z>0</z></translation>"
        f"</transformation><sections>{secs}</sections><positionings>{poss}</positionings></fuselage>"
    )


def _cpacs(*, ref_area=120.0, mtom_kg=None, pax=None, range_m=None, engines=None, mach=0.78,
           tails=True, wetted_m2=None) -> str:
    ax, az = _naca_like(tc=0.15)
    hx, hz = _naca_like(tc=0.10)
    cy, cz = _circle()
    # Main wing: 2 m unswept centre segment then a 15 m swept, tapered outer panel.
    wings = _wing("mainWing", "wingFoil", [6.0, 6.0, 1.5],
                  [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (15.0, 27.0, 5.0)], translation=(12.0, 0.0, -1.2))
    if tails:
        wings += _wing("htp", "tailFoil", [3.7, 1.2], [(0.0, 0.0, 0.0), (6.2, 30.0, 5.0)], translation=(31.0, 0.0, 0.7))
        wings += _wing("vtp", "tailFoil", [5.4, 1.9], [(0.0, 0.0, 0.0), (6.0, 40.0, 0.0)],
                       symmetric=False, translation=(30.0, 0.0, 1.9), rot_x=90.0)
    fus = _fuselage("fus", [(0.0, 0.0, 0.0), (2.0, 1.2, 1.3), (4.0, 1.975, 2.07), (25.0, 1.975, 2.07),
                            (5.0, 0.8, 0.9), (1.5, 0.0, 0.0)])
    analyses = ""
    if mtom_kg is not None:
        analyses = f"<analyses><massBreakdown><designMasses><mTOM><mass>{mtom_kg}</mass></mTOM></designMasses></massBreakdown></analyses>"
    glob = ""
    if pax is not None or range_m is not None:
        glob = "<global>"
        if pax is not None:
            glob += f"<payload><paxSeats><actual>{pax}</actual></payload>".replace("</payload>", "</paxSeats></payload>")
        if range_m is not None:
            glob += f"<designRange><required>{range_m}</required></designRange>"
        glob += "</global>"
    eng_model = eng_lib = ""
    if engines is not None:
        n, thrust_n, mass_kg = engines
        eng_model = "<engines>" + "".join(
            f"<engine uID='engPos{i}'{' symmetry=' + chr(39) + 'x-z-plane' + chr(39) if n == 2 and i == 1 else ''}>"
            "<engineUID>cfm</engineUID><parentUID>mainWing</parentUID></engine>"
            for i in range(1, (1 if n == 2 else n) + 1)) + "</engines>"
        eng_lib = (f"<engines><engine uID='cfm'><name>cfm</name><analysis><thrust00>{thrust_n}</thrust00>"
                   f"<mass><mass>{mass_kg}</mass></mass></analysis></engine></engines>")
    aero = f"<aero><solver>su2_cfd</solver><mach>{mach}</mach>" if mach is not None else "<aero>"
    if wetted_m2 is not None:
        aero += f"<wettedAreaM2>{wetted_m2}</wettedAreaM2>"
    aero += "<coefficients><CL>0.5</CL></coefficients></aero>"
    return (
        "<?xml version='1.0'?><cpacs><vehicles><aircraft><model uID='ac'>"
        f"<name>synthetic</name><reference><area>{ref_area}</area><length>4.2</length></reference>"
        f"{glob}<fuselages>{fus}</fuselages><wings>{wings}</wings>{eng_model}{analyses}"
        f"<analysisResults>{aero}</analysisResults></model></aircraft>{eng_lib}"
        "<profiles><fuselageProfiles><fuselageProfile uID='circle'><pointList>"
        f"<x>{';'.join('0' for _ in cy.split(';'))}</x><y>{cy}</y><z>{cz}</z></pointList></fuselageProfile></fuselageProfiles>"
        "<wingAirfoils>"
        f"<wingAirfoil uID='wingFoil'><pointList><x>{ax}</x><y>{';'.join('0' for _ in ax.split(';'))}</y><z>{az}</z></pointList></wingAirfoil>"
        f"<wingAirfoil uID='tailFoil'><pointList><x>{hx}</x><y>{';'.join('0' for _ in hx.split(';'))}</y><z>{hz}</z></pointList></wingAirfoil>"
        "</wingAirfoils></profiles></vehicles></cpacs>"
    )


def _write(tmp_path: Path, name: str, xml: str) -> Path:
    p = tmp_path / name
    p.write_text(xml, encoding="utf-8")
    return p


# ------------------------------------------------------------------ geometry readers (no solver)


def test_airfoil_thickness_ratio_recovers_the_generating_thickness() -> None:
    mod = _load()
    xs, zs = _naca_like(tc=0.12)
    xs = [float(v) for v in xs.split(";")]
    zs = [float(v) for v in zs.split(";")]
    assert mod.airfoil_thickness_ratio(xs, zs) == pytest.approx(0.12, rel=1e-3)


def test_lifting_surfaces_and_fuselage_are_read_from_sections(tmp_path: Path) -> None:
    mod = _load()
    root = ET.fromstring(_cpacs())
    fus = mod.read_fuselage(root, "syn")
    assert fus["fuselage_length_m"] == pytest.approx(37.5)
    assert fus["fuselage_max_width_m"] == pytest.approx(2 * 1.975, rel=1e-3)
    assert fus["fuselage_max_height_m"] == pytest.approx(2 * 2.07, rel=1e-3)

    surf = mod.read_lifting_surfaces(root, "syn", 0.5 * fus["fuselage_max_width_m"])
    assert surf["wing_uid"] == "mainWing"
    half = 2.0 + 15.0 * math.cos(math.radians(27.0)) * math.cos(math.radians(5.0))
    assert surf["wing_span_m"] == pytest.approx(2 * half, rel=1e-6)
    assert surf["wing_taper_ratio"] == pytest.approx(0.25)
    # Section 2 (y = 2.0) is just outside the 1.975 m half-width, so the reference
    # trapezoid runs from the root: quarter-chord sweep of root -> tip.
    dx_le = 15.0 * math.sin(math.radians(27.0))
    dx_qc = dx_le + 0.25 * (1.5 - 6.0)
    assert surf["wing_sweep_qc_deg"] == pytest.approx(math.degrees(math.atan2(dx_qc, half)), rel=1e-6)
    assert surf["wing_dihedral_deg"] == pytest.approx(math.degrees(math.atan2(15.0 * math.cos(math.radians(27.0)) * math.sin(math.radians(5.0)), half)), rel=1e-6)
    assert surf["wing_tc"] == pytest.approx(0.15, rel=1e-3)
    assert surf["htail_uid"] == "htp" and surf["vtail_uids"] == ["vtp"]
    ht_half = 6.2 * math.cos(math.radians(30.0)) * math.cos(math.radians(5.0))
    assert surf["htail_area_m2"] == pytest.approx(2 * 0.5 * (3.7 + 1.2) * ht_half, rel=1e-6)
    assert surf["htail_aspect_ratio"] == pytest.approx((2 * ht_half) ** 2 / surf["htail_area_m2"], rel=1e-6)
    vt_h = 6.0 * math.cos(math.radians(40.0))
    assert surf["vtail_area_each_m2"] == pytest.approx(0.5 * (5.4 + 1.9) * vt_h, rel=1e-6)
    assert surf["vtail_count"] == 1
    assert surf["htail_on_vtail_fraction"] == 0.0  # HTP root below the VTP root: body-mounted


def test_engines_are_read_when_the_file_has_them() -> None:
    mod = _load()
    root = ET.fromstring(_cpacs(engines=(2, 120000.0, 2380.0)))
    eng = mod.read_engines(root, "syn")
    assert eng["count"] == 2 and eng["sls_thrust_n"] == 120000.0 and eng["mass_kg"] == 2380.0
    assert mod.read_engines(ET.fromstring(_cpacs()), "syn") is None


# ------------------------------------------------------------------ structured errors (no solver)


def test_missing_aircraft_properties_are_missing_input_errors_naming_the_flag(tmp_path: Path, capsys) -> None:
    mod = _load()
    p = _write(tmp_path, "s.xml", _cpacs())
    with pytest.raises(mod.EstimateError) as info:
        mod.estimate_oew(p, passengers=150)
    assert info.value.error_type == "missing_input"
    assert "--design-gross-weight-kg" in str(info.value) and "mTOM" in str(info.value)
    with pytest.raises(mod.EstimateError, match="--passengers") as info:
        mod.estimate_oew(p, design_gross_weight_kg=73500.0)
    assert info.value.error_type == "missing_input"
    with pytest.raises(mod.EstimateError, match="--design-range-km"):
        mod.estimate_oew(p, design_gross_weight_kg=73500.0, passengers=150)
    with pytest.raises(mod.EstimateError, match="--num-engines"):
        mod.estimate_oew(p, design_gross_weight_kg=73500.0, passengers=150, design_range_km=5000.0)
    with pytest.raises(mod.EstimateError, match="--passenger-compartment-length-m"):
        mod.estimate_oew(p, **{**D150_CLI, "passenger_compartment_length_m": None})
    # CLI prints the structured error and exits 2 before Aviary is ever touched.
    assert mod.main([str(p)]) == 2
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["type"] == "missing_input"


def test_file_value_and_cli_value_together_are_a_conflict(tmp_path: Path) -> None:
    mod = _load()
    p = _write(tmp_path, "c.xml", _cpacs(mtom_kg=73500.0))
    with pytest.raises(mod.EstimateError) as info:
        mod.estimate_oew(p, **D150_CLI)
    assert info.value.error_type == "conflicting_input"
    assert "--design-gross-weight-kg" in str(info.value) and "mTOM" in str(info.value)


def test_geometry_gaps_are_structured_errors(tmp_path: Path) -> None:
    mod = _load()
    no_tails = _write(tmp_path, "t.xml", _cpacs(tails=False))
    with pytest.raises(mod.EstimateError) as info:
        mod.estimate_oew(no_tails, **D150_CLI)
    assert info.value.error_type == "missing_input" and "horizontal tail" in str(info.value)
    bare = _write(tmp_path, "b.xml", "<cpacs><vehicles><aircraft><model><reference><area>1</area></reference></model></aircraft></vehicles></cpacs>")
    with pytest.raises(mod.EstimateError, match="fuselages/fuselage") as info:
        mod.estimate_oew(bare, **D150_CLI)
    assert info.value.error_type == "missing_input"
    with pytest.raises(mod.EstimateError) as info:
        mod.estimate_oew(tmp_path / "nope.xml", **D150_CLI)
    assert info.value.error_type == "missing_input"
    with pytest.raises(mod.EstimateError) as info:
        mod.estimate_oew(_write(tmp_path, "bad.xml", "<cpacs><unclosed>"), **D150_CLI)
    assert info.value.error_type == "bad_input"


def test_missing_aviary_is_a_missing_dependency_error(tmp_path: Path, monkeypatch, capsys) -> None:
    mod = _load()
    p = _write(tmp_path, "s.xml", _cpacs())

    def _no_aviary():
        raise mod.EstimateError("missing_dependency", "NASA Aviary (with OpenMDAO) is not importable: No module named 'aviary'")

    monkeypatch.setattr(mod, "_import_aviary", _no_aviary)
    with pytest.raises(mod.EstimateError) as info:
        mod.estimate_oew(p, **D150_CLI)
    assert info.value.error_type == "missing_dependency"
    argv = [str(p)] + [f"--{k.replace('_', '-')}={v}" for k, v in D150_CLI.items()]
    assert mod.main(argv) == 2
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "missing_dependency"


def test_set_override_syntax_is_checked() -> None:
    mod = _load()
    assert mod._parse_set(["aircraft:wing:mass_scaler=0.9", "aircraft:paint:mass_per_unit_area=0.037:lbm/ft**2"]) == {
        "aircraft:wing:mass_scaler": (0.9, "unitless"),
        "aircraft:paint:mass_per_unit_area": (0.037, "lbm/ft**2"),
    }
    with pytest.raises(mod.EstimateError) as info:
        mod._parse_set(["aircraft:wing:mass_scaler"])
    assert info.value.error_type == "bad_input"
    with pytest.raises(mod.EstimateError):
        mod._parse_set(["aircraft:wing:mass_scaler=big"])


# ------------------------------------------------------------------ the real Aviary path


@needs_aviary
def test_aviary_flops_run_on_synthetic_cpacs_is_consistent_and_traced(tmp_path: Path) -> None:
    mod = _load()
    import aviary

    p = _write(tmp_path, "syn.xml", _cpacs(wetted_m2=700.0))
    est = mod.estimate_oew(p, **D150_CLI)

    assert est["method"] == f"FLOPS via Aviary {aviary.__version__}"
    assert 25_000 < est["oew_kg"] < 60_000
    assert est["oew_lb"] == pytest.approx(est["oew_kg"] / 0.45359237)
    b = est["breakdown"]
    # Aviary's MassSummation identities hold for the numbers we report.
    s, pr, sy = b["structure"], b["propulsion"], b["systems_and_equipment"]
    assert s["total_kg"] == pytest.approx(sum(v for k, v in s.items() if k != "total_kg"), rel=1e-9)
    assert pr["total_kg"] == pytest.approx(sum(v for k, v in pr.items() if k != "total_kg"), rel=1e-9)
    assert sy["total_kg"] == pytest.approx(sum(v for k, v in sy.items() if k != "total_kg"), rel=1e-9)
    assert b["empty_mass_kg"] == pytest.approx(s["total_kg"] + pr["total_kg"] + sy["total_kg"] + b["empty_mass_margin_kg"], rel=1e-9)
    assert est["oew_kg"] == pytest.approx(b["empty_mass_kg"] + sum(b["operating_items"].values()), rel=1e-9)
    assert est["oew_kg"] == b["operating_mass_kg"]
    # The stated engine mass passes straight through; nacelles and gear are non-zero.
    assert pr["engines_kg"] == pytest.approx(2 * 2380.0, rel=1e-9)
    assert s["nacelles_kg"] > 0 and s["main_landing_gear_kg"] > 0 and sy["furnishings_kg"] > 0
    assert b["operating_items"]["cabin_crew_kg"] > 0  # 4 attendants from Aviary's rule for 150 seats
    assert est["derived"]["fuel_capacity_kg"] == pytest.approx(est["derived"]["wing_fuel_capacity_kg"])
    assert est["derived"]["aspect_ratio"] == pytest.approx(
        next(r["value"] for r in est["inputs"] if r["name"] == "aircraft:wing:span") ** 2 / 120.0, rel=1e-6)

    sources = {r["name"]: r["source"] for r in est["inputs"]}
    assert sources["aircraft:wing:area"] == "cpacs://vehicles/aircraft/model/reference/area"
    assert sources["mission:design:gross_mass"] == "cli:--design-gross-weight-kg"
    assert sources["aircraft:crew_and_payload:design:num_passengers"] == "cli:--passengers"
    assert sources["mission:summary:cruise_mach"].startswith("cpacs:") and sources["mission:summary:cruise_mach"].endswith("aero/mach")
    assert sources["aircraft:wing:ultimate_load_factor"] == "flops_default"
    assert sources["aircraft:crew_and_payload:num_flight_attendants"] == "aviary:preprocess_crewpayload"
    assert {r["name"]: r["value"] for r in est["inputs"]}["aircraft:crew_and_payload:num_flight_attendants"] == 4
    assert any("cross-check" in c and "700.0 m^2" in c for c in est["caveats"])
    assert any("caller-stated inputs" in c for c in est["caveats"])


@needs_aviary
def test_file_stated_properties_are_used_and_labelled(tmp_path: Path) -> None:
    mod = _load()
    xml = _cpacs(mtom_kg=73500.0, pax=150, range_m=5_000_000.0, engines=(2, 120000.0, 2380.0))
    p = _write(tmp_path, "full.xml", xml)
    cli = {k: v for k, v in D150_CLI.items()
           if k not in ("design_gross_weight_kg", "passengers", "design_range_km", "num_engines", "engine_sls_thrust_kn", "engine_mass_kg")}
    est = mod.estimate_oew(p, **cli)
    sources = {r["name"]: r["source"] for r in est["inputs"]}
    assert sources["mission:design:gross_mass"].endswith("designMasses/mTOM/mass")
    assert sources["aircraft:crew_and_payload:design:num_passengers"].endswith("global/payload/paxSeats/actual")
    assert sources["mission:design:range"].endswith("global/designRange/required")
    assert sources["aircraft:engine:num_engines"].endswith("engines/engine")
    assert sources["aircraft:engine:scaled_sls_thrust"].endswith("analysis/thrust00")
    values = {r["name"]: r["value"] for r in est["inputs"]}
    assert values["mission:design:range"] == pytest.approx(5000.0)
    assert values["aircraft:engine:scaled_sls_thrust"] == pytest.approx(120000.0)
    # Same aircraft either way, so the same OEW as with everything on the CLI.
    ref = mod.estimate_oew(_write(tmp_path, "cli.xml", _cpacs()), **D150_CLI)
    assert est["oew_kg"] == pytest.approx(ref["oew_kg"], rel=1e-9)


@needs_aviary
def test_set_overrides_reach_aviary_and_fuel_capacity_can_be_stated(tmp_path: Path) -> None:
    mod = _load()
    p = _write(tmp_path, "syn.xml", _cpacs())
    base = mod.estimate_oew(p, **D150_CLI)
    scaled = mod.estimate_oew(p, **D150_CLI, set=["aircraft:wing:mass_scaler=0.5"])
    assert scaled["breakdown"]["structure"]["wing_kg"] == pytest.approx(0.5 * base["breakdown"]["structure"]["wing_kg"], rel=1e-9)
    stated = mod.estimate_oew(p, **D150_CLI, fuel_capacity_kg=19000.0)
    assert stated["derived"]["fuel_capacity_kg"] == pytest.approx(19000.0, rel=1e-9)
    assert stated["breakdown"]["propulsion"]["fuel_system_kg"] != base["breakdown"]["propulsion"]["fuel_system_kg"]


@needs_aviary
def test_write_records_estimate_and_touches_nothing_else(tmp_path: Path, capsys) -> None:
    mod = _load()
    p = _write(tmp_path, "w.xml", _cpacs())
    before = ET.fromstring(p.read_text(encoding="utf-8"))
    argv = [str(p)] + [f"--{k.replace('_', '-')}={v}" for k, v in D150_CLI.items()] + ["--write"]
    assert mod.main(argv) == 0
    out = capsys.readouterr().out
    assert f"written to   {p}" in out and "FLOPS via Aviary" in out

    root = ET.fromstring(p.read_text(encoding="utf-8"))
    node = root.find("vehicles/aircraft/model/analysisResults/massProperties/oewEstimateFlops")
    assert node is not None
    oew = float(node.findtext("oewKg"))
    assert 25_000 < oew < 60_000
    assert node.findtext("method").startswith("FLOPS via Aviary ")
    assert float(node.findtext("breakdown/structure/wingKg")) > 0
    assert float(node.findtext("breakdown/operatingMassKg")) == oew
    inputs = {i.findtext("name"): i for i in node.find("inputs")}
    assert inputs["mission:design:gross_mass"].findtext("source") == "cli:--design-gross-weight-kg"
    assert inputs["aircraft:wing:area"].findtext("source") == "cpacs://vehicles/aircraft/model/reference/area"
    assert len(node.find("caveats")) >= 3
    for path in ("vehicles/aircraft/model/reference/area", "vehicles/aircraft/model/wings",
                 "vehicles/aircraft/model/analysisResults/aero"):
        assert ET.tostring(root.find(path)) == ET.tostring(before.find(path))

    # Re-running replaces the estimate rather than adding a second one; --out leaves the input alone.
    assert mod.main(argv) == 0
    capsys.readouterr()
    root = ET.fromstring(p.read_text(encoding="utf-8"))
    assert len(root.findall("vehicles/aircraft/model/analysisResults/massProperties/oewEstimateFlops")) == 1
    original = p.read_text(encoding="utf-8")
    out_path = tmp_path / "out.xml"
    assert mod.main(argv[:-1] + ["--out", str(out_path), "--json"]) == 0
    assert p.read_text(encoding="utf-8") == original
    assert json.loads(capsys.readouterr().out)["written_to"] == str(out_path)


@needs_aviary
@pytest.mark.skipif(not _D150.is_file(), reason="public D150_v30.xml is not beside the repos")
def test_public_d150_runs_with_the_journal_cli_values() -> None:
    mod = _load()
    est = mod.estimate_oew(_D150, **D150_CLI)
    assert est["method"].startswith("FLOPS via Aviary ")
    sources = {r["name"]: r["source"] for r in est["inputs"]}
    assert sources["aircraft:wing:span"].startswith("cpacs:") and sources["mission:design:gross_mass"] == "cli:--design-gross-weight-kg"
    values = {r["name"]: r["value"] for r in est["inputs"]}
    assert values["aircraft:wing:span"] == pytest.approx(33.93, abs=0.01)  # the SU2 server's span for this file
    assert values["aircraft:fuselage:max_width"] == pytest.approx(3.95, abs=0.01)
    assert 30_000 < est["oew_kg"] < 45_000
