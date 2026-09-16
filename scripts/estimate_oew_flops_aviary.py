#!/usr/bin/env python3
"""Operating empty weight from NASA Aviary's FLOPS-based mass equations.

This is the second, independent OEW estimator, beside the wetted-area rule in
estimate_oew_from_wetted_area.py. Where that script multiplies one measured
area by one class constant, this one runs Aviary's FLOPS mass build-up: an
OpenMDAO problem made of Aviary's own groups

    aviary.subsystems.geometry.flops_based.prep_geom.PrepGeom
        FLOPS derived geometry: fuselage average diameter and planform area,
        wing / tail / fuselage / nacelle wetted areas, total wetted area
    aviary.subsystems.mass.flops_based.mass_premission.MassPremission
        every FLOPS transport mass component (wing, tails, fuselage, gear,
        nacelles, engines, fuel system, systems and equipment, operating
        items) and the MassSummation that adds them up

with the options wired through aviary.variable_info.functions.setup_model_options
and Aviary's own crew-count rules from aviary.utils.preprocessors.
No mass equation is typed here. Every number in the breakdown is an output of
an Aviary component; this script only reads the geometry out of the CPACS file,
converts units, and records where each input came from.

The OEW reported is Aviary's Aircraft.Design.OPERATING_MASS: empty mass plus
crew, unusable fuel, engine oil, passenger service items and cargo containers.
Aircraft.Design.EMPTY_MASS is reported beside it.

Inputs come from the CPACS file wherever the file states them:

    reference/area                                     wing reference area
    wings/wing (sections, positionings, airfoils)      span, taper, quarter-chord
                                                       sweep, dihedral, t/c, tail
                                                       areas, aspect ratios
    fuselages/fuselage (sections, profiles)            length, max width, max height
    analyses/massBreakdown/designMasses/mTOM/mass      design gross mass
    global/payload/paxSeats/{actual,required}          passenger count (CPACS 3)
    global/paxSeats                                    passenger count (CPACS 2)
    global/designRange/{actual,required}               design range
    engines/engine + vehicles/engines/engine/analysis  engine count, SLS thrust, mass
    analysisResults/aero/mach                          cruise Mach of the recorded
                                                       SU2 run

Whatever FLOPS needs and the file does not state must be given on the command
line (--design-gross-weight-kg, --passengers, --design-range-km, engine and
nacelle data, --passenger-compartment-length-m). There is no default for any
aircraft property: a missing one is a structured error
{"error": {"type": "missing_input", ...}} naming the flag. Only FLOPS method
constants (ultimate load factor 3.75, flap area ratio 0.333, hydraulic pressure
3000 psi, wing fuel capacity factor 23, mass scalers 1.0) are set to the values
the FLOPS manual documents as its defaults; each is listed in the output with
source "flops_default" and can be changed with --set.

Usage:
    estimate_oew_flops_aviary.py CPACS.xml --design-gross-weight-kg 73500 --passengers 150 \\
        --design-range-km 5000 --num-engines 2 --engine-sls-thrust-kn 120 --engine-mass-kg 2380 \\
        --nacelle-diameter-m 2.1 --nacelle-length-m 4.4 --engine-spanwise-fraction 0.34 \\
        --passenger-compartment-length-m 27.5 [--max-mach 0.82] [--json] [--write [--out NEW.xml]]

``--write`` records the estimate at
//vehicles/aircraft/model/analysisResults/massProperties/oewEstimateFlops
(replacing an earlier oewEstimateFlops there) and touches no other node.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

KG_PER_LB = 0.45359237
M_PER_FT = 0.3048
M_PER_NMI = 1852.0

MODEL_PATH = "vehicles/aircraft/model"
REF_AREA_NODE = "reference/area"
MTOM_NODE = "analyses/massBreakdown/designMasses/mTOM/mass"
PAX_NODES = ("global/payload/paxSeats/actual", "global/payload/paxSeats/required", "global/paxSeats")
RANGE_NODES = ("global/designRange/actual", "global/designRange/required")
AERO_MACH_NODE = "analysisResults/aero/mach"
ESTIMATE_PARENT = ("analysisResults", "massProperties")
ESTIMATE_TAG = "oewEstimateFlops"

# FLOPS method constants, set to the values the FLOPS user's guide documents as
# defaults (WTIN namelist). They describe the method, not the aircraft; the
# aircraft's own properties never get a default here.
FLOPS_DEFAULTS: dict[str, tuple[float, str, str]] = {
    "aircraft:wing:ultimate_load_factor": (3.75, "unitless", "WTIN.ULF default 3.75"),
    "aircraft:wing:control_surface_area_ratio": (0.333, "unitless", "WTIN.FLAPR default 0.333"),
    "aircraft:hydraulics:system_pressure": (3000.0, "psi", "WTIN.HYDPR default 3000 psi"),
    "aircraft:fuel:capacity_factor": (23.0, "unitless", "WTIN.FWMAX default 23 (wing fuel capacity formula)"),
}

# CPACS files carry xsi:noNamespaceSchemaLocation; keep the prefix on rewrite.
ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")


class EstimateError(ValueError):
    """A structured refusal, printable as {"error": {"type": ..., "message": ...}}."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"type": self.error_type, "message": self.message}}


def _xpath(node: str) -> str:
    return f"//{MODEL_PATH}/{node}"


def _import_aviary() -> dict[str, Any]:
    """Import the Aviary pieces this script runs. A structured error when Aviary is absent."""
    try:
        import aviary
        import numpy as np
        import openmdao.api as om
        from aviary.subsystems.geometry.flops_based.prep_geom import PrepGeom
        from aviary.subsystems.mass.flops_based.mass_premission import MassPremission
        from aviary.utils.aviary_values import AviaryValues
        from aviary.utils.preprocessors import preprocess_crewpayload
        from aviary.variable_info.enums import LegacyCode, Verbosity
        from aviary.variable_info.functions import override_aviary_vars, setup_model_options
        from aviary.variable_info.variable_meta_data import _MetaData as MetaData
        from aviary.variable_info.variables import Aircraft, Mission, Settings
    except ImportError as exc:  # pragma: no cover - exercised by monkeypatching in tests
        raise EstimateError(
            "missing_dependency",
            f"NASA Aviary (with OpenMDAO) is not importable: {exc}. Install it into the project "
            "venv with `pip install aviary` (https://github.com/OpenMDAO/Aviary); this estimator "
            "runs Aviary's FLOPS mass groups and has no fallback.",
        ) from exc
    return {
        "aviary": aviary, "np": np, "om": om, "PrepGeom": PrepGeom, "MassPremission": MassPremission,
        "AviaryValues": AviaryValues, "preprocess_crewpayload": preprocess_crewpayload,
        "LegacyCode": LegacyCode, "Verbosity": Verbosity, "override_aviary_vars": override_aviary_vars,
        "setup_model_options": setup_model_options, "Aircraft": Aircraft, "Mission": Mission,
        "Settings": Settings, "MetaData": MetaData,
    }


# --------------------------------------------------------------------------- CPACS reading


def load_cpacs(path: Path | str) -> ET.Element:
    p = Path(path)
    if not p.is_file():
        raise EstimateError("missing_input", f"CPACS file not found: {p}")
    try:
        return ET.parse(p).getroot()
    except ET.ParseError as exc:
        raise EstimateError("bad_input", f"{p} is not well-formed XML: {exc}") from exc


def _float_text(el: ET.Element | None, what: str, label: str, positive: bool = True) -> float | None:
    """Float text of ``el``; None when absent; error when unusable."""
    if el is None:
        return None
    text = (el.text or "").strip()
    if not text:
        return None
    try:
        val = float(text)
    except ValueError:
        raise EstimateError("bad_input", f"{label}: {what} is '{text}', not a number") from None
    if positive and not val > 0:  # also rejects NaN
        raise EstimateError("bad_input", f"{label}: {what} is {val}; it must be positive")
    return val


def _xyz(el: ET.Element | None) -> tuple[float, float, float]:
    if el is None:
        return (0.0, 0.0, 0.0)
    return tuple(float(el.findtext(k) or 0.0) for k in ("x", "y", "z"))  # type: ignore[return-value]


def _scale(el: ET.Element | None) -> tuple[float, float, float]:
    if el is None:
        return (1.0, 1.0, 1.0)
    return tuple(float(el.findtext(k) or 1.0) for k in ("x", "y", "z"))  # type: ignore[return-value]


def _resolve_section_positions(container: ET.Element) -> dict[str, tuple[float, float, float]]:
    """Section origins in the component's local frame.

    CPACS places a section through a chain of ``positionings`` (length, sweep,
    dihedral from a ``fromSectionUID``) and/or the section's own translation and
    its first element's translation; all three are summed here. Same resolution
    as the SU2 server's span helper, extended to x and z.
    """
    trans: dict[str, tuple[float, float, float]] = {}
    for sec in container.findall("sections/section"):
        uid = sec.get("uID") or ""
        t = _xyz(sec.find("transformation/translation"))
        e = _xyz(sec.find("elements/element/transformation/translation"))
        trans[uid] = (t[0] + e[0], t[1] + e[1], t[2] + e[2])

    pos: dict[str, tuple[float, float, float]] = {}
    pending = list(container.findall("positionings/positioning"))
    for _ in range(len(pending) + 1):
        if not pending:
            break
        rest = []
        for pz in pending:
            to_uid = pz.findtext("toSectionUID") or ""
            from_uid = pz.findtext("fromSectionUID")
            base = (0.0, 0.0, 0.0) if not from_uid else pos.get(from_uid)
            if base is None:
                rest.append(pz)
                continue
            length = float(pz.findtext("length") or 0.0)
            sweep = math.radians(float(pz.findtext("sweepAngle") or 0.0))
            dihedral = math.radians(float(pz.findtext("dihedralAngle") or 0.0))
            pos[to_uid] = (
                base[0] + length * math.sin(sweep),
                base[1] + length * math.cos(sweep) * math.cos(dihedral),
                base[2] + length * math.cos(sweep) * math.sin(dihedral),
            )
        pending = rest

    out: dict[str, tuple[float, float, float]] = {}
    for uid in trans:
        p = pos.get(uid, (0.0, 0.0, 0.0))
        t = trans[uid]
        out[uid] = (p[0] + t[0], p[1] + t[1], p[2] + t[2])
    return out


def _profile_points(root: ET.Element, kind: str, uid: str, label: str) -> dict[str, list[float]]:
    """Point list of a wingAirfoil or fuselageProfile by uID."""
    node_name = "wingAirfoil" if kind == "airfoil" else "fuselageProfile"
    parent = "wingAirfoils" if kind == "airfoil" else "fuselageProfiles"
    for prof in root.iterfind(f"vehicles/profiles/{parent}/{node_name}"):
        if prof.get("uID") == uid:
            pts = {}
            for axis in ("x", "y", "z"):
                text = prof.findtext(f"pointList/{axis}") or ""
                pts[axis] = [float(v) for v in text.split(";") if v.strip()]
            if not pts["x"] and not pts["y"]:
                raise EstimateError("bad_input", f"{label}: profile {uid} has no points")
            return pts
    raise EstimateError("missing_input", f"{label}: profile {uid} is referenced but not defined under vehicles/profiles")


def airfoil_thickness_ratio(xs: list[float], zs: list[float]) -> float:
    """Maximum thickness / chord of an airfoil given as one closed point loop.

    The loop is split at the leading edge (minimum x) into two surfaces; the
    lower surface is interpolated onto the upper surface's stations and the
    largest vertical gap is the thickness. For NACA 0012 this returns 0.1200.
    """
    if len(xs) < 4 or len(xs) != len(zs):
        raise EstimateError("bad_input", "airfoil point list is too short or inconsistent")
    chord = max(xs) - min(xs)
    if chord <= 0:
        raise EstimateError("bad_input", "airfoil has zero chord")
    i_le = min(range(len(xs)), key=lambda i: xs[i])
    a = sorted(zip(xs[: i_le + 1], zs[: i_le + 1]))
    b = sorted(zip(xs[i_le:], zs[i_le:]))
    if len(a) < 2 or len(b) < 2:
        return (max(zs) - min(zs)) / chord

    def interp(curve: list[tuple[float, float]], x: float) -> float:
        for (x0, z0), (x1, z1) in zip(curve, curve[1:]):
            if x0 <= x <= x1:
                return z0 if x1 == x0 else z0 + (z1 - z0) * (x - x0) / (x1 - x0)
        return curve[0][1] if x < curve[0][0] else curve[-1][1]

    lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    thickness = max(abs(interp(a, x) - interp(b, x)) for x, _ in a + b if lo <= x <= hi)
    return thickness / chord


def _wing_sections(root: ET.Element, wing: ET.Element, label: str) -> list[dict[str, float]]:
    """Per-section origin (local frame), chord and thickness ratio, sorted outboard."""
    positions = _resolve_section_positions(wing)
    rows = []
    for sec in wing.findall("sections/section"):
        uid = sec.get("uID") or ""
        el = sec.find("elements/element")
        if el is None:
            raise EstimateError("bad_input", f"{label}: wing section {uid} has no element")
        ssc = _scale(sec.find("transformation/scaling"))
        esc = _scale(el.find("transformation/scaling"))
        airfoil_uid = el.findtext("airfoilUID") or ""
        pts = _profile_points(root, "airfoil", airfoil_uid, label)
        if not pts["x"] or not pts["z"]:
            raise EstimateError("bad_input", f"{label}: airfoil {airfoil_uid} lacks x/z points")
        profile_chord = max(pts["x"]) - min(pts["x"])
        tc_airfoil = airfoil_thickness_ratio(pts["x"], pts["z"])
        cx = ssc[0] * esc[0]
        cz = ssc[2] * esc[2]
        chord = profile_chord * cx
        if chord <= 0:
            raise EstimateError("bad_input", f"{label}: wing section {uid} has zero chord")
        x, y, z = positions[uid]
        rows.append({"x": x, "y": y, "z": z, "chord": chord, "tc": tc_airfoil * cz / cx})
    if len(rows) < 2:
        raise EstimateError("bad_input", f"{label}: wing {wing.get('uID')} has fewer than two sections")
    rows.sort(key=lambda r: abs(r["y"]))
    return rows


def _planform_area(rows: list[dict[str, float]], symmetric: bool) -> float:
    area = 0.0
    for r0, r1 in zip(rows, rows[1:]):
        area += 0.5 * (r0["chord"] + r1["chord"]) * abs(r1["y"] - r0["y"])
    return 2.0 * area if symmetric else area


def _weighted_tc(rows: list[dict[str, float]]) -> float:
    num = den = 0.0
    for r0, r1 in zip(rows, rows[1:]):
        a = 0.5 * (r0["chord"] + r1["chord"]) * abs(r1["y"] - r0["y"])
        num += a * 0.5 * (r0["tc"] + r1["tc"])
        den += a
    return num / den if den > 0 else rows[0]["tc"]


def _wing_rotation_x_deg(wing: ET.Element) -> float:
    rot = wing.find("transformation/rotation")
    return float(rot.findtext("x") or 0.0) if rot is not None else 0.0


def read_lifting_surfaces(root: ET.Element, label: str, fuselage_half_width_m: float | None) -> dict[str, Any]:
    """Main wing, horizontal tail and vertical tail geometry from wings/wing.

    Classification: a wing rotated about x by about 90 degrees is a vertical
    tail; of the rest, the largest planform is the main wing and the other is
    the horizontal tail. Anything else (canards, several horizontal tails) is
    refused rather than guessed.
    """
    model = root.find(MODEL_PATH)
    wings = list(model.iterfind("wings/wing")) if model is not None else []
    if not wings:
        raise EstimateError("missing_input", f"{label}: no //{MODEL_PATH}/wings/wing; FLOPS needs the wing geometry")

    horizontal, vertical = [], []
    for w in wings:
        rows = _wing_sections(root, w, label)
        symmetric = bool(w.get("symmetry"))
        rx = _wing_rotation_x_deg(w)
        trans = _xyz(w.find("transformation/translation"))
        entry = {"uid": w.get("uID") or "", "rows": rows, "symmetric": symmetric, "rx_deg": rx,
                 "translation": trans, "area": _planform_area(rows, symmetric)}
        (vertical if abs(math.sin(math.radians(rx))) > math.cos(math.radians(45.0)) else horizontal).append(entry)

    if not horizontal:
        raise EstimateError("missing_input", f"{label}: no horizontal lifting surface found among wings/wing")
    horizontal.sort(key=lambda e: e["area"], reverse=True)
    main, tails = horizontal[0], horizontal[1:]
    if len(tails) > 1:
        raise EstimateError(
            "bad_input",
            f"{label}: {len(tails)} horizontal surfaces besides the main wing "
            f"({', '.join(t['uid'] for t in tails)}); this estimator handles one horizontal tail only",
        )
    if not tails:
        raise EstimateError("missing_input", f"{label}: no horizontal tail among wings/wing; FLOPS needs its area")
    if not vertical:
        raise EstimateError("missing_input", f"{label}: no vertical tail (wing rotated about x by ~90 deg) among wings/wing")

    rows = main["rows"]
    root_sec, tip_sec = rows[0], rows[-1]
    half_span = abs(tip_sec["y"])
    span = 2.0 * half_span if main["symmetric"] else abs(tip_sec["y"] - root_sec["y"])
    if span <= 0:
        raise EstimateError("bad_input", f"{label}: main wing {main['uid']} has zero span")
    # Reference trapezoid: quarter-chord line from the section at the fuselage
    # side (outermost section within the fuselage half-width, else the root) to the tip.
    side_sec = root_sec
    if fuselage_half_width_m is not None:
        inside = [r for r in rows[:-1] if abs(r["y"]) <= fuselage_half_width_m]
        if inside:
            side_sec = inside[-1]
    dy = abs(tip_sec["y"]) - abs(side_sec["y"])
    if dy <= 0:
        raise EstimateError("bad_input", f"{label}: main wing sections do not extend outboard")
    dx_qc = (tip_sec["x"] + 0.25 * tip_sec["chord"]) - (side_sec["x"] + 0.25 * side_sec["chord"])
    sweep_deg = math.degrees(math.atan2(dx_qc, dy))
    dihedral_deg = math.degrees(math.atan2(tip_sec["z"] - side_sec["z"], dy))

    ht = tails[0]
    ht_rows = ht["rows"]
    ht_span = 2.0 * abs(ht_rows[-1]["y"]) if ht["symmetric"] else abs(ht_rows[-1]["y"] - ht_rows[0]["y"])
    vt = vertical[0]
    vt_rows = vt["rows"]
    vt_height = abs(vt_rows[-1]["y"] - vt_rows[0]["y"])
    vt_area_each = sum(v["area"] for v in vertical) / len(vertical)
    if vt_height <= 0 or vt_area_each <= 0 or ht_span <= 0 or ht["area"] <= 0:
        raise EstimateError("bad_input", f"{label}: a tail surface has zero span or area")

    # Horizontal tail mount fraction on the vertical tail (FLOPS HHT): 0 body-mounted, 1 T-tail.
    rx = math.radians(vt["rx_deg"])
    vt_root_z = vt["translation"][2] + vt_rows[0]["y"] * math.sin(rx) + vt_rows[0]["z"] * math.cos(rx)
    vt_tip_z = vt["translation"][2] + vt_rows[-1]["y"] * math.sin(rx) + vt_rows[-1]["z"] * math.cos(rx)
    ht_root_z = ht["translation"][2] + ht_rows[0]["z"]
    hht = 0.0
    if vt_tip_z != vt_root_z:
        hht = min(1.0, max(0.0, (ht_root_z - vt_root_z) / (vt_tip_z - vt_root_z)))

    def wpath(uid: str) -> str:
        return _xpath(f"wings/wing[@uID='{uid}']")

    return {
        "wing_uid": main["uid"],
        "wing_span_m": span,
        "wing_taper_ratio": tip_sec["chord"] / root_sec["chord"],
        "wing_sweep_qc_deg": sweep_deg,
        "wing_dihedral_deg": dihedral_deg,
        "wing_tc": _weighted_tc(rows),
        "wing_planform_area_m2": main["area"],
        "wing_xpath": wpath(main["uid"]),
        "wing_sweep_definition": (
            f"quarter-chord line from section at |y|={abs(side_sec['y']):.3f} m "
            f"(fuselage side) to tip at |y|={abs(tip_sec['y']):.3f} m"
        ),
        "htail_uid": ht["uid"],
        "htail_area_m2": ht["area"],
        "htail_aspect_ratio": ht_span**2 / ht["area"],
        "htail_taper_ratio": ht_rows[-1]["chord"] / ht_rows[0]["chord"],
        "htail_tc": _weighted_tc(ht_rows),
        "htail_xpath": wpath(ht["uid"]),
        "vtail_uids": [v["uid"] for v in vertical],
        "vtail_count": len(vertical),
        "vtail_area_each_m2": vt_area_each,
        "vtail_aspect_ratio": vt_height**2 / vt["area"],
        "vtail_taper_ratio": vt_rows[-1]["chord"] / vt_rows[0]["chord"],
        "vtail_tc": _weighted_tc(vt_rows),
        "vtail_xpath": wpath(vt["uid"]),
        "htail_on_vtail_fraction": hht,
    }


def read_fuselage(root: ET.Element, label: str) -> dict[str, Any]:
    """Length, maximum width and maximum height of the fuselage from its sections."""
    model = root.find(MODEL_PATH)
    fuselages = list(model.iterfind("fuselages/fuselage")) if model is not None else []
    if not fuselages:
        raise EstimateError("missing_input", f"{label}: no //{MODEL_PATH}/fuselages/fuselage; FLOPS needs the fuselage")
    fus = fuselages[0]
    uid = fus.get("uID") or ""
    positions = _resolve_section_positions(fus)
    xs, widths, heights = [], [], []
    for sec in fus.findall("sections/section"):
        suid = sec.get("uID") or ""
        el = sec.find("elements/element")
        if el is None:
            continue
        ssc = _scale(sec.find("transformation/scaling"))
        esc = _scale(el.find("transformation/scaling"))
        pts = _profile_points(root, "fuselage", el.findtext("profileUID") or "", label)
        if not pts["y"] or not pts["z"]:
            raise EstimateError("bad_input", f"{label}: fuselage profile of section {suid} lacks y/z points")
        widths.append((max(pts["y"]) - min(pts["y"])) * ssc[1] * esc[1])
        heights.append((max(pts["z"]) - min(pts["z"])) * ssc[2] * esc[2])
        xs.append(positions[suid][0])
    if len(xs) < 2:
        raise EstimateError("bad_input", f"{label}: fuselage {uid} has fewer than two usable sections")
    length = max(xs) - min(xs)
    if length <= 0 or max(widths) <= 0 or max(heights) <= 0:
        raise EstimateError("bad_input", f"{label}: fuselage {uid} resolves to zero length, width or height")
    return {
        "fuselage_uid": uid,
        "fuselage_length_m": length,
        "fuselage_max_width_m": max(widths),
        "fuselage_max_height_m": max(heights),
        "fuselage_count": len(fuselages),
        "fuselage_xpath": _xpath(f"fuselages/fuselage[@uID='{uid}']"),
    }


def read_engines(root: ET.Element, label: str) -> dict[str, Any] | None:
    """Engine count, SLS thrust and dry mass when the file states them, else None.

    Count from //vehicles/aircraft/model/engines/engine (doubled for a symmetry
    attribute); thrust00 [N] and mass [kg] from the referenced
    //vehicles/engines/engine/analysis.
    """
    model = root.find(MODEL_PATH)
    placements = list(model.iterfind("engines/engine")) if model is not None else []
    if not placements:
        return None
    count = 0
    engine_uids = set()
    for pl in placements:
        count += 2 if pl.get("symmetry") else 1
        engine_uids.add(pl.findtext("engineUID") or "")
    if len(engine_uids) != 1:
        raise EstimateError("bad_input", f"{label}: engines of several types ({sorted(engine_uids)}); one engine type is supported")
    (euid,) = engine_uids
    thrust_n = mass_kg = None
    for eng in root.iterfind("vehicles/engines/engine"):
        if eng.get("uID") == euid:
            thrust_n = _float_text(eng.find("analysis/thrust00"), "thrust00", label)
            mass_kg = _float_text(eng.find("analysis/mass/mass"), "engine mass", label)
    out: dict[str, Any] = {"count": count, "count_xpath": _xpath("engines/engine")}
    if thrust_n is not None:
        out["sls_thrust_n"] = thrust_n
        out["thrust_xpath"] = f"//vehicles/engines/engine[@uID='{euid}']/analysis/thrust00"
    if mass_kg is not None:
        out["mass_kg"] = mass_kg
        out["mass_xpath"] = f"//vehicles/engines/engine[@uID='{euid}']/analysis/mass/mass"
    return out


# --------------------------------------------------------------------------- input assembly


class _Inputs:
    """Collects Aviary inputs with provenance; refuses when a required one is missing."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.rows: list[dict[str, Any]] = []
        self.values: dict[str, tuple[Any, str]] = {}

    def add(self, name: str, value: Any, units: str, source: str, note: str = "") -> None:
        self.values[name] = (value, units)
        row = {"name": name, "value": value, "units": units, "source": source}
        if note:
            row["note"] = note
        self.rows.append(row)

    def resolve(self, name: str, units: str, *, file_value: float | None, file_xpath: str | None,
                cli_value: float | None, cli_flag: str, what: str, convert=lambda v: v) -> float:
        if file_value is not None and cli_value is not None:
            raise EstimateError(
                "conflicting_input",
                f"{self.label} states {what} at {file_xpath} ({file_value:g}) and {cli_flag} was also given "
                f"({cli_value:g}); remove the flag to use the file's value or edit the file.",
            )
        if file_value is not None:
            self.add(name, convert(file_value), units, f"cpacs:{file_xpath}")
            return convert(file_value)
        if cli_value is not None:
            self.add(name, convert(cli_value), units, f"cli:{cli_flag}")
            return convert(cli_value)
        raise EstimateError(
            "missing_input",
            f"{self.label} does not state {what}"
            + (f" (looked at {file_xpath})" if file_xpath else "")
            + f" and {cli_flag} was not given. FLOPS needs it; there is no default.",
        )


def _parse_set(items: list[str] | None) -> dict[str, tuple[float, str]]:
    """--set NAME=VALUE[:UNITS] overrides for Aviary inputs."""
    out: dict[str, tuple[float, str]] = {}
    for item in items or []:
        if "=" not in item:
            raise EstimateError("bad_input", f"--set expects NAME=VALUE[:UNITS], got '{item}'")
        name, rest = item.split("=", 1)
        val_txt, _, units = rest.partition(":")
        try:
            val = float(val_txt)
        except ValueError:
            raise EstimateError("bad_input", f"--set {name}: '{val_txt}' is not a number") from None
        out[name.strip()] = (val, units.strip() or "unitless")
    return out


def collect_inputs(root: ET.Element, label: str, cli: dict[str, Any]) -> tuple[_Inputs, dict[str, Any]]:
    """Read the CPACS geometry, merge the CLI values, and refuse on gaps or conflicts."""
    model = root.find(MODEL_PATH)
    if model is None:
        raise EstimateError("missing_input", f"{label}: no //{MODEL_PATH} element")
    inp = _Inputs(label)

    ref_area = _float_text(model.find(REF_AREA_NODE), "reference area", label)
    if ref_area is None:
        raise EstimateError("missing_input", f"{label} states no wing reference area at {_xpath(REF_AREA_NODE)}")
    inp.add("aircraft:wing:area", ref_area, "m**2", f"cpacs:{_xpath(REF_AREA_NODE)}")

    fus = read_fuselage(root, label)
    inp.add("aircraft:fuselage:length", fus["fuselage_length_m"], "m", f"cpacs:{fus['fuselage_xpath']}",
            "nose-to-tail extent of the resolved section positions")
    inp.add("aircraft:fuselage:max_width", fus["fuselage_max_width_m"], "m", f"cpacs:{fus['fuselage_xpath']}",
            "largest scaled profile width over all sections")
    inp.add("aircraft:fuselage:max_height", fus["fuselage_max_height_m"], "m", f"cpacs:{fus['fuselage_xpath']}",
            "largest scaled profile height over all sections")

    surf = read_lifting_surfaces(root, label, 0.5 * fus["fuselage_max_width_m"])
    inp.add("aircraft:wing:span", surf["wing_span_m"], "m", f"cpacs:{surf['wing_xpath']}",
            "from resolved section positions; aspect ratio = span^2 / reference area")
    inp.add("aircraft:wing:taper_ratio", surf["wing_taper_ratio"], "unitless", f"cpacs:{surf['wing_xpath']}",
            "tip chord / centreline root chord")
    inp.add("aircraft:wing:sweep", surf["wing_sweep_qc_deg"], "deg", f"cpacs:{surf['wing_xpath']}",
            surf["wing_sweep_definition"])
    inp.add("aircraft:wing:dihedral", surf["wing_dihedral_deg"], "deg", f"cpacs:{surf['wing_xpath']}",
            "between the same two sections as the sweep")
    inp.add("aircraft:wing:thickness_to_chord", surf["wing_tc"], "unitless", f"cpacs:{surf['wing_xpath']}",
            "planform-area-weighted airfoil thickness ratio of the sections")
    inp.add("aircraft:horizontal_tail:area", surf["htail_area_m2"], "m**2", f"cpacs:{surf['htail_xpath']}")
    inp.add("aircraft:horizontal_tail:aspect_ratio", surf["htail_aspect_ratio"], "unitless", f"cpacs:{surf['htail_xpath']}")
    inp.add("aircraft:horizontal_tail:taper_ratio", surf["htail_taper_ratio"], "unitless", f"cpacs:{surf['htail_xpath']}")
    inp.add("aircraft:horizontal_tail:thickness_to_chord", surf["htail_tc"], "unitless", f"cpacs:{surf['htail_xpath']}")
    inp.add("aircraft:horizontal_tail:vertical_tail_fraction", surf["htail_on_vtail_fraction"], "unitless",
            f"cpacs:{surf['htail_xpath']}", "0 body-mounted, 1 T-tail, from the tails' z placement")
    inp.add("aircraft:vertical_tail:area", surf["vtail_area_each_m2"], "m**2", f"cpacs:{surf['vtail_xpath']}",
            "per tail")
    inp.add("aircraft:vertical_tail:aspect_ratio", surf["vtail_aspect_ratio"], "unitless", f"cpacs:{surf['vtail_xpath']}")
    inp.add("aircraft:vertical_tail:taper_ratio", surf["vtail_taper_ratio"], "unitless", f"cpacs:{surf['vtail_xpath']}")
    inp.add("aircraft:vertical_tail:thickness_to_chord", surf["vtail_tc"], "unitless", f"cpacs:{surf['vtail_xpath']}")
    inp.add("aircraft:vertical_tail:num_tails", surf["vtail_count"], "unitless", f"cpacs:{surf['vtail_xpath']}")
    inp.add("aircraft:fuselage:num_fuselages", fus["fuselage_count"], "unitless", f"cpacs:{_xpath('fuselages/fuselage')}")

    inp.resolve("mission:design:gross_mass", "kg", file_value=_float_text(model.find(MTOM_NODE), "mTOM", label),
                file_xpath=_xpath(MTOM_NODE), cli_value=cli.get("design_gross_weight_kg"),
                cli_flag="--design-gross-weight-kg", what="a design gross mass")

    pax_val = pax_node = None
    for node in PAX_NODES:
        pax_val = _float_text(model.find(node), "passenger count", label)
        if pax_val is not None:
            pax_node = _xpath(node)
            break
    pax = inp.resolve("aircraft:crew_and_payload:design:num_passengers", "unitless", file_value=pax_val,
                      file_xpath=pax_node or _xpath(PAX_NODES[0]), cli_value=cli.get("passengers"),
                      cli_flag="--passengers", what="a passenger count", convert=lambda v: int(round(v)))

    range_val = range_node = None
    for node in RANGE_NODES:
        range_val = _float_text(model.find(node), "design range", label)
        if range_val is not None:
            range_node = _xpath(node)
            break
    if range_val is not None:
        range_km = inp.resolve("mission:design:range", "km", file_value=range_val, file_xpath=range_node,
                               cli_value=cli.get("design_range_km"), cli_flag="--design-range-km",
                               what="a design range", convert=lambda v: v / 1000.0)
    else:
        range_km = inp.resolve("mission:design:range", "km", file_value=None, file_xpath=_xpath(RANGE_NODES[0]),
                               cli_value=cli.get("design_range_km"), cli_flag="--design-range-km",
                               what="a design range")

    mach_file = _float_text(model.find(AERO_MACH_NODE), "mach", label)
    cruise_mach = inp.resolve("mission:summary:cruise_mach", "unitless", file_value=mach_file,
                              file_xpath=_xpath(AERO_MACH_NODE), cli_value=cli.get("cruise_mach"),
                              cli_flag="--cruise-mach", what="a cruise Mach number")
    if cli.get("max_mach") is not None:
        inp.add("mission:constraints:max_mach", float(cli["max_mach"]), "unitless", "cli:--max-mach")
    else:
        inp.add("mission:constraints:max_mach", cruise_mach, "unitless", "aviary_default",
                "Aviary documents Mission.Constraints.MAX_MACH as the cruise Mach number; give --max-mach to state MMO")

    eng = read_engines(root, label)
    num_engines = inp.resolve("aircraft:engine:num_engines", "unitless", file_value=eng["count"] if eng else None,
                              file_xpath=eng["count_xpath"] if eng else _xpath("engines/engine"),
                              cli_value=cli.get("num_engines"), cli_flag="--num-engines",
                              what="an engine count", convert=lambda v: int(round(v)))
    thrust_n = inp.resolve("aircraft:engine:scaled_sls_thrust", "N",
                           file_value=eng.get("sls_thrust_n") if eng else None,
                           file_xpath=eng.get("thrust_xpath") if eng else "//vehicles/engines/engine/analysis/thrust00",
                           cli_value=cli.get("engine_sls_thrust_kn"), cli_flag="--engine-sls-thrust-kn",
                           what="the sea-level static thrust per engine",
                           convert=(lambda v: v) if (eng and "sls_thrust_n" in eng) else (lambda v: v * 1000.0))
    inp.resolve("aircraft:engine:reference_mass", "kg", file_value=eng.get("mass_kg") if eng else None,
                file_xpath=eng.get("mass_xpath") if eng else "//vehicles/engines/engine/analysis/mass/mass",
                cli_value=cli.get("engine_mass_kg"), cli_flag="--engine-mass-kg",
                what="the dry mass per engine")
    mount = cli.get("engine_mount") or "wing"
    inp.add("engine_mount", mount, "", "cli:--engine-mount" if cli.get("engine_mount") else "cli:--engine-mount (wing, the FLOPS transport default)")
    inp.add("aircraft:propulsion:total_scaled_sls_thrust", num_engines * thrust_n, "N", "derived",
            "num_engines x sls thrust per engine")
    inp.resolve("aircraft:nacelle:avg_diameter", "m", file_value=None, file_xpath=None,
                cli_value=cli.get("nacelle_diameter_m"), cli_flag="--nacelle-diameter-m",
                what="the nacelle average diameter")
    inp.resolve("aircraft:nacelle:avg_length", "m", file_value=None, file_xpath=None,
                cli_value=cli.get("nacelle_length_m"), cli_flag="--nacelle-length-m",
                what="the nacelle average length")
    if mount == "wing":
        inp.resolve("aircraft:engine:wing_locations", "unitless", file_value=None, file_xpath=None,
                    cli_value=cli.get("engine_spanwise_fraction"), cli_flag="--engine-spanwise-fraction",
                    what="the engine position as a fraction of semispan (main-gear length needs it)")
    inp.resolve("aircraft:fuselage:passenger_compartment_length", "m", file_value=None, file_xpath=None,
                cli_value=cli.get("passenger_compartment_length_m"), cli_flag="--passenger-compartment-length-m",
                what="the passenger compartment length (furnishings mass needs it)")
    if cli.get("fuel_capacity_kg") is not None:
        inp.add("aircraft:fuel:total_capacity", float(cli["fuel_capacity_kg"]), "kg", "cli:--fuel-capacity-kg")

    for name, (val, units, note) in FLOPS_DEFAULTS.items():
        inp.add(name, val, units, "flops_default", note)
    for name, (val, units) in _parse_set(cli.get("set")).items():
        inp.add(name, val, units, "cli:--set")

    extra = {"pax": pax, "range_km": range_km, "num_engines": num_engines, "mount": mount,
             "wing_planform_area_m2": surf["wing_planform_area_m2"], "vtail_count": surf["vtail_count"],
             "fuselage_count": fus["fuselage_count"]}
    return inp, extra


# --------------------------------------------------------------------------- Aviary run


def run_aviary(inp: _Inputs, extra: dict[str, Any]) -> dict[str, Any]:
    """Build and run Aviary's FLOPS PrepGeom + MassPremission on the collected inputs."""
    av = _import_aviary()
    np, om = av["np"], av["om"]
    Aircraft, Mission, Settings = av["Aircraft"], av["Mission"], av["Settings"]
    AviaryValues = av["AviaryValues"]
    vals = inp.values

    opts = AviaryValues()
    opts.set_val(Settings.MASS_METHOD, av["LegacyCode"].FLOPS)
    opts.set_val(Settings.VERBOSITY, av["Verbosity"].QUIET)
    pax = int(extra["pax"])
    for key in (Aircraft.CrewPayload.Design.NUM_PASSENGERS, Aircraft.CrewPayload.NUM_PASSENGERS,
                Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS, Aircraft.CrewPayload.NUM_TOURIST_CLASS):
        opts.set_val(key, pax)
    opts.set_val(Mission.Design.RANGE, float(extra["range_km"]), "km")
    av["preprocess_crewpayload"](opts)  # FLOPS crew-count and baggage-per-passenger rules, Aviary's code
    for key, note in ((Aircraft.CrewPayload.NUM_FLIGHT_CREW, "flight crew"),
                      (Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS, "flight attendants"),
                      (Aircraft.CrewPayload.NUM_GALLEY_CREW, "galley crew")):
        inp.add(key, int(opts.get_val(key)), "unitless", "aviary:preprocess_crewpayload", f"{note} from passenger count")
    inp.add(Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER, float(opts.get_val(Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER, "lbm")),
            "lbm", "aviary:preprocess_crewpayload", "FLOPS baggage allowance from design range")
    inp.add(Aircraft.CrewPayload.MASS_PER_PASSENGER, float(av["MetaData"][Aircraft.CrewPayload.MASS_PER_PASSENGER]["default_value"]),
            "lbm", "aviary_default", "FLOPS WPPASS default, from Aviary's variable metadata")

    n_eng = int(extra["num_engines"])
    n_wing = n_eng if extra["mount"] == "wing" else 0
    opts.set_val(Aircraft.Engine.NUM_ENGINES, np.array([n_eng]))
    opts.set_val(Aircraft.Engine.NUM_WING_ENGINES, np.array([n_wing]))
    opts.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, np.array([n_eng - n_wing]))
    opts.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, n_eng)
    opts.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, n_wing)
    opts.set_val(Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES, n_eng - n_wing)
    ref_mass, ref_units = vals["aircraft:engine:reference_mass"]
    opts.set_val(Aircraft.Engine.REFERENCE_MASS, np.array([ref_mass]), ref_units)
    thrust, thrust_units = vals["aircraft:engine:scaled_sls_thrust"]
    opts.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, np.array([thrust]), thrust_units)
    opts.set_val(Aircraft.Engine.SCALE_MASS, np.array([False]))  # engine mass = the stated dry mass
    opts.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, np.array([0.0]))
    opts.set_val(Mission.Constraints.MAX_MACH, float(vals["mission:constraints:max_mach"][0]))
    opts.set_val(Aircraft.Design.USE_ALT_MASS, False)
    opts.set_val(Aircraft.VerticalTail.NUM_TAILS, int(extra["vtail_count"]))
    opts.set_val(Aircraft.Fuselage.NUM_FUSELAGES, int(extra["fuselage_count"]))

    # Outputs of Aviary components that the caller states instead: Aviary's own
    # override mechanism renames the computed output and feeds the stated value.
    overrides = AviaryValues()
    overrides.set_val(Settings.VERBOSITY, av["Verbosity"].QUIET)
    if "aircraft:fuel:total_capacity" in vals:
        cap, cap_units = vals["aircraft:fuel:total_capacity"]
        overrides.set_val(Aircraft.Fuel.TOTAL_CAPACITY, cap, cap_units)
    else:
        # FLOPS defaults FULFMX = FULAUX = 0: no fuselage or auxiliary tanks, so the
        # total capacity is the wing capacity Aviary computes from the FWMAX formula.
        overrides.set_val(Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY, 0.0, "lbm")
        overrides.set_val(Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY, 0.0, "lbm")
        inp.add("aircraft:fuel:fuselage_fuel_capacity", 0.0, "kg", "flops_default", "WTIN.FULFMX default 0")
        inp.add("aircraft:fuel:auxiliary_fuel_capacity", 0.0, "kg", "flops_default", "WTIN.FULAUX default 0")

    PrepGeom, MassPremission, override_aviary_vars = av["PrepGeom"], av["MassPremission"], av["override_aviary_vars"]

    class _Premission(om.Group):
        def setup(self):
            self.add_subsystem("geometry", PrepGeom(), promotes_inputs=["*"], promotes_outputs=["*"])
            self.add_subsystem("mass", MassPremission(), promotes_inputs=["*"], promotes_outputs=["*"])

        def configure(self):
            override_aviary_vars(self, overrides)

    prob = om.Problem(reports=False)
    prob.model.add_subsystem("premission", _Premission(), promotes_inputs=["*"], promotes_outputs=["*"])
    av["setup_model_options"](prob, opts)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.setup(check=False)
            prob.final_setup()
            skip = {"engine_mount", "aircraft:engine:num_engines", "aircraft:engine:reference_mass",
                    "aircraft:crew_and_payload:design:num_passengers", "mission:constraints:max_mach",
                    "aircraft:vertical_tail:num_tails", "aircraft:fuselage:num_fuselages",
                    "aircraft:fuel:total_capacity", "aircraft:fuel:fuselage_fuel_capacity",
                    "aircraft:fuel:auxiliary_fuel_capacity", Aircraft.CrewPayload.NUM_FLIGHT_CREW,
                    Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS, Aircraft.CrewPayload.NUM_GALLEY_CREW,
                    Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER, Aircraft.CrewPayload.MASS_PER_PASSENGER}
            for name, (val, units) in vals.items():
                if name in skip:
                    continue
                if name in ("aircraft:engine:scaled_sls_thrust", "aircraft:nacelle:avg_diameter", "aircraft:nacelle:avg_length"):
                    val = np.array([val])
                if name == "aircraft:engine:wing_locations":
                    val = np.full(max(1, n_wing // 2), val)
                prob.set_val(name, val, units=units or None)
            if "aircraft:fuel:total_capacity" in vals:
                cap, cap_units = vals["aircraft:fuel:total_capacity"]
                prob.set_val(Aircraft.Fuel.TOTAL_CAPACITY, cap, units=cap_units)
            prob.run_model()
    except EstimateError:
        raise
    except Exception as exc:  # OpenMDAO/Aviary failures are reported, not papered over
        raise EstimateError("solver_error", f"Aviary FLOPS mass run failed: {type(exc).__name__}: {exc}") from exc

    def kg(name: str) -> float:
        return float(np.sum(prob.get_val(name, units="kg")))

    def m2(name: str) -> float:
        return float(np.sum(prob.get_val(name, units="m**2")))

    breakdown = {
        "structure": {
            "wing_kg": kg(Aircraft.Wing.MASS),
            "horizontal_tail_kg": kg(Aircraft.HorizontalTail.MASS),
            "vertical_tail_kg": kg(Aircraft.VerticalTail.MASS),
            "fins_kg": kg(Aircraft.Fins.MASS),
            "canard_kg": kg(Aircraft.Canard.MASS),
            "fuselage_kg": kg(Aircraft.Fuselage.MASS),
            "main_landing_gear_kg": kg(Aircraft.LandingGear.MAIN_GEAR_MASS),
            "nose_landing_gear_kg": kg(Aircraft.LandingGear.NOSE_GEAR_MASS),
            "nacelles_kg": kg(Aircraft.Nacelle.MASS),
            "paint_kg": kg(Aircraft.Paint.MASS),
            "total_kg": kg(Aircraft.Design.STRUCTURE_MASS),
        },
        "propulsion": {
            "engines_kg": kg(Aircraft.Propulsion.TOTAL_ENGINE_MASS),
            "thrust_reversers_kg": kg(Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS),
            "misc_propulsion_kg": kg(Aircraft.Propulsion.TOTAL_MISC_MASS),
            "fuel_system_kg": kg(Aircraft.Fuel.FUEL_SYSTEM_MASS),
            "total_kg": kg(Aircraft.Propulsion.MASS),
        },
        "systems_and_equipment": {
            "surface_controls_kg": kg(Aircraft.Wing.SURFACE_CONTROL_MASS),
            "apu_kg": kg(Aircraft.APU.MASS),
            "instruments_kg": kg(Aircraft.Instruments.MASS),
            "hydraulics_kg": kg(Aircraft.Hydraulics.MASS),
            "electrical_kg": kg(Aircraft.Electrical.MASS),
            "avionics_kg": kg(Aircraft.Avionics.MASS),
            "furnishings_kg": kg(Aircraft.Furnishings.MASS),
            "air_conditioning_kg": kg(Aircraft.AirConditioning.MASS),
            "anti_icing_kg": kg(Aircraft.AntiIcing.MASS),
            "total_kg": kg(Aircraft.Design.SYSTEMS_EQUIP_MASS),
        },
        "empty_mass_margin_kg": kg(Aircraft.Design.EMPTY_MASS_MARGIN),
        "empty_mass_kg": kg(Aircraft.Design.EMPTY_MASS),
        "operating_items": {
            "flight_crew_kg": kg(Aircraft.CrewPayload.FLIGHT_CREW_MASS),
            "cabin_crew_kg": kg(Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS),
            "unusable_fuel_kg": kg(Aircraft.Fuel.UNUSABLE_FUEL_MASS),
            "engine_oil_kg": kg(Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS),
            "passenger_service_kg": kg(Aircraft.CrewPayload.PASSENGER_SERVICE_MASS),
            "cargo_containers_kg": kg(Aircraft.CrewPayload.CARGO_CONTAINER_MASS),
        },
        "operating_mass_kg": kg(Aircraft.Design.OPERATING_MASS),
    }
    derived = {
        "aspect_ratio": float(prob.get_val(Aircraft.Wing.ASPECT_RATIO)[0]),
        "fuel_capacity_kg": kg(Aircraft.Fuel.TOTAL_CAPACITY),
        "wing_fuel_capacity_kg": kg(Aircraft.Fuel.WING_FUEL_CAPACITY),
        "touchdown_mass_kg": kg(Aircraft.Design.TOUCHDOWN_MASS),
        "zero_fuel_mass_kg": kg(Aircraft.Design.ZERO_FUEL_MASS),
        "flops_total_wetted_area_m2": m2(Aircraft.Design.TOTAL_WETTED_AREA),
        "flops_fuselage_wetted_area_m2": m2(Aircraft.Fuselage.WETTED_AREA),
        "main_gear_oleo_length_m": float(prob.get_val(Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, units="m")[0]),
    }
    return {"breakdown": breakdown, "derived": derived, "aviary_version": av["aviary"].__version__}


# --------------------------------------------------------------------------- estimate


def estimate_oew(cpacs: Path | str, **cli: Any) -> dict[str, Any]:
    """FLOPS-via-Aviary OEW for a CPACS file. CLI keys match the argparse dests."""
    p = Path(cpacs)
    root = load_cpacs(p)
    label = p.name
    inp, extra = collect_inputs(root, label, cli)
    run = run_aviary(inp, extra)

    caveats = [
        "FLOPS is a regression on conventional aluminium tube-and-wing transports of the 1970s-1990s; "
        "all mass scalers are left at 1.0 (uncalibrated), and FLOPS default omissions apply: no paint "
        "mass, no thrust-reverser mass, no empty-mass margin, no composites credit. Change any of these with --set.",
        "Engine mass is the stated dry mass per engine (FLOPS takes engine weight as an input, it does "
        "not estimate it); nacelle geometry and passenger compartment length are caller-stated because "
        "the CPACS file does not carry them.",
        "All seats are counted as tourist class, as FLOPS does when no class split is given.",
    ]
    ref_area = inp.values["aircraft:wing:area"][0]
    planform = extra["wing_planform_area_m2"]
    if abs(planform - ref_area) / ref_area > 0.05:
        caveats.append(
            f"the wing planform resolved from the sections ({planform:.1f} m^2) differs from the "
            f"reference area ({ref_area:.1f} m^2) by more than 5 percent; FLOPS uses the reference area"
        )
    aero_area_el = root.find(f"{MODEL_PATH}/analysisResults/aero/wettedAreaM2")
    su2_area = _float_text(aero_area_el, "wettedAreaM2", label) if aero_area_el is not None else None
    if su2_area is not None:
        caveats.append(
            f"cross-check: FLOPS's own total wetted area estimate is {run['derived']['flops_total_wetted_area_m2']:.1f} m^2 "
            f"against {su2_area:.1f} m^2 measured on the SU2 mesh at {_xpath('analysisResults/aero/wettedAreaM2')} "
            "(FLOPS includes nacelles, the mesh has none)"
        )
    cli_rows = [r for r in inp.rows if str(r["source"]).startswith("cli:")]
    if cli_rows:
        caveats.append("caller-stated inputs (not in the CPACS file): " + ", ".join(
            f"{r['name']}={r['value']:g} {r['units']}".strip() if isinstance(r["value"], (int, float)) else f"{r['name']}={r['value']}"
            for r in cli_rows))

    return {
        "cpacs": str(p),
        "oew_kg": run["breakdown"]["operating_mass_kg"],
        "oew_lb": run["breakdown"]["operating_mass_kg"] / KG_PER_LB,
        "empty_mass_kg": run["breakdown"]["empty_mass_kg"],
        "breakdown": run["breakdown"],
        "derived": run["derived"],
        "method": f"FLOPS via Aviary {run['aviary_version']}",
        "aviary_components": [
            "aviary.subsystems.geometry.flops_based.prep_geom.PrepGeom",
            "aviary.subsystems.mass.flops_based.mass_premission.MassPremission",
            "aviary.utils.preprocessors.preprocess_crewpayload",
        ],
        "inputs": inp.rows,
        "caveats": caveats,
    }


def write_estimate(cpacs: Path | str, estimate: dict[str, Any], out_path: Path | str | None = None) -> Path:
    """Record ``estimate`` at //vehicles/aircraft/model/analysisResults/massProperties/oewEstimateFlops.

    Only that element is created or replaced; every other node is left as read.
    Writes in place unless ``out_path`` is given. Returns the path written.
    """
    p = Path(cpacs)
    root = load_cpacs(p)
    model = root.find(MODEL_PATH)
    if model is None:
        raise EstimateError("missing_input", f"{p.name}: no //{MODEL_PATH} element to write under")
    parent = model
    for tag in ESTIMATE_PARENT:
        child = parent.find(tag)
        if child is None:
            child = ET.SubElement(parent, tag)
        parent = child
    old = parent.find(ESTIMATE_TAG)
    if old is not None:
        parent.remove(old)

    est_el = ET.SubElement(parent, ESTIMATE_TAG)
    ET.SubElement(est_el, "oewKg").text = repr(float(estimate["oew_kg"]))
    ET.SubElement(est_el, "emptyMassKg").text = repr(float(estimate["empty_mass_kg"]))
    ET.SubElement(est_el, "method").text = estimate["method"]

    def _emit(parent_el: ET.Element, obj: dict[str, Any]) -> None:
        for key, val in obj.items():
            tag = "".join(w.capitalize() if i else w for i, w in enumerate(key.split("_")))
            if isinstance(val, dict):
                _emit(ET.SubElement(parent_el, tag), val)
            else:
                ET.SubElement(parent_el, tag).text = repr(float(val))

    _emit(ET.SubElement(est_el, "breakdown"), estimate["breakdown"])
    _emit(ET.SubElement(est_el, "derived"), estimate["derived"])
    inputs_el = ET.SubElement(est_el, "inputs")
    for row in estimate["inputs"]:
        in_el = ET.SubElement(inputs_el, "input")
        ET.SubElement(in_el, "name").text = str(row["name"])
        ET.SubElement(in_el, "value").text = repr(row["value"]) if isinstance(row["value"], (int, float)) else str(row["value"])
        ET.SubElement(in_el, "units").text = str(row["units"])
        ET.SubElement(in_el, "source").text = str(row["source"])
        if row.get("note"):
            ET.SubElement(in_el, "note").text = str(row["note"])
    cav_el = ET.SubElement(est_el, "caveats")
    for caveat in estimate["caveats"]:
        ET.SubElement(cav_el, "caveat").text = caveat

    out = Path(out_path) if out_path is not None else p
    out.write_text(ET.tostring(root, encoding="unicode", xml_declaration=True), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cpacs", type=Path, help="CPACS file with wing, tail and fuselage geometry")
    g = p.add_argument_group("aircraft properties the file may lack (no defaults)")
    g.add_argument("--design-gross-weight-kg", type=float, default=None, help="design gross mass (FLOPS DGW)")
    g.add_argument("--passengers", type=int, default=None, help="design passenger count")
    g.add_argument("--design-range-km", type=float, default=None, help="design range")
    g.add_argument("--cruise-mach", type=float, default=None, help="cruise Mach, if the file records no aero run")
    g.add_argument("--max-mach", type=float, default=None, help="maximum operating Mach (FLOPS VMMO)")
    g.add_argument("--num-engines", type=int, default=None, help="number of engines")
    g.add_argument("--engine-mount", choices=("wing", "fuselage"), default=None,
                   help="where the engines hang (wing when omitted, the FLOPS transport default)")
    g.add_argument("--engine-sls-thrust-kn", type=float, default=None, help="sea-level static thrust per engine")
    g.add_argument("--engine-mass-kg", type=float, default=None, help="dry mass per engine")
    g.add_argument("--nacelle-diameter-m", type=float, default=None, help="nacelle average diameter")
    g.add_argument("--nacelle-length-m", type=float, default=None, help="nacelle average length")
    g.add_argument("--engine-spanwise-fraction", type=float, default=None,
                   help="wing-engine position as a fraction of semispan (main-gear length)")
    g.add_argument("--passenger-compartment-length-m", type=float, default=None, help="cabin length (furnishings)")
    g.add_argument("--fuel-capacity-kg", type=float, default=None,
                   help="total fuel capacity; when omitted Aviary computes the wing capacity with the FLOPS "
                        "formula and no fuselage or auxiliary tanks")
    p.add_argument("--set", action="append", default=None, metavar="NAME=VALUE[:UNITS]",
                   help="override any Aviary input, e.g. aircraft:wing:mass_scaler=0.9 or "
                        "aircraft:paint:mass_per_unit_area=0.037:lbm/ft**2")
    p.add_argument("--write", action="store_true",
                   help="record the estimate at analysisResults/massProperties/oewEstimateFlops (in place unless --out)")
    p.add_argument("--out", type=Path, default=None, help="with --write: write the updated CPACS here instead of in place")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cli = {k: v for k, v in vars(a).items() if k not in ("cpacs", "write", "out", "json")}
    try:
        est = estimate_oew(a.cpacs, **cli)
        if a.write or a.out is not None:
            est["written_to"] = str(write_estimate(a.cpacs, est, a.out))
    except EstimateError as exc:
        print(json.dumps(exc.to_dict()))
        return 2 if exc.error_type in ("missing_input", "conflicting_input", "missing_dependency") else 1

    if a.json:
        print(json.dumps(est, indent=2, default=float))
        return 0
    b = est["breakdown"]
    print(est["cpacs"])
    print(f"  OEW          {est['oew_kg']:,.1f} kg ({est['oew_lb']:,.1f} lb)   empty mass {est['empty_mass_kg']:,.1f} kg")
    print(f"  method       {est['method']}")
    for group in ("structure", "propulsion", "systems_and_equipment"):
        print(f"  {group:22s} {b[group]['total_kg']:>10,.1f} kg")
        for key, val in b[group].items():
            if key != "total_kg":
                print(f"      {key[:-3]:26s} {val:>10,.1f}")
    print(f"  {'empty mass margin':22s} {b['empty_mass_margin_kg']:>10,.1f} kg")
    print(f"  {'operating items':22s} {sum(b['operating_items'].values()):>10,.1f} kg")
    for key, val in b["operating_items"].items():
        print(f"      {key[:-3]:26s} {val:>10,.1f}")
    print("  inputs")
    for row in est["inputs"]:
        val = f"{row['value']:.6g}" if isinstance(row["value"], (int, float)) else str(row["value"])
        print(f"      {row['name']:52s} {val:>12s} {row['units']:10s} {row['source']}")
    for caveat in est["caveats"]:
        print(f"  caveat       {caveat}")
    if "written_to" in est:
        print(f"  written to   {est['written_to']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
