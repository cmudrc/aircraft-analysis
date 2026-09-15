#!/usr/bin/env python3
"""Operating empty weight from wetted area: OEW = K * A_wet.

Ron Engelbeck (Boeing, 2026-09) gave the simplest empty-weight method there
is: operating empty weight is proportional to total wetted area, with K about
12 lb/ft^2 for conventional aluminium transports, and K should be calibrated
on a similar aircraft whose OEW is known. This script does that and nothing
more.

The wetted area is read from the CPACS file, never computed here and never
guessed. It is taken from the first of these that exists:

    //vehicles/aircraft/model/analysisResults/aero/wettedAreaM2
        written by the SU2 server: the sum of the wall-marker faces of the
        mesh the run used (the sibling wettedAreaSource says so)
    //vehicles/aircraft/model/analysisResults/tigl/fusedBody/surfaceAreaM2
        written by the TiGL server: the surface area of the fused closed body

If neither exists the result is a structured error, not a formula.

K is a method constant and has no built-in default. The caller states it with
``--k-lb-ft2`` or calibrates it with ``--calibrate-from REF.xml``, which reads
the reference file's stated operating empty mass at
//vehicles/aircraft/model/analyses/massBreakdown/mOEM/massDescription/mass
and its wetted area (same lookup) and uses K = OEM / A_wet.

Usage:
    estimate_oew_from_wetted_area.py CPACS.xml --k-lb-ft2 12
    estimate_oew_from_wetted_area.py CPACS.xml --calibrate-from REF.xml
    estimate_oew_from_wetted_area.py CPACS.xml --k-lb-ft2 12 --write [--out NEW.xml]

``--write`` records the estimate at
//vehicles/aircraft/model/analysisResults/massProperties/oewEstimate (replacing
an earlier oewEstimate there) and touches no other node.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

LB_PER_KG = 1.0 / 0.45359237
FT2_PER_M2 = 1.0 / (0.3048 * 0.3048)

METHOD = "OEW = K * A_wet (rule of thumb, Ron Engelbeck, Boeing, 2026-09)"
NO_NACELLE_CAVEAT = "wetted area excludes engine nacelles and pylons if the CPACS file has none"

MODEL_PATH = "vehicles/aircraft/model"
AERO_NODE = "analysisResults/aero/wettedAreaM2"
AERO_SOURCE_NODE = "analysisResults/aero/wettedAreaSource"
TIGL_NODE = "analysisResults/tigl/fusedBody/surfaceAreaM2"
TIGL_NACELLES_NODE = "analysisResults/tigl/fusedBody/nacellesPylonsIncluded"
TIGL_COMPONENTS = "analysisResults/tigl/components/component"
OEM_NODE = "analyses/massBreakdown/mOEM/massDescription/mass"
ESTIMATE_PARENT = ("analysisResults", "massProperties")
ESTIMATE_TAG = "oewEstimate"

_ENGINE_WORDS = ("engine", "nacelle", "pylon")

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


def load_cpacs(path: Path | str) -> ET.Element:
    p = Path(path)
    if not p.is_file():
        raise EstimateError("missing_input", f"CPACS file not found: {p}")
    try:
        return ET.parse(p).getroot()
    except ET.ParseError as exc:
        raise EstimateError("bad_input", f"{p} is not well-formed XML: {exc}") from exc


def _positive_float(el: ET.Element | None, what: str, label: str) -> float | None:
    """Float text of ``el``; None when the node is absent; error when it is unusable."""
    if el is None:
        return None
    text = (el.text or "").strip()
    try:
        val = float(text)
    except ValueError:
        raise EstimateError("bad_input", f"{label}: {what} is '{text}', not a number") from None
    if not val > 0:  # also rejects NaN
        raise EstimateError("bad_input", f"{label}: {what} is {val}; it must be positive")
    return val


def find_wetted_area(root: ET.Element, label: str = "CPACS") -> dict[str, Any]:
    """Wetted area [m^2] with provenance: aero/wettedAreaM2, else tigl/fusedBody/surfaceAreaM2.

    Raises EstimateError("missing_input") when the file states neither. There is
    no formula fallback: a wetted area this tool did not read from the file would
    be a number nobody measured.
    """
    model = root.find(MODEL_PATH)
    if model is None:
        raise EstimateError("missing_input", f"{label}: no //{MODEL_PATH} element")

    val = _positive_float(model.find(AERO_NODE), "wettedAreaM2", label)
    if val is not None:
        src_el = model.find(AERO_SOURCE_NODE)
        source = (src_el.text or "").strip() if src_el is not None else ""
        return {
            "wetted_area_m2": val,
            "wetted_area_source": source or f"SU2 server value at {_xpath(AERO_NODE)}, no wettedAreaSource stated",
            "wetted_area_node": _xpath(AERO_NODE),
        }

    val = _positive_float(model.find(TIGL_NODE), "surfaceAreaM2", label)
    if val is not None:
        inc_el = model.find(TIGL_NACELLES_NODE)
        note = f"; nacellesPylonsIncluded={inc_el.text.strip()}" if inc_el is not None and inc_el.text else ""
        return {
            "wetted_area_m2": val,
            "wetted_area_source": f"TiGL fused-body surface area at {_xpath(TIGL_NODE)}{note}",
            "wetted_area_node": _xpath(TIGL_NODE),
        }

    raise EstimateError(
        "missing_input",
        f"{label} states no wetted area at {_xpath(AERO_NODE)} or {_xpath(TIGL_NODE)}. "
        "Run the SU2 server (a CFD run records the wall-face area) or the TiGL server "
        "(a fused-body export records the surface area) first; this tool does not "
        "estimate wetted area.",
    )


def engine_components_present(root: ET.Element) -> bool:
    """True when the TiGL component list names an engine, nacelle or pylon."""
    for comp in root.iterfind(f"{MODEL_PATH}/{TIGL_COMPONENTS}"):
        for tag in ("type", "uid", "name"):
            el = comp.find(tag)
            text = (el.text or "").lower() if el is not None else ""
            if any(word in text for word in _ENGINE_WORDS):
                return True
    return False


def calibrate_k_from_reference(ref_path: Path | str) -> dict[str, Any]:
    """K = OEM / A_wet [lb/ft^2] from a reference CPACS with a stated OEM and wetted area."""
    p = Path(ref_path)
    root = load_cpacs(p)
    label = f"reference {p.name}"
    model = root.find(MODEL_PATH)
    oem_kg = _positive_float(model.find(OEM_NODE) if model is not None else None, "mOEM mass", label)
    if oem_kg is None:
        raise EstimateError(
            "missing_input",
            f"{label} states no operating empty mass at {_xpath(OEM_NODE)}; K cannot be calibrated on it.",
        )
    area = find_wetted_area(root, label)
    k = (oem_kg * LB_PER_KG) / (area["wetted_area_m2"] * FT2_PER_M2)
    return {
        "k_lb_per_ft2": k,
        "k_source": (
            f"calibrated on {p}: OEM {oem_kg:g} kg / wetted area {area['wetted_area_m2']:g} m^2"
        ),
        "calibration": {
            "reference_cpacs": str(p),
            "reference_oem_kg": oem_kg,
            "reference_oem_node": _xpath(OEM_NODE),
            "reference_wetted_area_m2": area["wetted_area_m2"],
            "reference_wetted_area_source": area["wetted_area_source"],
            "reference_wetted_area_node": area["wetted_area_node"],
        },
    }


def estimate_from_root(
    root: ET.Element,
    *,
    k_lb_ft2: float,
    k_source: str,
    label: str = "CPACS",
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """OEW = K * A_wet on an already-parsed CPACS tree. Pure arithmetic once the inputs exist."""
    if not k_lb_ft2 > 0:
        raise EstimateError("bad_input", f"K must be positive, got {k_lb_ft2}")
    area = find_wetted_area(root, label)
    a_ft2 = area["wetted_area_m2"] * FT2_PER_M2
    oew_lb = k_lb_ft2 * a_ft2

    caveats: list[str] = []
    if not engine_components_present(root):
        caveats.append(NO_NACELLE_CAVEAT)
    if calibration is None:
        caveats.append(
            "K is a class constant (about 12 lb/ft^2 for conventional aluminium transports); "
            "calibrate it on a similar aircraft with a known OEW before relying on the number"
        )
    else:
        caveats.append(
            "K was calibrated on one reference aircraft; the estimate is only as good as the "
            "resemblance in class and construction"
        )

    out: dict[str, Any] = {
        "oew_kg": oew_lb / LB_PER_KG,
        "oew_lb": oew_lb,
        "k_lb_per_ft2": k_lb_ft2,
        "k_source": k_source,
        **area,
        "wetted_area_ft2": a_ft2,
        "method": METHOD,
        "caveats": caveats,
    }
    if calibration is not None:
        out["calibration"] = calibration
    return out


def estimate_oew(
    cpacs: Path | str,
    *,
    k_lb_ft2: float | None = None,
    calibrate_from: Path | str | None = None,
) -> dict[str, Any]:
    """Estimate OEW for a CPACS file. Exactly one of ``k_lb_ft2`` / ``calibrate_from`` is required."""
    if k_lb_ft2 is None and calibrate_from is None:
        raise EstimateError(
            "missing_input",
            "K is a method constant with no built-in default: state it with --k-lb-ft2 "
            "(Ron Engelbeck: about 12 lb/ft^2 for conventional aluminium transports) or "
            "calibrate it with --calibrate-from REF.xml.",
        )
    if k_lb_ft2 is not None and calibrate_from is not None:
        raise EstimateError("conflicting_input", "Give --k-lb-ft2 or --calibrate-from, not both.")

    p = Path(cpacs)
    root = load_cpacs(p)
    if calibrate_from is not None:
        cal = calibrate_k_from_reference(calibrate_from)
        k, k_source, calibration = cal["k_lb_per_ft2"], cal["k_source"], cal["calibration"]
    else:
        k, k_source, calibration = float(k_lb_ft2), "stated by the caller (--k-lb-ft2)", None

    est = estimate_from_root(root, k_lb_ft2=k, k_source=k_source, label=p.name, calibration=calibration)
    est["cpacs"] = str(p)
    return est


def write_estimate(cpacs: Path | str, estimate: dict[str, Any], out_path: Path | str | None = None) -> Path:
    """Record ``estimate`` at //vehicles/aircraft/model/analysisResults/massProperties/oewEstimate.

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
    for tag, val in (
        ("oewKg", repr(float(estimate["oew_kg"]))),
        ("method", estimate["method"]),
        ("kLbPerFt2", repr(float(estimate["k_lb_per_ft2"]))),
        ("kSource", estimate["k_source"]),
        ("wettedAreaM2", repr(float(estimate["wetted_area_m2"]))),
        ("wettedAreaSource", estimate["wetted_area_source"]),
    ):
        ET.SubElement(est_el, tag).text = val
    cav_el = ET.SubElement(est_el, "caveats")
    for caveat in estimate["caveats"]:
        ET.SubElement(cav_el, "caveat").text = caveat

    out = Path(out_path) if out_path is not None else p
    out.write_text(ET.tostring(root, encoding="unicode", xml_declaration=True), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cpacs", type=Path, help="CPACS file carrying a wetted area from the SU2 or TiGL server")
    p.add_argument("--k-lb-ft2", type=float, default=None,
                   help="K [lb/ft^2], stated by you (Ron Engelbeck: about 12 for conventional "
                        "aluminium transports). No default.")
    p.add_argument("--calibrate-from", type=Path, default=None,
                   help="reference CPACS with a stated mOEM mass and wetted area; K = OEM / A_wet")
    p.add_argument("--write", action="store_true",
                   help="record the estimate at analysisResults/massProperties/oewEstimate "
                        "(in place unless --out)")
    p.add_argument("--out", type=Path, default=None,
                   help="with --write: write the updated CPACS here instead of in place")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    a = p.parse_args(argv)

    try:
        est = estimate_oew(a.cpacs, k_lb_ft2=a.k_lb_ft2, calibrate_from=a.calibrate_from)
        if a.write or a.out is not None:
            est["written_to"] = str(write_estimate(a.cpacs, est, a.out))
    except EstimateError as exc:
        print(json.dumps(exc.to_dict()))
        return 2 if exc.error_type in ("missing_input", "conflicting_input") else 1

    if a.json:
        print(json.dumps(est, indent=2))
        return 0
    print(est["cpacs"])
    print(f"  wetted area  {est['wetted_area_m2']:,.3f} m^2 ({est['wetted_area_ft2']:,.1f} ft^2)")
    print(f"               source: {est['wetted_area_source']}")
    print(f"  K            {est['k_lb_per_ft2']:.4g} lb/ft^2  source: {est['k_source']}")
    print(f"  OEW          {est['oew_kg']:,.1f} kg ({est['oew_lb']:,.1f} lb)")
    print(f"  method       {est['method']}")
    for caveat in est["caveats"]:
        print(f"  caveat       {caveat}")
    if "written_to" in est:
        print(f"  written to   {est['written_to']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
