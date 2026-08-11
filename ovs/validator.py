"""CPACS output validators for the OVS CI checks.

Each validator checks that a specific MCP populated its expected XPaths
with values in physically plausible ranges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.etree import ElementTree as ET


@dataclass
class ValidationResult:
    """Outcome of a single validation check."""

    check: str
    passed: bool
    message: str
    value: str | float | None = None


@dataclass
class ValidationReport:
    """Aggregated results for one CPACS file."""

    source: str
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def summary(self) -> str:
        total = len(self.results)
        failed = self.failed_count
        status = "PASS" if self.passed else "FAIL"
        lines = [f"OVS Report [{status}]: {total - failed}/{total} checks passed for {self.source}"]
        for r in self.results:
            mark = "OK" if r.passed else "FAIL"
            lines.append(f"  [{mark}] {r.check}: {r.message}")
        return "\n".join(lines)


def validate_cpacs_structure(xml_string: str, source: str = "unknown") -> ValidationReport:
    """Validate basic CPACS structure (well-formed XML, required elements)."""
    report = ValidationReport(source=source)

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        report.results.append(ValidationResult("xml_parse", False, f"Invalid XML: {e}"))
        return report
    report.results.append(ValidationResult("xml_parse", True, "Valid XML"))

    if root.tag != "cpacs":
        report.results.append(ValidationResult("root_tag", False, f"Root is '{root.tag}', expected 'cpacs'"))
    else:
        report.results.append(ValidationResult("root_tag", True, "Root tag is 'cpacs'"))

    vehicles = root.find(".//vehicles")
    report.results.append(ValidationResult(
        "vehicles_present", vehicles is not None,
        "vehicles element present" if vehicles is not None else "Missing vehicles element"
    ))

    aircraft = root.find(".//vehicles/aircraft")
    report.results.append(ValidationResult(
        "aircraft_present", aircraft is not None,
        "aircraft element present" if aircraft is not None else "Missing aircraft element"
    ))

    return report


def validate_tigl_output(xml_string: str, source: str = "unknown") -> ValidationReport:
    """Validate TiGL adapter populated its XPaths."""
    report = ValidationReport(source=source)
    root = ET.fromstring(xml_string)

    tigl = root.find(".//vehicles/aircraft/model/analysisResults/tigl")
    if tigl is None:
        report.results.append(ValidationResult("tigl_results", False, "No tigl results found"))
        return report
    report.results.append(ValidationResult("tigl_results", True, "tigl results present"))

    for tag, label in [("wingCount", "wing count"), ("fuselageCount", "fuselage count")]:
        el = tigl.find(tag)
        if el is not None and el.text:
            val = int(el.text)
            ok = val >= 0
            report.results.append(ValidationResult(f"tigl_{tag}", ok, f"{label}={val}", val))
        else:
            report.results.append(ValidationResult(f"tigl_{tag}", False, f"Missing {label}"))

    comps = tigl.find("components")
    has_comps = comps is not None and len(comps) > 0
    report.results.append(ValidationResult(
        "tigl_components", has_comps,
        f"{len(comps)} components" if has_comps else "No components listed"
    ))

    return report


def validate_su2_output(xml_string: str, source: str = "unknown") -> ValidationReport:
    """Validate SU2 adapter populated its XPaths with plausible values."""
    report = ValidationReport(source=source)
    root = ET.fromstring(xml_string)

    aero = root.find(".//vehicles/aircraft/model/analysisResults/aero")
    if aero is None:
        report.results.append(ValidationResult("su2_results", False, "No aero results found"))
        return report
    report.results.append(ValidationResult("su2_results", True, "aero results present"))

    solver_el = aero.find("solver")
    report.results.append(ValidationResult(
        "su2_solver", solver_el is not None and solver_el.text == "su2_cfd",
        f"solver={solver_el.text if solver_el is not None else 'missing'}"
    ))

    converged_el = aero.find("converged")
    is_converged = converged_el is not None and converged_el.text == "true"
    report.results.append(ValidationResult(
        "su2_converged", is_converged,
        "SU2 converged" if is_converged else "SU2 did not converge (check error element)"
    ))

    error_el = aero.find("error")
    if error_el is not None:
        err_msg = ""
        msg_el = error_el.find("message")
        if msg_el is not None and msg_el.text:
            err_msg = msg_el.text
        report.results.append(ValidationResult(
            "su2_no_error", False, f"SU2 error: {err_msg}"
        ))
    else:
        report.results.append(ValidationResult("su2_no_error", True, "No SU2 errors"))

    coeffs = aero.find("coefficients")
    if coeffs is None:
        report.results.append(ValidationResult("su2_coefficients", False, "No coefficients element"))
        return report

    checks = {
        "CL": (-2.0, 3.0),
        "CD": (0.0, 2.0),
        "L_over_D": (-100.0, 100.0),
    }
    for name, (lo, hi) in checks.items():
        el = coeffs.find(name)
        if el is not None and el.text:
            val = float(el.text)
            ok = lo <= val <= hi
            report.results.append(ValidationResult(
                f"su2_{name}", ok,
                f"{name}={val:.6f} (range [{lo}, {hi}])", val
            ))
        else:
            report.results.append(ValidationResult(f"su2_{name}", not is_converged,
                f"Missing {name}" + (" (expected since not converged)" if not is_converged else "")))

    return report


def validate_pycycle_output(xml_string: str, source: str = "unknown") -> ValidationReport:
    """Validate pyCycle adapter populated its XPaths."""
    report = ValidationReport(source=source)
    root = ET.fromstring(xml_string)

    mcp_res = root.find(".//vehicles/engines/engine/analysis/mcpResults")
    if mcp_res is None:
        report.results.append(ValidationResult("pycycle_results", False, "No mcpResults found"))
        return report
    report.results.append(ValidationResult("pycycle_results", True, "mcpResults present"))

    solver_el = mcp_res.find("solver")
    report.results.append(ValidationResult(
        "pycycle_solver", solver_el is not None and solver_el.text == "pycycle_openmdao",
        f"solver={solver_el.text if solver_el is not None else 'missing'}"
    ))

    error_el = mcp_res.find("error")
    has_error = error_el is not None
    if has_error:
        err_msg = ""
        msg_el = error_el.find("message")
        if msg_el is not None and msg_el.text:
            err_msg = msg_el.text
        report.results.append(ValidationResult(
            "pycycle_no_error", False, f"pyCycle error: {err_msg}"
        ))
    else:
        report.results.append(ValidationResult("pycycle_no_error", True, "No pyCycle errors"))

    checks = {
        "TSFC_lb_lbf_hr": (0.0, 5.0),
        "Fn_N": (0.0, 1e7),
        "OPR": (1.0, 100.0),
        "BPR": (0.0, 50.0),
        "fuelFlow_kg_s": (0.0, 100.0),
    }
    for name, (lo, hi) in checks.items():
        el = mcp_res.find(name)
        if el is not None and el.text:
            val = float(el.text)
            ok = lo <= val <= hi
            report.results.append(ValidationResult(
                f"pycycle_{name}", ok,
                f"{name}={val} (range [{lo}, {hi}])", val
            ))
        else:
            report.results.append(ValidationResult(f"pycycle_{name}", not has_error,
                f"Missing {name}" + (" (expected due to error)" if has_error else "")))

    return report


def validate_mission_output(xml_string: str, source: str = "unknown") -> ValidationReport:
    """Validate Mission adapter populated its XPaths."""
    report = ValidationReport(source=source)
    root = ET.fromstring(xml_string)

    mission = root.find(".//vehicles/aircraft/model/analysisResults/mission")
    if mission is None:
        report.results.append(ValidationResult("mission_results", False, "No mission results found"))
        return report
    report.results.append(ValidationResult("mission_results", True, "mission results present"))

    success_el = mission.find("success")
    is_success = success_el is not None and success_el.text == "true"
    report.results.append(ValidationResult(
        "mission_success", is_success,
        "Mission succeeded" if is_success else "Mission did not succeed"
    ))

    checks = {
        "totalFuelBurnedKg": (0.0, 500000.0),
        "totalDistanceNm": (0.0, 20000.0),
        "totalTimeHr": (0.0, 50.0),
        "fuelFraction": (0.0, 1.0),
    }
    for name, (lo, hi) in checks.items():
        el = mission.find(name)
        if el is not None and el.text:
            val = float(el.text)
            ok = lo <= val <= hi
            report.results.append(ValidationResult(
                f"mission_{name}", ok,
                f"{name}={val:.4f} (range [{lo}, {hi}])", val
            ))
        else:
            report.results.append(ValidationResult(f"mission_{name}", False, f"Missing {name}"))

    segs = mission.find("segments")
    has_segs = segs is not None and len(segs) > 0
    report.results.append(ValidationResult(
        "mission_segments", has_segs,
        f"{len(segs)} segments" if has_segs else "No segments in results"
    ))

    return report


def validate_full_pipeline(xml_string: str, source: str = "unknown") -> ValidationReport:
    """Run all validators on a final pipeline output CPACS."""
    combined = ValidationReport(source=source)
    for validator in [
        validate_cpacs_structure,
        validate_tigl_output,
        validate_su2_output,
        validate_pycycle_output,
        validate_mission_output,
    ]:
        sub = validator(xml_string, source)
        combined.results.extend(sub.results)
    return combined
