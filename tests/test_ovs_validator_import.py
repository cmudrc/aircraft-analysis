"""Cross-platform smoke tests for the OVS validator library.

These tests do NOT depend on any solver, only on Python + xml stdlib,
so they are safe to run in the cross-platform CI matrix.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest


def test_validator_module_imports() -> None:
    from ovs import validator

    assert hasattr(validator, "__file__")


def test_minimal_cpacs_parses() -> None:
    sample = """<?xml version='1.0'?>
    <cpacs>
      <vehicles>
        <aircraft>
          <model uID='m1'>
            <name>Test</name>
          </model>
        </aircraft>
      </vehicles>
    </cpacs>"""
    root = ET.fromstring(sample)
    assert root.tag == "cpacs"
    assert root.find(".//model").get("uID") == "m1"


@pytest.mark.parametrize("value,low,high,ok", [
    (0.5, 0.0, 1.0, True),
    (1.1, 0.0, 1.0, False),
    (-0.1, 0.0, 1.0, False),
    (0.0, 0.0, 1.0, True),
])
def test_plausibility_range_logic(value: float, low: float, high: float, ok: bool) -> None:
    assert (low <= value <= high) == ok
