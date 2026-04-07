"""Versioned CPACS XML manager.

Provides a thread-safe, version-tracked CPACS document that multiple MCPs
can read from and write back to via XPath-based operations.
"""

from __future__ import annotations

import copy
import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class CPACSVersion:
    """Snapshot of the CPACS document at a point in time."""

    version_id: int
    xml_string: str
    sha256: str
    timestamp: str
    author: str
    description: str


class CPACSManager:
    """Thread-safe, version-tracked CPACS XML manager.

    Loads a CPACS XML file, tracks every mutation as a numbered version,
    and provides XPath-based read/write helpers for the MCP adapters.
    """

    def __init__(self) -> None:
        self._versions: list[CPACSVersion] = []
        self._tree: ET.ElementTree | None = None
        self._root: ET.Element | None = None
        self._lock = threading.Lock()
        self._source_path: str | None = None

    @property
    def current_version(self) -> int:
        return len(self._versions) - 1 if self._versions else -1

    @property
    def root(self) -> ET.Element:
        if self._root is None:
            raise RuntimeError("No CPACS document loaded")
        return self._root

    def load_file(self, path: str | Path) -> CPACSVersion:
        """Load a CPACS XML file from disk as version 0."""
        path = Path(path)
        xml_string = path.read_text(encoding="utf-8")
        self._source_path = str(path)
        return self._load_xml(xml_string, author="file_load", description=f"Loaded from {path.name}")

    def load_string(self, xml_string: str, source_name: str = "string") -> CPACSVersion:
        """Load CPACS XML from a string as version 0."""
        return self._load_xml(xml_string, author="string_load", description=f"Loaded from {source_name}")

    def _load_xml(self, xml_string: str, author: str, description: str) -> CPACSVersion:
        with self._lock:
            self._root = ET.fromstring(xml_string)
            self._tree = ET.ElementTree(self._root)
            self._versions.clear()
            version = self._snapshot(author, description)
            return version

    def _snapshot(self, author: str, description: str) -> CPACSVersion:
        xml_bytes = ET.tostring(self._root, encoding="unicode")
        sha = hashlib.sha256(xml_bytes.encode("utf-8")).hexdigest()[:16]
        ver = CPACSVersion(
            version_id=len(self._versions),
            xml_string=xml_bytes,
            sha256=sha,
            timestamp=datetime.now(timezone.utc).isoformat(),
            author=author,
            description=description,
        )
        self._versions.append(ver)
        return ver

    def commit(self, author: str, description: str) -> CPACSVersion:
        """Commit the current state as a new version."""
        with self._lock:
            return self._snapshot(author, description)

    def get_version(self, version_id: int) -> CPACSVersion:
        """Retrieve a specific version snapshot."""
        with self._lock:
            if version_id < 0 or version_id >= len(self._versions):
                raise IndexError(f"Version {version_id} not found (have 0..{len(self._versions) - 1})")
            return self._versions[version_id]

    def get_current_xml(self) -> str:
        """Return the current CPACS XML as a string."""
        with self._lock:
            return ET.tostring(self._root, encoding="unicode")

    def version_history(self) -> list[dict[str, str | int]]:
        """Return metadata for all versions."""
        with self._lock:
            return [
                {
                    "version_id": v.version_id,
                    "sha256": v.sha256,
                    "timestamp": v.timestamp,
                    "author": v.author,
                    "description": v.description,
                }
                for v in self._versions
            ]

    def restore(self, version_id: int) -> CPACSVersion:
        """Restore the CPACS state to a previous version (creates a new version)."""
        with self._lock:
            target = self._versions[version_id]
            self._root = ET.fromstring(target.xml_string)
            self._tree = ET.ElementTree(self._root)
            return self._snapshot("restore", f"Restored to version {version_id}")

    # ── XPath Helpers ────────────────────────────────────────────────

    def read_text(self, xpath: str) -> str | None:
        """Read text content at an XPath, or None if missing."""
        with self._lock:
            el = self._root.find(xpath)
            return el.text if el is not None else None

    def read_attrib(self, xpath: str, attrib: str) -> str | None:
        """Read an attribute from the element at an XPath."""
        with self._lock:
            el = self._root.find(xpath)
            return el.get(attrib) if el is not None else None

    def read_element(self, xpath: str) -> ET.Element | None:
        """Return a deep copy of the element at an XPath."""
        with self._lock:
            el = self._root.find(xpath)
            return copy.deepcopy(el) if el is not None else None

    def read_elements(self, xpath: str) -> list[ET.Element]:
        """Return deep copies of all elements matching an XPath."""
        with self._lock:
            return [copy.deepcopy(el) for el in self._root.findall(xpath)]

    def write_text(self, xpath: str, value: str) -> None:
        """Set text content at an XPath, creating intermediate elements as needed."""
        with self._lock:
            el = self._ensure_path(xpath)
            el.text = value

    def write_element(self, xpath: str, element: ET.Element) -> None:
        """Replace or insert an element subtree at an XPath."""
        with self._lock:
            parts = xpath.strip("/").split("/")
            parent_path = "/".join(parts[:-1])
            tag = parts[-1]

            parent = self._ensure_path(parent_path) if parent_path else self._root

            existing = parent.find(tag)
            if existing is not None:
                parent.remove(existing)
            parent.append(element)

    def remove_element(self, xpath: str) -> bool:
        """Remove the element at an XPath. Returns True if removed."""
        with self._lock:
            parts = xpath.strip("/").split("/")
            if len(parts) < 2:
                return False
            parent_path = "/".join(parts[:-1])
            tag = parts[-1]
            parent = self._root.find(parent_path)
            if parent is None:
                return False
            child = parent.find(tag)
            if child is None:
                return False
            parent.remove(child)
            return True

    def _ensure_path(self, xpath: str) -> ET.Element:
        """Walk/create elements along an XPath, returning the leaf."""
        parts = xpath.strip("/").split("/")
        current = self._root
        for part in parts:
            if not part:
                continue
            child = current.find(part)
            if child is None:
                child = ET.SubElement(current, part)
            current = child
        return current

    def save(self, path: str | Path) -> None:
        """Write the current CPACS XML to disk."""
        with self._lock:
            tree = ET.ElementTree(self._root)
            ET.indent(tree, space="  ")
            tree.write(str(path), encoding="unicode", xml_declaration=True)

    def extract_reference_data(self) -> dict[str, float | str | None]:
        """Extract common aircraft reference data from the CPACS."""
        model_uid = self.read_attrib(".//vehicles/aircraft/model", "uID")
        ref_area = self.read_text(".//vehicles/aircraft/model/reference/area")
        ref_length = self.read_text(".//vehicles/aircraft/model/reference/length")
        name = self.read_text(".//vehicles/aircraft/model/name")

        return {
            "model_uid": model_uid,
            "name": name,
            "ref_area_m2": float(ref_area) if ref_area else None,
            "ref_length_m": float(ref_length) if ref_length else None,
        }
