# ======================================================================
# utils/detector_mapping.py
# ----------------------------------------------------------------------
# Parse SUMO config (.sumocfg) to locate additional-files, then parse
# detector .add.xml files and produce:
#
#   det_to_lane: dict[detector_id] -> lane_id
#
# Supports:
#   - laneAreaDetector (E2)
#   - inductionLoop (E1)  (optional)
#
# Also provides:
#   build_detector_lane_mapping()  # alias used by older code
# ======================================================================

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List


def _expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _is_xml_file(p: str) -> bool:
    p = p.lower()
    return p.endswith(".xml") or p.endswith(".xml.gz")


def _parse_sumocfg_additional_files(sumocfg_path: str) -> List[str]:
    """
    Reads a .sumocfg and returns a list of resolved additional-files paths.

    SUMO .sumocfg contains:
      <input>
        <net-file value="..."/>
        <route-files value="..."/>
        <additional-files value="a.add.xml,b.add.xml"/>
      </input>
    """
    sumocfg_path = _expand(sumocfg_path)
    if not os.path.isfile(sumocfg_path):
        raise FileNotFoundError(f"[ERROR] .sumocfg not found: {sumocfg_path}")

    base_dir = os.path.dirname(sumocfg_path)

    tree = ET.parse(sumocfg_path)
    root = tree.getroot()

    additional_value = None
    for inp in root.findall("input"):
        add = inp.find("additional-files")
        if add is not None and "value" in add.attrib:
            additional_value = add.attrib.get("value")
            break

    if not additional_value:
        return []

    raw_parts: List[str] = []
    for chunk in additional_value.replace(";", ",").split(","):
        c = chunk.strip()
        if c:
            raw_parts.append(c)

    resolved: List[str] = []
    for p in raw_parts:
        if not os.path.isabs(p):
            p = os.path.join(base_dir, p)
        resolved.append(_expand(p))

    return resolved


def _parse_detectors_from_add_xml(add_xml_path: str) -> Dict[str, str]:
    """
    Parse a SUMO additional file to extract detectors and their lanes.

    Supported:
      <laneAreaDetector id="det1" lane="edge_0" .../>
      <inductionLoop id="e1_0" lane="edge_0" .../>
    """
    add_xml_path = _expand(add_xml_path)
    if not os.path.isfile(add_xml_path):
        return {}

    # ElementTree does not parse .xml.gz without gzip handling.
    # Your detectors.add.xml is plain .xml, so we skip gz for now.
    if add_xml_path.lower().endswith(".gz"):
        return {}

    tree = ET.parse(add_xml_path)
    root = tree.getroot()

    det_to_lane: Dict[str, str] = {}

    for det in root.findall(".//laneAreaDetector"):
        det_id = det.attrib.get("id")
        lane = det.attrib.get("lane")
        if det_id and lane:
            det_to_lane[det_id] = lane

    for det in root.findall(".//inductionLoop"):
        det_id = det.attrib.get("id")
        lane = det.attrib.get("lane")
        if det_id and lane:
            det_to_lane[det_id] = lane

    return det_to_lane


def load_detectors_from_sumocfg(sumocfg_path: str) -> Dict[str, str]:
    """
    Main entrypoint: returns det_to_lane mapping.
    """
    add_files = _parse_sumocfg_additional_files(sumocfg_path)

    det_to_lane: Dict[str, str] = {}
    for f in add_files:
        if not _is_xml_file(f):
            continue
        parsed = _parse_detectors_from_add_xml(f)
        if parsed:
            det_to_lane.update(parsed)

    if not det_to_lane:
        print("[WARN] No detectors found from additional-files in .sumocfg. "
              "Check that detectors.add.xml is listed in <additional-files>.")
    else:
        print(f"[INFO] Detectors loaded: {len(det_to_lane)}")

    return det_to_lane


# Backwards-compatible alias (your env imports this name)
def build_detector_lane_mapping(sumocfg_path: str) -> Dict[str, str]:
    return load_detectors_from_sumocfg(sumocfg_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python utils/detector_mapping.py path/to/osm.sumocfg")
        raise SystemExit(1)

    cfg = sys.argv[1]
    mapping = load_detectors_from_sumocfg(cfg)
    print("\nExamples (det_id -> lane_id):")
    for i, (k, v) in enumerate(list(mapping.items())[:20], start=1):
        print(f"{i:02d}. {k} -> {v}")
