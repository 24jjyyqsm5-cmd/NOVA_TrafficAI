# ======================================================================
# utils/movement_mapping.py
# ----------------------------------------------------------------------
# Minimal, robust movement mapping layer for detector-only PPO.
#
# Goals:
#   - Provide MovementMappings container required by state_builder + env
#   - Provide build_movement_mappings(tls_ids, det_to_lane, lane_candidates=None)
#       * lane_candidates is OPTIONAL
#       * if lane_candidates is None, build it using TraCI controlled lanes
#   - Provide get_movement_service_mask(tls_id, tls_signal_to_bucket)
#
# Notes:
#   - Avoids privileged simulation metrics
#   - Uses only:
#       * TraCI topology/controller interfaces
#       * det_to_lane mapping
# ======================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np

try:
    import traci  # type: ignore
except Exception as e:
    raise RuntimeError("[ERROR] traci not importable. Check SUMO / TraCI install.") from e


# Canonical 12 movement buckets (index only)
MOVES = [
    "N_thru", "N_left", "N_right",
    "S_thru", "S_left", "S_right",
    "E_thru", "E_left", "E_right",
    "W_thru", "W_left", "W_right",
]


@dataclass
class MovementMappings:
    """
    Required by state_builder:
      - tls_inlane_candidates[tls_id] -> dict[lane_id] -> bool
      - det_to_bucket[det_id] -> bucket_idx (0..11)
      - tls_signal_to_bucket[tls_id] -> dict[signal_index] -> bucket_idx
    """
    tls_inlane_candidates: Dict[str, Dict[str, bool]]
    det_to_bucket: Dict[str, int]
    tls_signal_to_bucket: Dict[str, Dict[int, int]]


def _bucket_from_lane_id(lane_id: str) -> int:
    """
    Heuristic bucket assignment from lane_id text.

    IMPORTANT:
    This is a safe fallback. With OSM lane IDs, direction inference is imperfect.
    Still, this gives stable buckets without crashing.

    - left/right inferred from substrings
    - direction inferred from N/E/S/W-ish substrings (often absent)
    - default direction group: N_*
    """
    s = lane_id.lower()

    is_left = ("left" in s) or (":l" in s) or ("_l" in s) or ("-l" in s)
    is_right = ("right" in s) or (":r" in s) or ("_r" in s) or ("-r" in s)

    dir_bucket_base = 0  # default N_*

    if "south" in s or s.startswith("s") or "_s" in s or "-s" in s:
        dir_bucket_base = 3
    elif "east" in s or s.startswith("e") or "_e" in s or "-e" in s:
        dir_bucket_base = 6
    elif "west" in s or s.startswith("w") or "_w" in s or "-w" in s:
        dir_bucket_base = 9

    if is_left and not is_right:
        return dir_bucket_base + 1
    if is_right and not is_left:
        return dir_bucket_base + 2
    return dir_bucket_base + 0


def _build_lane_candidates_from_traci(tls_ids: List[str]) -> Dict[str, Dict[str, bool]]:
    """
    lane_candidates per TLS using controller topology only:
      traci.trafficlight.getControlledLanes(tls_id)
    """
    out: Dict[str, Dict[str, bool]] = {}
    for tls_id in tls_ids:
        lanes: Set[str] = set()
        try:
            cl = traci.trafficlight.getControlledLanes(tls_id)
            for ln in cl:
                if ln:
                    lanes.add(str(ln))
        except Exception:
            pass
        out[tls_id] = {ln: True for ln in sorted(lanes)}
    return out


def _build_tls_signal_to_bucket(tls_ids: List[str]) -> Dict[str, Dict[int, int]]:
    """
    Build mapping from signal index -> movement bucket using controlledLinks.

    controlledLinks structure:
      list indexed by signalIndex:
        each item is list of link tuples: (inLane, outLane, viaLane)

    We map signalIndex -> bucket using the inbound lane heuristic.
    This is topology/controller-derived and does NOT use privileged metrics.
    """
    out: Dict[str, Dict[int, int]] = {}

    for tls_id in tls_ids:
        sig_map: Dict[int, int] = {}
        try:
            clinks = traci.trafficlight.getControlledLinks(tls_id)
            # clinks is a list where index == signal index
            for sig_idx, links_for_sig in enumerate(clinks):
                if not links_for_sig:
                    continue
                # pick first inbound lane we can find
                in_lane = None
                for link in links_for_sig:
                    # link is a tuple/list (inLane, outLane, viaLane)
                    if link and len(link) >= 1 and link[0]:
                        in_lane = str(link[0])
                        break
                if in_lane is None:
                    continue
                sig_map[int(sig_idx)] = int(_bucket_from_lane_id(in_lane))
        except Exception:
            # if anything fails, keep empty for that TLS
            sig_map = {}

        out[tls_id] = sig_map

    return out


def get_movement_service_mask(tls_id: str, tls_signal_to_bucket: Dict[str, Dict[int, int]]) -> np.ndarray:
    """
    Returns (12,) mask where 1 means that movement bucket is currently green.

    Uses current controller state string (RrGgYy...) and tls_signal_to_bucket mapping.
    """
    m = np.zeros(12, dtype=np.float32)

    sig_map = tls_signal_to_bucket.get(tls_id, {})
    if not sig_map:
        return m

    try:
        state = traci.trafficlight.getRedYellowGreenState(tls_id)
    except Exception:
        return m

    for idx, b in sig_map.items():
        if 0 <= idx < len(state) and 0 <= b < 12:
            if state[idx] in ("G", "g"):
                m[b] = 1.0

    return m


def build_movement_mappings(
    tls_ids: List[str],
    det_to_lane: Dict[str, str],
    lane_candidates: Optional[Dict[str, Dict[str, bool]]] = None,
) -> MovementMappings:
    """
    Build canonical movement mappings.

    lane_candidates is OPTIONAL.
      - If not provided, computed from TraCI controlled lanes.
    """
    tls_ids = list(tls_ids)

    if lane_candidates is None:
        lane_candidates = _build_lane_candidates_from_traci(tls_ids)

    det_to_bucket: Dict[str, int] = {}
    for det_id, lane_id in det_to_lane.items():
        try:
            det_to_bucket[str(det_id)] = int(_bucket_from_lane_id(str(lane_id)))
        except Exception:
            det_to_bucket[str(det_id)] = 0

    tls_signal_to_bucket = _build_tls_signal_to_bucket(tls_ids)

    return MovementMappings(
        tls_inlane_candidates=lane_candidates,
        det_to_bucket=det_to_bucket,
        tls_signal_to_bucket=tls_signal_to_bucket,
    )
