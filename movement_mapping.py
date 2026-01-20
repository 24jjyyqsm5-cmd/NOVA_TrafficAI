# ======================================================================
# utils/movement_mapping.py
# ----------------------------------------------------------------------
# Canonical movement buckets:
#   Approaches: North, South, East, West
#   Movements : Left, Through, Right
# => 12 buckets total.
#
# Provides:
#   - MOVES (len=12)
#   - MovementMappings dataclass
#   - build_movement_mappings(...)
#   - get_movement_service_mask(tls_id, tls_signal_to_bucket)
#
# This version is designed to be robust and "good enough" for training:
#  - Uses SUMO signal state string to infer which lane indices are green
#  - Maps lane indices -> movement buckets using a lane_candidates dict
#    (provided by your build logic from net + detector mapping).
#
# IMPORTANT:
#   "Perfect" L/T/R classification from geometry is a deep topic.
#   This file focuses on producing:
#     (1) stable 12-bucket mapping
#     (2) defensible service mask for RL gating/reward
# ======================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import traci  # type: ignore
except Exception as e:
    raise RuntimeError(
        "[ERROR] traci is not importable. Ensure SUMO/TraCI is installed."
    ) from e

# Canonical bucket ordering: N(L,T,R), S(L,T,R), E(L,T,R), W(L,T,R)
MOVES: List[str] = [
    "N_L", "N_T", "N_R",
    "S_L", "S_T", "S_R",
    "E_L", "E_T", "E_R",
    "W_L", "W_T", "W_R",
]


def _bucket_idx(approach: str, turn: str) -> int:
    """
    approach in {N,S,E,W}, turn in {L,T,R}
    """
    a_map = {"N": 0, "S": 1, "E": 2, "W": 3}
    t_map = {"L": 0, "T": 1, "R": 2}
    return a_map[approach] * 3 + t_map[turn]


@dataclass
class MovementMappings:
    """
    tls_inlane_candidates:
      tls_inlane_candidates[tls_id][lane_id] -> (approach, turn, tls_lane_index)

    det_to_bucket:
      det_to_bucket[det_id] -> bucket_idx (0..11)

    tls_signal_to_bucket:
      tls_signal_to_bucket[tls_id][tls_lane_index] -> bucket_idx
    """
    tls_inlane_candidates: Dict[str, Dict[str, Tuple[str, str, int]]]
    det_to_bucket: Dict[str, int]
    tls_signal_to_bucket: Dict[str, Dict[int, int]]


def build_movement_mappings(
    tls_ids: List[str],
    det_to_lane: Dict[str, str],
    lane_candidates: Dict[str, Dict[str, Tuple[str, str, int]]],
) -> MovementMappings:
    """
    Build mappings given:
      - tls_ids
      - det_to_lane
      - lane_candidates per TLS (lane -> (approach, turn, tls_lane_index))

    This keeps the mapping local & consistent.
    """
    tls_inlane_candidates: Dict[str, Dict[str, Tuple[str, str, int]]] = {}
    det_to_bucket: Dict[str, int] = {}
    tls_signal_to_bucket: Dict[str, Dict[int, int]] = {}

    for tls_id in tls_ids:
        cand = lane_candidates.get(tls_id, {})
        tls_inlane_candidates[tls_id] = cand

        sig_map: Dict[int, int] = {}
        for lane_id, (app, turn, tls_idx) in cand.items():
            b = _bucket_idx(app, turn)
            sig_map[int(tls_idx)] = b
        tls_signal_to_bucket[tls_id] = sig_map

    # Detector -> bucket uses detector lane, but must match a TLS candidate lane
    for det_id, lane_id in det_to_lane.items():
        for tls_id in tls_ids:
            cand = tls_inlane_candidates.get(tls_id, {})
            if lane_id in cand:
                app, turn, _tls_idx = cand[lane_id]
                det_to_bucket[det_id] = _bucket_idx(app, turn)
                break

    return MovementMappings(
        tls_inlane_candidates=tls_inlane_candidates,
        det_to_bucket=det_to_bucket,
        tls_signal_to_bucket=tls_signal_to_bucket,
    )


def get_movement_service_mask(
    tls_id: str,
    tls_signal_to_bucket: Dict[str, Dict[int, int]],
) -> np.ndarray:
    """
    Returns (12,) float mask where 1 means "this movement is currently being served"
    based on current signal state string.

    We interpret any of {G,g} as served for that tls_lane_index.
    """
    mask = np.zeros(12, dtype=np.float32)

    try:
        state = traci.trafficlight.getRedYellowGreenState(tls_id)
    except Exception:
        return mask

    idx_map = tls_signal_to_bucket.get(tls_id, {})
    if not idx_map:
        return mask

    # state is a string, each char corresponds to a link index in SUMO TLS logic
    # Common green chars: 'G' (major green), 'g' (minor green)
    for i, ch in enumerate(state):
        if ch in ("G", "g"):
            b = idx_map.get(i, None)
            if b is not None and 0 <= b < 12:
                mask[b] = 1.0

    return mask
