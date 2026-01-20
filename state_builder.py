# ======================================================================
# utils/state_builder.py
# ----------------------------------------------------------------------
# Detector-only state builder for shared-policy PPO with canonical movement
# buckets (Approach 2) + binary movement-service mask (M1).
#
# Requires:
#   - TraCI connected (SUMO running)
#   - det_to_lane mapping (from utils/detector_mapping.py)
#   - movement mappings (from utils/movement_mapping.py)
#
# Outputs per TLS:
#   Observation vector length = 12*4 + 12 + 2 = 62
#     - 12 movements x 4 detector-derived features:
#         1) occ_stop (0..1)  [occupancy if available else presence]
#         2) q_proxy  (EWMA of occ_stop)
#         3) trend    (q_proxy - prev_q_proxy)
#         4) starvation timer (normalized optional; raw here)
#     - 12 movement-service mask (M1): 1 if movement currently green
#     - time_in_phase (seconds, tracked locally)
#     - eligible_to_change (1 if time_in_phase >= min_green else 0)
#
# No simulator "privileged" lane waiting time is used here.
# ======================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import traci  # type: ignore
except Exception as e:
    raise RuntimeError(
        "[ERROR] traci is not importable. Ensure SUMO/TraCI is installed and "
        "your Python environment can import traci."
    ) from e

from utils.movement_mapping import (
    MovementMappings,
    get_movement_service_mask,
    MOVES,
)


# ------------------------------
# Helper / config defaults
# ------------------------------
DEFAULT_EWMA_BETA = 0.95
DEFAULT_MIN_GREEN = 10.0
DEFAULT_STEP_LENGTH = 1.0


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ------------------------------
# Data containers
# ------------------------------
@dataclass
class TLSBucketDetectors:
    """
    For each TLS, list of detectors in each canonical movement bucket.
    tls_bucket_dets[tls_id][bucket_idx] -> list[det_id]
    """
    tls_bucket_dets: Dict[str, Dict[int, List[str]]]


# ------------------------------
# Building detector grouping per TLS
# ------------------------------
def build_tls_bucket_detectors(
    tls_ids: List[str],
    det_to_lane: Dict[str, str],
    mappings: MovementMappings,
) -> TLSBucketDetectors:
    """
    Assign stop-line detectors to a TLS and movement bucket.

    We only attach a detector to a given TLS if the detector's lane appears
    among that TLS's inbound lane candidates (mappings.tls_inlane_candidates).

    This keeps the state "local" to each intersection, even if detector ids
    are global.
    """
    tls_bucket_dets: Dict[str, Dict[int, List[str]]] = {}

    for tls_id in tls_ids:
        lane_candidates = mappings.tls_inlane_candidates.get(tls_id, {})
        lanes_in_tls = set(lane_candidates.keys())

        bucket_map: Dict[int, List[str]] = {b: [] for b in range(12)}
        for det_id, bucket in mappings.det_to_bucket.items():
            lane = det_to_lane.get(det_id)
            if lane and lane in lanes_in_tls:
                bucket_map[bucket].append(det_id)

        tls_bucket_dets[tls_id] = bucket_map

    return TLSBucketDetectors(tls_bucket_dets=tls_bucket_dets)


# ------------------------------
# Reading E2 detector signals robustly
# ------------------------------
def read_lanearea_detector_value(det_id: str) -> float:
    """
    Returns a stop-line "occupancy-like" value in [0,1] if possible.

    Priority:
      1) lanearea.getLastStepOccupancy()  -> typically 0..100 (%)
      2) lanearea.getLastStepVehicleNumber() -> presence -> 0 or 1
      3) lanearea.getJamLengthVehicle() -> presence -> 0 or 1
    """
    # 1) Occupancy
    try:
        occ = traci.lanearea.getLastStepOccupancy(det_id)
        # SUMO often returns occupancy in percent [0..100]
        occ_f = _safe_float(occ, 0.0)
        if occ_f > 1.0:
            occ_f = occ_f / 100.0
        return _clip01(occ_f)
    except Exception:
        pass

    # 2) Vehicle number -> presence
    try:
        n = traci.lanearea.getLastStepVehicleNumber(det_id)
        n_f = _safe_float(n, 0.0)
        return 1.0 if n_f > 0.0 else 0.0
    except Exception:
        pass

    # 3) Jam length by vehicles -> presence
    try:
        j = traci.lanearea.getJamLengthVehicle(det_id)
        j_f = _safe_float(j, 0.0)
        return 1.0 if j_f > 0.0 else 0.0
    except Exception:
        pass

    return 0.0


def aggregate_bucket_value(det_ids: List[str]) -> float:
    """
    Aggregate multiple detectors in the same movement bucket into one value.
    We use max() because presence/occupancy near stop line should indicate demand
    if any lane in that movement is occupied.
    """
    if not det_ids:
        return 0.0
    vals = [read_lanearea_detector_value(d) for d in det_ids]
    return float(max(vals)) if vals else 0.0


# ------------------------------
# State Builder
# ------------------------------
class DetectorOnlyStateBuilder:
    """
    Produces detector-only observations for all TLSs.

    Tracks per-TLS:
      - q_proxy (EWMA) per movement bucket
      - previous q_proxy for trend
      - starvation timers per movement bucket
      - time_in_phase (seconds) using TraCI phase index changes
      - previous occ_stop to infer "service" during green

    NOTE:
      We do NOT use lane waiting time / queue from simulator.
      We only use detector readings + controller state.
    """

    def __init__(
        self,
        tls_ids: List[str],
        tls_bucket_dets: TLSBucketDetectors,
        mappings: MovementMappings,
        step_length: float = DEFAULT_STEP_LENGTH,
        min_green: float = DEFAULT_MIN_GREEN,
        ewma_beta: float = DEFAULT_EWMA_BETA,
        starve_increment: float = 1.0,
        debug: bool = False,
    ):
        self.tls_ids = list(tls_ids)
        self.tls_bucket_dets = tls_bucket_dets
        self.mappings = mappings

        self.step_length = float(step_length)
        self.min_green = float(min_green)
        self.beta = float(ewma_beta)
        self.starve_inc = float(starve_increment)
        self.debug = bool(debug)

        # Per-TLS tracked state
        self.q_proxy: Dict[str, np.ndarray] = {}
        self.q_proxy_prev: Dict[str, np.ndarray] = {}
        self.starve: Dict[str, np.ndarray] = {}
        self.occ_prev: Dict[str, np.ndarray] = {}

        self.last_phase_index: Dict[str, int] = {}
        self.time_in_phase: Dict[str, float] = {}

        self._init_tls_state()

    def _init_tls_state(self) -> None:
        for tls_id in self.tls_ids:
            self.q_proxy[tls_id] = np.zeros(12, dtype=np.float32)
            self.q_proxy_prev[tls_id] = np.zeros(12, dtype=np.float32)
            self.starve[tls_id] = np.zeros(12, dtype=np.float32)
            self.occ_prev[tls_id] = np.zeros(12, dtype=np.float32)

            try:
                ph = int(traci.trafficlight.getPhase(tls_id))
            except Exception:
                ph = -1

            self.last_phase_index[tls_id] = ph
            self.time_in_phase[tls_id] = 0.0

    def reset(self) -> None:
        """Reset internal trackers (call at episode reset)."""
        self._init_tls_state()

    def _update_time_in_phase(self, tls_id: str) -> Tuple[float, float]:
        """
        Updates and returns (time_in_phase, eligible_to_change).
        Uses TraCI phase index changes to detect phase changes.
        """
        try:
            ph = int(traci.trafficlight.getPhase(tls_id))
        except Exception:
            ph = self.last_phase_index.get(tls_id, -1)

        last_ph = self.last_phase_index.get(tls_id, -1)

        if ph != last_ph:
            self.last_phase_index[tls_id] = ph
            self.time_in_phase[tls_id] = 0.0
        else:
            self.time_in_phase[tls_id] = self.time_in_phase.get(tls_id, 0.0) + self.step_length

        tip = self.time_in_phase[tls_id]
        eligible = 1.0 if tip >= self.min_green else 0.0
        return tip, eligible

    def _read_occ_stop(self, tls_id: str) -> np.ndarray:
        """
        Read aggregated stop-line detector values per bucket for one TLS.
        """
        bucket_map = self.tls_bucket_dets.tls_bucket_dets.get(tls_id, {})
        occ = np.zeros(12, dtype=np.float32)
        for b in range(12):
            dets = bucket_map.get(b, [])
            occ[b] = float(aggregate_bucket_value(dets))
        return occ

    def _infer_service(self, occ_now: np.ndarray, occ_prev: np.ndarray, served_mask: np.ndarray) -> np.ndarray:
        """
        Infer whether a movement was served this step using only detector trends.

        Basic heuristic:
          If movement is green AND occ decreased compared to previous -> served = 1
          Else 0

        This is intentionally conservative and detector-only.
        """
        served = np.zeros(12, dtype=np.float32)
        for i in range(12):
            if served_mask[i] > 0.5 and (occ_prev[i] - occ_now[i]) > 0.05:
                served[i] = 1.0
        return served

    def build_obs_for_tls(self, tls_id: str) -> np.ndarray:
        """
        Returns observation vector shape (62,) for one TLS.
        """
        # 1) detector bucket aggregates
        occ_stop = self._read_occ_stop(tls_id)

        # 2) movement-service mask (M1) from controller state
        served_mask = get_movement_service_mask(tls_id, self.mappings.tls_signal_to_bucket)

        # 3) time in phase + eligible-to-change (min green gating)
        tip, eligible = self._update_time_in_phase(tls_id)

        # 4) inferred service from detector change during green
        occ_prev = self.occ_prev.get(tls_id, np.zeros(12, dtype=np.float32))
        did_serve = self._infer_service(occ_stop, occ_prev, served_mask)

        # 5) EWMA queue proxy + trend
        q_prev = self.q_proxy.get(tls_id, np.zeros(12, dtype=np.float32))
        q_prev_saved = q_prev.copy()

        # Update EWMA: q = beta*q + (1-beta)*occ_stop
        q_new = self.beta * q_prev + (1.0 - self.beta) * occ_stop

        # trend = q_new - q_prev_saved
        trend = q_new - q_prev_saved

        # 6) starvation update (demand present + not served)
        starve = self.starve.get(tls_id, np.zeros(12, dtype=np.float32))
        for i in range(12):
            demand_present = occ_stop[i] > 0.1  # threshold for "there is demand"
            if demand_present and did_serve[i] < 0.5:
                starve[i] += self.starve_inc * self.step_length
            elif did_serve[i] > 0.5:
                starve[i] = 0.0

        # Save state
        self.q_proxy_prev[tls_id] = q_prev_saved
        self.q_proxy[tls_id] = q_new
        self.starve[tls_id] = starve
        self.occ_prev[tls_id] = occ_stop

        # 7) Assemble obs vector (62,)
        # movement features (12 x 4): occ_stop, q_proxy, trend, starve
        feat = np.concatenate(
            [
                occ_stop.astype(np.float32),
                q_new.astype(np.float32),
                trend.astype(np.float32),
                starve.astype(np.float32),
                served_mask.astype(np.float32),
                np.array([tip, eligible], dtype=np.float32),
            ],
            axis=0,
        )

        # Safety assert
        if feat.shape[0] != 62:
            raise RuntimeError(f"[ERROR] Obs dim mismatch for {tls_id}: got {feat.shape[0]}, expected 62")

        return feat

    def build_obs_all(self) -> np.ndarray:
        """
        Returns stacked observations for all TLS IDs:
          shape (N, 62)
        """
        obs_list = [self.build_obs_for_tls(tls_id) for tls_id in self.tls_ids]
        return np.stack(obs_list, axis=0)

    def get_obs_dim(self) -> int:
        return 62

    def get_tls_ids(self) -> List[str]:
        return list(self.tls_ids)

    def debug_print_one(self, tls_id: str) -> None:
        """
        Print a quick debug snapshot of the current observation.
        """
        obs = self.build_obs_for_tls(tls_id)
        print(f"\n[TLS {tls_id}] obs dim={obs.shape[0]} (expected 62)")
        print("  occ_stop:", obs[0:12])
        print("  q_proxy :", obs[12:24])
        print("  trend   :", obs[24:36])
        print("  starve  :", obs[36:48])
        print("  mask(M1):", obs[48:60])
        print("  tip,elig:", obs[60:62])
