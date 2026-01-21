# ======================================================================
# utils/state_builder.py
# ----------------------------------------------------------------------
# Detector-only state builder for shared-policy PPO with canonical movement
# buckets + movement-service mask.
#
# Output per TLS: 62 dims
#   12 movements x 4 features = 48
#   + 12 service mask = 60
#   + time_in_phase, eligible = 62
# ======================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import traci  # type: ignore
except Exception as e:
    raise RuntimeError(
        "[ERROR] traci is not importable. Ensure SUMO/TraCI is installed and importable."
    ) from e

from utils.movement_mapping import MovementMappings, get_movement_service_mask

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


@dataclass
class TLSBucketDetectors:
    """
    tls_bucket_dets[tls_id][bucket_idx] -> list[det_id]
    """
    tls_bucket_dets: Dict[str, Dict[int, List[str]]]


def build_tls_bucket_detectors(
    tls_ids: List[str],
    det_to_lane: Dict[str, str],
    mappings: MovementMappings,
) -> TLSBucketDetectors:
    """
    Attach detectors to each TLS and movement bucket.

    A detector is attached to TLS if its lane is among that TLS's controlled lanes
    (mappings.tls_inlane_candidates).
    """
    tls_bucket_dets: Dict[str, Dict[int, List[str]]] = {}

    for tls_id in tls_ids:
        lane_candidates = mappings.tls_inlane_candidates.get(tls_id, {})
        lanes_in_tls = set(lane_candidates.keys())

        bucket_map: Dict[int, List[str]] = {b: [] for b in range(12)}
        for det_id, bucket in mappings.det_to_bucket.items():
            lane = det_to_lane.get(det_id)
            if lane and lane in lanes_in_tls:
                bucket_map[int(bucket)].append(det_id)

        tls_bucket_dets[tls_id] = bucket_map

    return TLSBucketDetectors(tls_bucket_dets=tls_bucket_dets)


def read_lanearea_detector_value(det_id: str) -> float:
    """
    Read E2 detector values robustly:
      1) occupancy -> [0..1]
      2) vehicle count -> presence
      3) jam length (veh) -> presence
    """
    try:
        occ = traci.lanearea.getLastStepOccupancy(det_id)
        occ_f = _safe_float(occ, 0.0)
        if occ_f > 1.0:
            occ_f = occ_f / 100.0
        return _clip01(occ_f)
    except Exception:
        pass

    try:
        n = traci.lanearea.getLastStepVehicleNumber(det_id)
        n_f = _safe_float(n, 0.0)
        return 1.0 if n_f > 0.0 else 0.0
    except Exception:
        pass

    try:
        j = traci.lanearea.getJamLengthVehicle(det_id)
        j_f = _safe_float(j, 0.0)
        return 1.0 if j_f > 0.0 else 0.0
    except Exception:
        pass

    return 0.0


def aggregate_bucket_value(det_ids: List[str]) -> float:
    if not det_ids:
        return 0.0
    vals = [read_lanearea_detector_value(d) for d in det_ids]
    return float(max(vals)) if vals else 0.0


class DetectorOnlyStateBuilder:
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

        self.q_proxy: Dict[str, np.ndarray] = {}
        self.starve: Dict[str, np.ndarray] = {}
        self.occ_prev: Dict[str, np.ndarray] = {}

        self.last_phase_index: Dict[str, int] = {}
        self.time_in_phase: Dict[str, float] = {}

        self._init_tls_state()

    def _init_tls_state(self) -> None:
        for tls_id in self.tls_ids:
            self.q_proxy[tls_id] = np.zeros(12, dtype=np.float32)
            self.starve[tls_id] = np.zeros(12, dtype=np.float32)
            self.occ_prev[tls_id] = np.zeros(12, dtype=np.float32)

            try:
                ph = int(traci.trafficlight.getPhase(tls_id))
            except Exception:
                ph = -1

            self.last_phase_index[tls_id] = ph
            self.time_in_phase[tls_id] = 0.0

    def reset(self) -> None:
        self._init_tls_state()

    def _update_time_in_phase(self, tls_id: str) -> Tuple[float, float]:
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
        bucket_map = self.tls_bucket_dets.tls_bucket_dets.get(tls_id, {})
        occ = np.zeros(12, dtype=np.float32)
        for b in range(12):
            dets = bucket_map.get(b, [])
            occ[b] = float(aggregate_bucket_value(dets))
        return occ

    def _infer_service(self, occ_now: np.ndarray, occ_prev: np.ndarray, served_mask: np.ndarray) -> np.ndarray:
        served = np.zeros(12, dtype=np.float32)
        for i in range(12):
            if served_mask[i] > 0.5 and (occ_prev[i] - occ_now[i]) > 0.05:
                served[i] = 1.0
        return served

    def build_obs_for_tls(self, tls_id: str) -> np.ndarray:
        occ_stop = self._read_occ_stop(tls_id)
        served_mask = get_movement_service_mask(tls_id, self.mappings.tls_signal_to_bucket)
        tip, eligible = self._update_time_in_phase(tls_id)

        occ_prev = self.occ_prev.get(tls_id, np.zeros(12, dtype=np.float32))
        did_serve = self._infer_service(occ_stop, occ_prev, served_mask)

        q_prev = self.q_proxy.get(tls_id, np.zeros(12, dtype=np.float32))
        q_new = self.beta * q_prev + (1.0 - self.beta) * occ_stop
        trend = q_new - q_prev

        starve = self.starve.get(tls_id, np.zeros(12, dtype=np.float32))
        for i in range(12):
            demand_present = occ_stop[i] > 0.1
            if demand_present and did_serve[i] < 0.5:
                starve[i] += self.starve_inc * self.step_length
            elif did_serve[i] > 0.5:
                starve[i] = 0.0

        # Save
        self.q_proxy[tls_id] = q_new
        self.starve[tls_id] = starve
        self.occ_prev[tls_id] = occ_stop

        feat = np.concatenate(
            [
                occ_stop.astype(np.float32),   # 12
                q_new.astype(np.float32),      # 12
                trend.astype(np.float32),      # 12
                starve.astype(np.float32),     # 12
                served_mask.astype(np.float32),# 12
                np.array([tip, eligible], dtype=np.float32),  # 2
            ],
            axis=0,
        )

        if feat.shape[0] != 62:
            raise RuntimeError(f"[ERROR] Obs dim mismatch for {tls_id}: got {feat.shape[0]}, expected 62")

        return feat

    def build_obs_all(self) -> np.ndarray:
        obs_list = [self.build_obs_for_tls(tls_id) for tls_id in self.tls_ids]
        return np.stack(obs_list, axis=0)

    def get_obs_dim(self) -> int:
        return 62
