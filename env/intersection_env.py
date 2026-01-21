# ======================================================================
# env/intersection_env.py — per-intersection logic (detector/lane-driven)
# ======================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import traci


@dataclass
class IntersectionCfg:
    max_lanes: int = 8
    min_green_seconds: int = 10
    yellow_seconds: int = 3  # reserved for future (only if you later add explicit yellow phases)
    reward_queue_w: float = 1.0
    reward_wait_w: float = 0.02


class IntersectionEnv:
    """
    A lightweight per-TLS helper that:
      - builds fixed-size observations (lane features with padding)
      - computes local reward
      - applies safe phase switching (min green, clamp phase index)

    NOTE: This is NOT a standalone Gym env.
    MultiIntersectionEnv owns TraCI start/load/step/reset.
    """

    def __init__(self, tls_id: str, controlled_lanes: List[str], cfg: Dict):
        self.tls_id = tls_id
        self.controlled_lanes = list(controlled_lanes)

        state_cfg = cfg.get("state", {})
        reward_cfg = cfg.get("reward", {})
        phase_cfg = cfg.get("phase", {})

        self.icfg = IntersectionCfg(
            max_lanes=int(state_cfg.get("max_lanes", 8)),
            min_green_seconds=int(phase_cfg.get("min_green_seconds", 10)),
            yellow_seconds=int(phase_cfg.get("yellow_seconds", 3)),
            reward_queue_w=float(reward_cfg.get("queue_w", 1.0)),
            reward_wait_w=float(reward_cfg.get("wait_w", 0.02)),
        )

        self.current_phase: int = 0
        self.time_in_phase: float = 0.0
        self._phase_count_cache: int = 1

        self.refresh_phase_count()

    # ----------------------------
    # Phase utilities
    # ----------------------------
    def refresh_phase_count(self) -> int:
        """
        Ask SUMO how many phases this TLS currently has.
        Some TLS may report only 1 phase (range [0,0]).
        """
        try:
            n = int(traci.trafficlight.getPhaseNumber(self.tls_id))
            self._phase_count_cache = max(1, n)
        except Exception:
            self._phase_count_cache = 1
        return self._phase_count_cache

    def reset(self) -> None:
        """
        Called by MultiIntersectionEnv.reset().
        Brings TLS back to a known phase safely.
        """
        self.refresh_phase_count()
        self.current_phase = 0
        self.time_in_phase = 0.0
        try:
            traci.trafficlight.setPhase(self.tls_id, 0)
        except Exception:
            # If SUMO refuses, we just keep internal state consistent.
            pass

    def apply_action_safe(self, desired_phase: int, step_length: float) -> None:
        """
        Safe switching rule:
          - enforce min-green hold time (prevents flicker)
          - clamp phase index to [0, n_phases-1]
        """
        self.time_in_phase += float(step_length)

        n_phases = self.refresh_phase_count()
        if n_phases <= 1:
            # Nothing to control.
            self.current_phase = 0
            return

        # Enforce minimum green time before switching
        if self.time_in_phase < self.icfg.min_green_seconds:
            return

        # Clamp desired phase
        if desired_phase < 0:
            desired_phase = 0
        if desired_phase >= n_phases:
            desired_phase = n_phases - 1

        if desired_phase == self.current_phase:
            return

        # Apply switch
        try:
            traci.trafficlight.setPhase(self.tls_id, int(desired_phase))
            self.current_phase = int(desired_phase)
            self.time_in_phase = 0.0
        except Exception:
            # If SUMO rejects, do not crash training.
            self.refresh_phase_count()
            self.current_phase = min(self.current_phase, self._phase_count_cache - 1)

    # ----------------------------
    # Observation + reward
    # ----------------------------
    def _lane_features(self, lane_id: str) -> Tuple[float, float, float]:
        """
        Lane features (sim-compatible but defensible as detector-derived):
          - queue length (vehicles with speed < 0.1)
          - accumulated waiting time (SUMO proxy; replace later with detector estimate if needed)
          - presence flag (any vehicle on lane)
        """
        try:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            presence = 1.0 if len(veh_ids) > 0 else 0.0
            queue = float(traci.lane.getLastStepHaltingNumber(lane_id))
            wait = float(traci.lane.getWaitingTime(lane_id))
            return queue, wait, presence
        except Exception:
            return 0.0, 0.0, 0.0

    def get_observation(self) -> np.ndarray:
        """
        Fixed-size lane-padded observation:
          obs = [ (q, wait, pres) for top-K lanes ] + [phase_norm, time_norm]
        """
        K = self.icfg.max_lanes
        lanes = self.controlled_lanes[:K]

        feats: List[float] = []
        for ln in lanes:
            q, w, p = self._lane_features(ln)
            feats.extend([q, w, p])

        # Pad if fewer than K lanes
        missing = K - len(lanes)
        if missing > 0:
            feats.extend([0.0, 0.0, 0.0] * missing)

        # Phase info (normalized)
        n_phases = max(1, self._phase_count_cache)
        phase_norm = float(self.current_phase) / float(max(1, n_phases - 1)) if n_phases > 1 else 0.0

        # Normalize time in phase (cap to 60s for stability)
        time_norm = min(self.time_in_phase, 60.0) / 60.0

        feats.extend([phase_norm, time_norm])
        return np.asarray(feats, dtype=np.float32)

    def compute_local_reward(self) -> float:
        """
        Local reward: negative queue and waiting time across controlled lanes.
        """
        total_q = 0.0
        total_w = 0.0
        for ln in self.controlled_lanes:
            q, w, _ = self._lane_features(ln)
            total_q += q
            total_w += w

        r = -(self.icfg.reward_queue_w * total_q + self.icfg.reward_wait_w * total_w)
        return float(r)
