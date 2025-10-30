from __future__ import annotations
import numpy as np
import traci
from typing import List, Dict, Any, Optional


class IntersectionEnv:
    """
    Lightweight wrapper around one SUMO TLS.
    Produces a fixed-length feature vector (12 dim) per intersection:

      [qN, qE, qS, qW, vN, vE, vS, vW, press_in, press_out, cur_phase_idx_norm, time_in_phase_norm]

    Directions are aggregated by lane headings (N/E/S/W) using lane geometry (link edges).
    If a direction has no lanes, its features are 0.

    NOTE: We keep features compact+fixed to make the global obs shape stable.
    """

    DIRS = ("N", "E", "S", "W")

    def __init__(self, tls_id: str, cfg: Dict[str, Any]):
        self.tls_id = tls_id
        self.cfg = cfg
        self.phase_count: int = 0
        self.incoming_lanes: List[str] = []
        self.outgoing_lanes: List[str] = []
        self._phase_time: float = 0.0
        self._last_phase: Optional[int] = None

    # ---------- SUMO helpers ----------
    def init_from_sumo(self):
        # Phase count
        # robust: some tls have multiple programs; use current logic's number of phases
        prog = traci.trafficlight.getProgram(self.tls_id)
        all_logics = traci.trafficlight.getAllProgramLogics(self.tls_id)
        self.phase_count = 0
        for logic in all_logics:
            if logic.programID == prog:
                self.phase_count = len(logic.getPhases())
                break
        if self.phase_count == 0:
            # fallback: detect from tls state string length and greens
            self.phase_count = traci.trafficlight.getPhaseNumber(self.tls_id) if hasattr(
                traci.trafficlight, "getPhaseNumber"
            ) else len(traci.trafficlight.getRedYellowGreenState(self.tls_id))

        # Lanes
        self.incoming_lanes = []
        self.outgoing_lanes = []
        controlled_links = traci.trafficlight.getControlledLinks(self.tls_id)
        # controlled_links is list per signal index -> list of (incoming, outgoing, via)
        for links in controlled_links:
            for inc, out, _via in links:
                if inc not in self.incoming_lanes:
                    self.incoming_lanes.append(inc)
                if out not in self.outgoing_lanes:
                    self.outgoing_lanes.append(out)

        # reset phase timer tracking
        self._last_phase = self.current_phase()
        self._phase_time = 0.0

    def current_phase(self) -> int:
        try:
            return int(traci.trafficlight.getPhase(self.tls_id))
        except Exception:
            return 0

    def set_phase(self, idx: int):
        # guard
        if self.phase_count <= 0:
            return
        idx = int(idx) % self.phase_count
        traci.trafficlight.setPhase(self.tls_id, idx)

    # ---------- lane stats ----------
    @staticmethod
    def _lane_queue(lane: str) -> float:
        # vehicles with speed < 0.1 m/s
        return float(traci.lane.getLastStepHaltingNumber(lane))

    @staticmethod
    def _lane_speed(lane: str) -> float:
        v = traci.lane.getLastStepMeanSpeed(lane)
        return float(0.0 if np.isnan(v) else v)

    @staticmethod
    def _lane_density(lane: str) -> float:
        # vehs per km
        return float(traci.lane.getLastStepVehicleNumber(lane))

    # ---------- direction mapping ----------
    def _dir_key(self, lane_id: str) -> str:
        # Heuristic: use edge id suffix/heading hints; fallback to evenly distributing by index
        # Common OSM edges carry "_N", "_E", "_S", "_W" or similar. Try to pick that up.
        lid = lane_id.lower()
        for d in self.DIRS:
            if lid.endswith(f"_{d.lower()}") or f"_{d.lower()}_" in lid or lid.endswith(d.lower()):
                return d
        # fallback: hash position
        return self.DIRS[hash(lane_id) % 4]

    def _aggregate_by_dir(self, lanes: List[str]):
        q = {d: 0.0 for d in self.DIRS}
        v = {d: 0.0 for d in self.DIRS}
        n = {d: 0 for d in self.DIRS}

        for l in lanes:
            d = self._dir_key(l)
            q[d] += self._lane_queue(l)
            v[d] += self._lane_speed(l)
            n[d] += 1

        for d in self.DIRS:
            if n[d] > 0:
                v[d] /= n[d]
            else:
                v[d] = 0.0
        return q, v

    # ---------- features ----------
    def observe(self, dt: float, neighbor_pressure: float = 0.0) -> np.ndarray:
        """
        Return exactly 12 features.
        neighbor_pressure: external scalar (avg neighbors' (qin - qout)) if provided by MultiEnv.
        """
        q_in, v_in = self._aggregate_by_dir(self.incoming_lanes)

        # internal "pressure": in - out
        total_in = sum(self._lane_density(l) for l in self.incoming_lanes)
        total_out = sum(self._lane_density(l) for l in self.outgoing_lanes)
        press_in = float(total_in - total_out)
        press_out = float(total_out - total_in)  # symmetric info

        # phase/time features
        cur_phase = self.current_phase()
        if cur_phase == self._last_phase:
            self._phase_time += dt
        else:
            self._last_phase = cur_phase
            self._phase_time = 0.0

        # normalize lightweight
        vmax = 15.0  # ~54 km/h
        qmax = 20.0  # halting vehicles per direction (rough cap for stability)
        tnorm = min(self._phase_time / 60.0, 1.0)  # cap at 60s
        phasenorm = 0.0 if self.phase_count <= 1 else cur_phase / float(self.phase_count - 1)

        feat = np.array([
            min(q_in["N"] / qmax, 1.0),
            min(q_in["E"] / qmax, 1.0),
            min(q_in["S"] / qmax, 1.0),
            min(q_in["W"] / qmax, 1.0),
            min(v_in["N"] / vmax, 1.0),
            min(v_in["E"] / vmax, 1.0),
            min(v_in["S"] / vmax, 1.0),
            min(v_in["W"] / vmax, 1.0),
            np.tanh(press_in / qmax),
            np.tanh(press_out / qmax),
            phasenorm,
            tnorm,
        ], dtype=np.float32)

        # add small neighbor coordination via press_in (already included) + external hint
        if neighbor_pressure != 0.0:
            # Blend neighbor pressure by slightly adjusting internal pressure components
            adj = np.tanh(neighbor_pressure / qmax) * 0.25
            feat[8] = np.clip(feat[8] + adj, -1.0, 1.0)
            feat[9] = np.clip(feat[9] - adj, -1.0, 1.0)

        return feat

    # ---------- reward ----------
    def compute_reward(self, smooth_accel_ratio: float = 0.0) -> float:
        # Negative wait + queue + small positive throughput + small smoothness term
        total_wait = 0.0
        total_queue = 0.0
        throughput = 0.0

        for l in self.incoming_lanes:
            total_queue += self._lane_queue(l)
            # SUMO does not give direct "waiting time per lane" last step; approximate via halting number
            total_wait += self._lane_queue(l)  # proxy
            throughput += traci.lane.getLastStepVehicleNumber(l)

        rw = self.cfg["reward"]
        reward = (
            -rw.get("wait", 1.0) * total_wait
            -rw.get("queue", 0.1) * total_queue
            +rw.get("throughput", 0.003) * throughput
            +rw.get("smooth", 0.05) * smooth_accel_ratio
        )
        return float(reward)
