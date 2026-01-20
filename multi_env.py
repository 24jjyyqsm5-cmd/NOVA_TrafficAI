# ======================================================================
# env/multi_env.py
# ----------------------------------------------------------------------
# MultiIntersectionEnv
# - One env controls N TLS (one action per TLS)
# - Per-TLS MultiDiscrete action space (each TLS has its own phase count)
# - Safety:
#     - per-TLS min green gating
#     - clamp phase index to [0, phase_count-1]
#     - explicit yellow insertion using setRedYellowGreenState()
# - Control hardening:
#     - force a stable TLS program (first available)
#     - lock phases using setPhaseDuration(very_large) so SUMO doesn't auto-advance
# ======================================================================

from __future__ import annotations

import os
import time
import random
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym  # type: ignore
    from gym import spaces  # type: ignore

import traci

from utils.config import load_config
from utils.sumo_utils import ensure_sumo_binary, build_sumo_cmd

# detector mapping import: support either function name
try:
    from utils.detector_mapping import build_detector_lane_mapping  # type: ignore
except Exception:
    build_detector_lane_mapping = None  # type: ignore


class MultiIntersectionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.cfg = cfg or load_config()

        sumo_cfg_path = self.cfg.get("sumo", {}).get("config_file") or self.cfg.get("sumo", {}).get("config")
        if not sumo_cfg_path:
            raise ValueError("[ERROR] sumo.config_file missing in config.yaml")

        self.sumo_cfg_path = str(sumo_cfg_path)
        self.gui = bool(self.cfg.get("sumo", {}).get("gui", False))
        self.step_length = float(self.cfg.get("sumo", {}).get("step_length", 1.0))
        self.episode_seconds = float(self.cfg.get("training", {}).get("episode_seconds", 900))

        # Safety params
        self.min_green_seconds = float(self.cfg.get("phase", {}).get("min_green_seconds", 10))
        self.yellow_seconds = float(self.cfg.get("phase", {}).get("yellow_seconds", 4))

        # Training config
        self.max_tls = int(self.cfg.get("training", {}).get("max_tls", 24))
        self.tls_ids: List[str] = []

        # runtime
        self._sumo_proc: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None

        # per-TLS phase metadata
        self._phase_counts: Dict[str, int] = {}
        self._program_ids: Dict[str, str] = {}

        self._last_applied_phase: Dict[str, int] = {}
        self._time_in_phase: Dict[str, float] = {}

        # yellow transition tracking
        self._yellow_remaining_steps: Dict[str, int] = {}
        self._pending_target_phase: Dict[str, Optional[int]] = {}
        self._yellow_state_cache: Dict[Tuple[str, int, int], str] = {}

        # detectors
        self.det_to_lane: Dict[str, str] = {}

        # diagnostics counters (per 500 steps snapshot)
        self._diag_req_switch: Dict[str, int] = {}
        self._diag_applied_by_us: Dict[str, int] = {}
        self._diag_min_green_blocks: Dict[str, int] = {}
        self._diag_overridden_by_sumo: Dict[str, int] = {}

        # start SUMO once so we can discover TLS + define action space properly
        self._start_sumo_and_init()

        # ---- Build per-TLS action space ----
        nvec: List[int] = []
        for tls_id in self.tls_ids:
            c = int(self._phase_counts.get(tls_id, 1))
            c = max(1, c)
            nvec.append(c)

        self.action_space = spaces.MultiDiscrete(nvec=np.array(nvec, dtype=np.int64))

        # ---- Observation space ----
        # NOTE: Keep your existing state builder; this placeholder matches your current run.
        obs_dim = int(self.cfg.get("state", {}).get("obs_dim", 1488))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # reset counters
        self.sim_time = 0.0
        self.sim_step = 0

    # --------------------------
    # SUMO startup / shutdown
    # --------------------------
    def _pick_port(self) -> int:
        return random.randint(50000, 65000)

    def _start_sumo_and_init(self) -> None:
        sumo_bin = ensure_sumo_binary(gui=self.gui)
        self._port = self._pick_port()

        cmd = build_sumo_cmd(self.cfg, sumo_bin, self.sumo_cfg_path)
        cmd = cmd + ["--remote-port", str(self._port)]

        self._sumo_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        # Connect TraCI
        traci.init(self._port)

        # Load detectors mapping (optional)
        try:
            if build_detector_lane_mapping is not None:
                self.det_to_lane = build_detector_lane_mapping(self.sumo_cfg_path)  # type: ignore
            else:
                self.det_to_lane = {}
        except Exception as e:
            print(f"[WARN] detector mapping failed: {e}")
            self.det_to_lane = {}

        # Discover TLS
        all_tls = list(traci.trafficlight.getIDList())
        if not all_tls:
            raise RuntimeError("[ERROR] No traffic lights found in the network.")

        requested = self.cfg.get("training", {}).get("tls_ids", []) or []
        if requested:
            self.tls_ids = [t for t in requested if t in all_tls]
        else:
            self.tls_ids = all_tls[: self.max_tls]

        # Phase counts + force control program
        for tls_id in self.tls_ids:
            self._diag_req_switch[tls_id] = 0
            self._diag_applied_by_us[tls_id] = 0
            self._diag_min_green_blocks[tls_id] = 0
            self._diag_overridden_by_sumo[tls_id] = 0

            # read full logic
            try:
                defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
                if not defs:
                    self._phase_counts[tls_id] = 1
                    self._program_ids[tls_id] = "0"
                else:
                    logic = defs[0]
                    phases = getattr(logic, "phases", [])
                    self._phase_counts[tls_id] = max(1, len(phases))
                    # programID is commonly "0"
                    pid = getattr(logic, "programID", None)
                    self._program_ids[tls_id] = str(pid) if pid is not None else "0"
            except Exception:
                self._phase_counts[tls_id] = 1
                self._program_ids[tls_id] = "0"

            # Force the program so our phase indices apply to the correct logic
            try:
                traci.trafficlight.setProgram(tls_id, self._program_ids[tls_id])
            except Exception:
                # If setProgram fails, we still try to operate on current program
                pass

            # initialize timers
            try:
                cur = int(traci.trafficlight.getPhase(tls_id))
            except Exception:
                cur = 0

            # Clamp cur to valid
            pc = int(self._phase_counts.get(tls_id, 1))
            cur = int(max(0, min(cur, pc - 1)))

            self._last_applied_phase[tls_id] = cur
            self._time_in_phase[tls_id] = 0.0
            self._yellow_remaining_steps[tls_id] = 0
            self._pending_target_phase[tls_id] = None

            # Lock the current phase so SUMO doesn't auto-advance it
            self._lock_phase(tls_id, hold_seconds=999999.0)

        print(f"[INFO] TLS count: {len(self.tls_ids)}")
        print(f"[INFO] Example phase counts (first 10): {[self._phase_counts[t] for t in self.tls_ids[:10]]}")

    def _cleanup_sumo(self) -> None:
        try:
            traci.close()
        except Exception:
            pass
        if self._sumo_proc is not None:
            try:
                self._sumo_proc.terminate()
            except Exception:
                pass
            self._sumo_proc = None

    # --------------------------
    # Phase locking / enforcement
    # --------------------------
    def _lock_phase(self, tls_id: str, hold_seconds: float) -> None:
        """
        Prevent SUMO from auto-advancing an internal tls logic by making the
        current phase duration extremely long (or yellow duration short).
        """
        try:
            traci.trafficlight.setPhaseDuration(tls_id, float(hold_seconds))
        except Exception:
            pass

    # --------------------------
    # Yellow generation
    # --------------------------
    def _compute_yellow_state(self, tls_id: str, from_phase: int, to_phase: int) -> Optional[str]:
        """
        Build a yellow state string from current phase to target phase.
        Any signal that goes from (G/g) -> r becomes 'y' during transition.
        """
        key = (tls_id, from_phase, to_phase)
        if key in self._yellow_state_cache:
            return self._yellow_state_cache[key]

        try:
            defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
            if not defs:
                return None
            phases = getattr(defs[0], "phases", [])
            if not phases:
                return None

            from_phase = int(max(0, min(from_phase, len(phases) - 1)))
            to_phase = int(max(0, min(to_phase, len(phases) - 1)))

            s_from = phases[from_phase].state
            s_to = phases[to_phase].state
            if len(s_from) != len(s_to):
                return None

            y = []
            for a, b in zip(s_from, s_to):
                if (a in ("G", "g")) and (b in ("r", "R")):
                    y.append("y")
                else:
                    y.append(a)
            yellow_state = "".join(y)
            self._yellow_state_cache[key] = yellow_state
            return yellow_state
        except Exception:
            return None

    # --------------------------
    # Step helpers
    # --------------------------
    def _yellow_steps(self) -> int:
        return max(0, int(round(self.yellow_seconds / self.step_length)))

    def _can_switch(self, tls_id: str) -> bool:
        if self._yellow_remaining_steps.get(tls_id, 0) > 0:
            return False
        return self._time_in_phase.get(tls_id, 0.0) >= self.min_green_seconds

    def _sync_if_sumo_overrode(self, tls_id: str) -> None:
        """
        If SUMO changed the phase outside of our apply logic, detect it and sync.
        This is key for diagnosing "applied_by_us stays at 0".
        """
        if self._yellow_remaining_steps.get(tls_id, 0) > 0:
            return

        try:
            actual = int(traci.trafficlight.getPhase(tls_id))
        except Exception:
            return

        last = int(self._last_applied_phase.get(tls_id, actual))
        if actual != last:
            # SUMO overrode us (or advanced internally)
            self._diag_overridden_by_sumo[tls_id] = self._diag_overridden_by_sumo.get(tls_id, 0) + 1
            self._last_applied_phase[tls_id] = actual
            self._time_in_phase[tls_id] = 0.0

            # re-lock so it stops happening repeatedly
            self._lock_phase(tls_id, hold_seconds=999999.0)

    def _apply_action_for_tls(self, tls_id: str, desired_phase: int) -> None:
        phase_count = int(self._phase_counts.get(tls_id, 1))
        phase_count = max(1, phase_count)

        # clamp ALWAYS
        desired_phase = int(max(0, min(desired_phase, phase_count - 1)))

        # If in yellow, decrement and finish transition when due
        if self._yellow_remaining_steps.get(tls_id, 0) > 0:
            self._yellow_remaining_steps[tls_id] -= 1
            if self._yellow_remaining_steps[tls_id] <= 0:
                target = self._pending_target_phase.get(tls_id, None)
                if target is not None:
                    try:
                        traci.trafficlight.setPhase(tls_id, int(target))
                        self._last_applied_phase[tls_id] = int(target)
                        self._time_in_phase[tls_id] = 0.0
                        self._diag_applied_by_us[tls_id] += 1
                        # lock after switching
                        self._lock_phase(tls_id, hold_seconds=999999.0)
                    except Exception:
                        pass
                self._pending_target_phase[tls_id] = None
            return

        # detect SUMO overrides before we decide what to do
        self._sync_if_sumo_overrode(tls_id)

        try:
            current_phase = int(traci.trafficlight.getPhase(tls_id))
        except Exception:
            current_phase = int(self._last_applied_phase.get(tls_id, 0))

        current_phase = int(max(0, min(current_phase, phase_count - 1)))

        if desired_phase != current_phase:
            self._diag_req_switch[tls_id] += 1

        # If no change requested
        if desired_phase == current_phase:
            return

        # min green gating
        if not self._can_switch(tls_id):
            self._diag_min_green_blocks[tls_id] += 1
            return

        # Insert yellow before switching
        y_steps = self._yellow_steps()
        if y_steps > 0:
            yellow_state = self._compute_yellow_state(tls_id, current_phase, desired_phase)
            if yellow_state:
                try:
                    traci.trafficlight.setRedYellowGreenState(tls_id, yellow_state)
                    self._yellow_remaining_steps[tls_id] = y_steps
                    self._pending_target_phase[tls_id] = desired_phase
                    # lock the yellow duration so SUMO respects it
                    self._lock_phase(tls_id, hold_seconds=float(self.yellow_seconds))
                    return
                except Exception:
                    pass

        # Direct switch
        try:
            traci.trafficlight.setPhase(tls_id, desired_phase)
            self._last_applied_phase[tls_id] = desired_phase
            self._time_in_phase[tls_id] = 0.0
            self._diag_applied_by_us[tls_id] += 1
            self._lock_phase(tls_id, hold_seconds=999999.0)
        except Exception:
            pass

    def _step_timers(self) -> None:
        for tls_id in self.tls_ids:
            self._time_in_phase[tls_id] = float(self._time_in_phase.get(tls_id, 0.0) + self.step_length)

    def _print_phase_diagnostics(self) -> None:
        print("\n" + "─" * 28 + " ⚠️  PHASE SWITCH DIAGNOSTICS " + "─" * 29)
        print(f"[diag] step={self.sim_step}  min_green={self.min_green_seconds:.1f}s  yellow={self.yellow_seconds:.1f}s  step_length={self.step_length:.1f}s")

        # show worst offenders by requested switches
        tls_sorted = sorted(self.tls_ids, key=lambda t: self._diag_req_switch.get(t, 0), reverse=True)
        for tls_id in tls_sorted[:6]:
            print(
                f"TLS {tls_id}: "
                f"req_switch={self._diag_req_switch.get(tls_id,0):5d}  "
                f"applied_by_us={self._diag_applied_by_us.get(tls_id,0):5d}  "
                f"minGreenBlocks={self._diag_min_green_blocks.get(tls_id,0):5d}  "
                f"overriddenBySUMO={self._diag_overridden_by_sumo.get(tls_id,0):5d}  "
                f"time_in_phase={self._time_in_phase.get(tls_id,0.0):6.1f}s"
            )

        # reset snapshot counters
        for tls_id in self.tls_ids:
            self._diag_req_switch[tls_id] = 0
            self._diag_applied_by_us[tls_id] = 0
            self._diag_min_green_blocks[tls_id] = 0
            self._diag_overridden_by_sumo[tls_id] = 0

    # --------------------------
    # Gym API
    # --------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # reload route/config to start episode cleanly
        try:
            traci.load(["-c", self.sumo_cfg_path, "--start"])
        except Exception:
            pass

        self.sim_time = 0.0
        self.sim_step = 0

        # re-force program + lock phases at episode start
        for tls_id in self.tls_ids:
            try:
                traci.trafficlight.setProgram(tls_id, self._program_ids.get(tls_id, "0"))
            except Exception:
                pass

            try:
                p = int(traci.trafficlight.getPhase(tls_id))
            except Exception:
                p = 0

            pc = int(self._phase_counts.get(tls_id, 1))
            p = int(max(0, min(p, pc - 1)))

            self._last_applied_phase[tls_id] = p
            self._time_in_phase[tls_id] = 0.0
            self._yellow_remaining_steps[tls_id] = 0
            self._pending_target_phase[tls_id] = None

            self._lock_phase(tls_id, hold_seconds=999999.0)

        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info: Dict[str, Any] = {"tls_ids": list(self.tls_ids)}
        return obs, info

    def step(self, action):
        act = np.array(action, dtype=np.int32).reshape(-1)
        n = len(self.tls_ids)
        if act.shape[0] != n:
            act = np.pad(act, (0, max(0, n - act.shape[0])), mode="constant")[:n]

        # apply per-TLS
        for i, tls_id in enumerate(self.tls_ids):
            self._apply_action_for_tls(tls_id, int(act[i]))

        # advance sim
        try:
            traci.simulationStep()
        except Exception:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {"error": "simulationStep_failed"}

        self.sim_step += 1
        self.sim_time += self.step_length
        self._step_timers()

        # every 500 steps, print diagnostics
        if self.sim_step % 500 == 0:
            self._print_phase_diagnostics()

        # Placeholder obs/reward (keep your existing logic if you have it)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0

        terminated = False
        truncated = bool(self.sim_time >= self.episode_seconds)
        info: Dict[str, Any] = {"sim_time": self.sim_time, "sim_step": self.sim_step}
        return obs, reward, terminated, truncated, info

    def close(self):
        self._cleanup_sumo()
        super().close()
