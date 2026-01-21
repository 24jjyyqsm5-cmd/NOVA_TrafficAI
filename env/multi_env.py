# ======================================================================
# env/multi_env.py
# ----------------------------------------------------------------------
# MultiIntersectionEnv (NOVA 2.0)
# - One env controls N TLS (one action per TLS)
# - Per-TLS MultiDiscrete action space (each TLS has its own phase count)
# - Safety:
#     - per-TLS min green gating
#     - clamp phase index to [0, phase_count-1]
#     - explicit yellow insertion using setRedYellowGreenState()
#
# Diagnostics:
#   req_switch      = count only when desired_phase != current_phase (true switch attempt)
#   applied_by_us   = count only when we actually apply a switch
#   minGreenBlocks  = count only when a true switch attempt is blocked by min green
#   phaseSetErrors  = setPhase failures from TraCI (range errors, etc.)
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

# If your project still uses detector_mapping.py naming:
# (Your earlier error was import mismatch; keep it aligned with your actual file.)
try:
    from utils.detector_mapping import load_detectors_from_sumocfg as build_detector_lane_mapping
except Exception:
    # fallback name if you already created build_detector_lane_mapping
    from utils.detector_mapping import build_detector_lane_mapping  # type: ignore


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

        # IMPORTANT: episode duration comes from config.
        # Set training.episode_seconds = 5400 to run full 5400s horizon.
        self.episode_seconds = float(self.cfg.get("training", {}).get("episode_seconds", 900))

        # Safety params
        self.min_green_seconds = float(self.cfg.get("phase", {}).get("min_green_seconds", 10))
        self.yellow_seconds = float(self.cfg.get("phase", {}).get("yellow_seconds", 3))

        # Reward weights (keep your existing reward logic if you already have one)
        self.alpha = float(self.cfg.get("reward", {}).get("alpha", 0.6))
        self.queue_w = float(self.cfg.get("reward", {}).get("queue_w", 1.0))
        self.wait_w = float(self.cfg.get("reward", {}).get("wait_w", 0.02))

        self.max_tls = int(self.cfg.get("training", {}).get("max_tls", 24))
        self.tls_ids: List[str] = []

        # runtime
        self._sumo_proc: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None

        # per-TLS phase metadata
        self._phase_counts: Dict[str, int] = {}
        self._time_in_phase: Dict[str, float] = {}

        # yellow transition tracking
        self._yellow_remaining_steps: Dict[str, int] = {}
        self._pending_target_phase: Dict[str, Optional[int]] = {}
        self._yellow_state_cache: Dict[Tuple[str, int, int], str] = {}

        # detectors
        self.det_to_lane: Dict[str, str] = {}

        # diagnostics counters (per episode)
        self._req_switch: Dict[str, int] = {}
        self._applied_by_us: Dict[str, int] = {}
        self._min_green_blocks: Dict[str, int] = {}
        self._phase_set_errors: Dict[str, int] = {}

        # Optional diagnostics toggles
        logging_cfg = self.cfg.get("logging", {})
        self._diag_enabled = bool(logging_cfg.get("phase_switch_diagnostics", True))
        self._diag_every = int(logging_cfg.get("phase_switch_diag_every", 500))

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
        # NOTE: your real obs should be detector-only 24*62=1488.
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

        # Load detectors mapping (if present)
        try:
            self.det_to_lane = build_detector_lane_mapping(self.sumo_cfg_path)
        except Exception as e:
            print(f"[WARN] detector mapping failed: {e}")
            self.det_to_lane = {}

        # Discover TLS
        all_tls = list(traci.trafficlight.getIDList())
        if not all_tls:
            raise RuntimeError("[ERROR] No traffic lights found in the network.")

        # Optional explicit list
        requested = self.cfg.get("training", {}).get("tls_ids", []) or []
        if requested:
            self.tls_ids = [t for t in requested if t in all_tls]
        else:
            self.tls_ids = all_tls[: self.max_tls]

        # Phase counts per TLS
        for tls_id in self.tls_ids:
            self._phase_counts[tls_id] = self._get_phase_count_safe(tls_id)
            self._time_in_phase[tls_id] = 0.0
            self._yellow_remaining_steps[tls_id] = 0
            self._pending_target_phase[tls_id] = None

            # diag counters
            self._req_switch[tls_id] = 0
            self._applied_by_us[tls_id] = 0
            self._min_green_blocks[tls_id] = 0
            self._phase_set_errors[tls_id] = 0

        print(f"[INFO] TLS count: {len(self.tls_ids)}")
        print(f"[INFO] Example phase counts (first 10): {[self._phase_counts[t] for t in self.tls_ids[:10]]}")

    def _get_phase_count_safe(self, tls_id: str) -> int:
        """
        Robustly determine phase count.
        We prefer complete definition phases length; if anything fails -> 1.
        """
        try:
            defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
            if not defs:
                return 1
            logic = defs[0]
            phases = getattr(logic, "phases", None)
            if not phases:
                return 1
            return max(1, int(len(phases)))
        except Exception:
            return 1

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
    # Yellow generation
    # --------------------------
    def _compute_yellow_state(self, tls_id: str, from_phase: int, to_phase: int) -> Optional[str]:
        """
        Build a yellow state string from current phase to target phase.
        Any signal that goes from (G/g) -> r becomes 'y' during transition.

        NOTE: SUMO phase "state" strings are like 'GrGr...' etc.
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
                    # keep "from" signal otherwise
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

    def _in_yellow(self, tls_id: str) -> bool:
        return self._yellow_remaining_steps.get(tls_id, 0) > 0

    def _can_switch(self, tls_id: str) -> bool:
        # can't switch while in yellow transition
        if self._in_yellow(tls_id):
            return False
        return self._time_in_phase.get(tls_id, 0.0) >= self.min_green_seconds

    def _apply_action_for_tls(self, tls_id: str, desired_phase_raw: int) -> None:
        """
        Apply one TLS action with:
          - clamp
          - true switch attempt counting (desired != current)
          - min green gating
          - explicit yellow insertion
        """
        phase_count = int(self._phase_counts.get(tls_id, 1))
        if phase_count <= 0:
            phase_count = 1

        # clamp ALWAYS
        desired_phase = int(max(0, min(int(desired_phase_raw), phase_count - 1)))

        # If currently in yellow, countdown and complete transition when due.
        if self._in_yellow(tls_id):
            self._yellow_remaining_steps[tls_id] -= 1
            if self._yellow_remaining_steps[tls_id] <= 0:
                target = self._pending_target_phase.get(tls_id, None)
                self._pending_target_phase[tls_id] = None
                if target is not None:
                    try:
                        traci.trafficlight.setPhase(tls_id, int(target))
                        self._applied_by_us[tls_id] += 1
                        self._time_in_phase[tls_id] = 0.0
                    except Exception:
                        self._phase_set_errors[tls_id] += 1
            return

        # Read current phase
        try:
            current_phase = int(traci.trafficlight.getPhase(tls_id))
        except Exception:
            current_phase = 0

        # TRUE "switch attempt": only if desired != current
        if desired_phase != current_phase:
            self._req_switch[tls_id] += 1

            # min green gating blocks true switch attempts
            if not self._can_switch(tls_id):
                self._min_green_blocks[tls_id] += 1
                return

            # Insert yellow before switching (if enabled)
            y_steps = self._yellow_steps()
            if y_steps > 0:
                yellow_state = self._compute_yellow_state(tls_id, current_phase, desired_phase)
                if yellow_state:
                    try:
                        traci.trafficlight.setRedYellowGreenState(tls_id, yellow_state)
                        self._yellow_remaining_steps[tls_id] = y_steps
                        self._pending_target_phase[tls_id] = desired_phase
                        return
                    except Exception:
                        # fall through to direct setPhase
                        pass

            # Direct switch
            try:
                traci.trafficlight.setPhase(tls_id, desired_phase)
                self._applied_by_us[tls_id] += 1
                self._time_in_phase[tls_id] = 0.0
            except Exception:
                self._phase_set_errors[tls_id] += 1

        # If desired == current: do nothing (and do NOT count req_switch)

    def _step_timers(self) -> None:
        for tls_id in self.tls_ids:
            self._time_in_phase[tls_id] = float(self._time_in_phase.get(tls_id, 0.0) + self.step_length)

    def _maybe_print_diag(self) -> None:
        if not self._diag_enabled:
            return
        if self.sim_step <= 0:
            return
        if (self.sim_step % self._diag_every) != 0:
            return

        # show top few by errors/blocks
        rows = []
        for tls_id in self.tls_ids:
            rows.append(
                (
                    tls_id,
                    self._req_switch.get(tls_id, 0),
                    self._applied_by_us.get(tls_id, 0),
                    self._min_green_blocks.get(tls_id, 0),
                    self._phase_set_errors.get(tls_id, 0),
                    self._time_in_phase.get(tls_id, 0.0),
                )
            )

        # sort by phase errors then blocks then req_switch
        rows.sort(key=lambda x: (x[4], x[3], x[1]), reverse=True)

        print("\n──────────────────────────── ⚠️  PHASE SWITCH DIAGNOSTICS ────────────────────────────")
        print(f"[diag] step={self.sim_step}  min_green={self.min_green_seconds:.1f}s  yellow={self.yellow_seconds:.1f}s  step_length={self.step_length:.1f}s")
        for tls_id, req, app, blk, err, tip in rows[:6]:
            print(
                f"TLS {tls_id}: req_switch={req:5d}  applied_by_us={app:5d}  "
                f"minGreenBlocks={blk:4d}  phaseSetErrors={err:4d}  time_in_phase={tip:.1f}s"
            )

    # --------------------------
    # Gym API
    # --------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Reset episode counters
        self.sim_time = 0.0
        self.sim_step = 0

        # Reset diagnostics per episode
        for tls_id in self.tls_ids:
            self._req_switch[tls_id] = 0
            self._applied_by_us[tls_id] = 0
            self._min_green_blocks[tls_id] = 0
            self._phase_set_errors[tls_id] = 0
            self._time_in_phase[tls_id] = 0.0
            self._yellow_remaining_steps[tls_id] = 0
            self._pending_target_phase[tls_id] = None

        # Reload scenario for a clean episode (your “first sim fails” behavior is fine)
        try:
            traci.load(["-c", self.sumo_cfg_path, "--start"])
        except Exception:
            pass

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
        self._maybe_print_diag()

        # Placeholder obs/reward (replace with your detector-only state + reward logic)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0

        terminated = False
        truncated = bool(self.sim_time >= self.episode_seconds)
        info: Dict[str, Any] = {
            "sim_time": self.sim_time,
            "sim_step": self.sim_step,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self._cleanup_sumo()
        super().close()
