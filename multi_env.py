# env/multi_env.py
from __future__ import annotations
import os
import time
import random
import numpy as np
import traci
from typing import Dict, Any, List
from gymnasium import Env, spaces

from env.intersection_env import IntersectionEnv
from utils.sumo_utils import build_sumo_cmd


class MultiIntersectionEnv(Env):
    """
    Multi-intersection SUMO environment for centralized PPO.
    Each intersection is controlled by an agent sharing a single PPO policy,
    but each executes its own independent phase selection.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.sumo_binary = cfg["sumo"].get("binary", "sumo")
        self.sumo_cmd = build_sumo_cmd(cfg, self.sumo_binary)
        self.max_steps = int(cfg["sumo"].get("max_steps", 3600))
        self.step_length = float(cfg["sumo"].get("step_length", 1.0))
        self.warmup_steps = int(cfg["sumo"].get("warmup_steps", 0))
        self.verbose = bool(cfg["sumo"].get("verbose", True))

        self.intersections: List[IntersectionEnv] = []
        self._phases_per_tls: List[int] = []
        self._num_tls: int = 1
        self._feat_per_tls: int = 12
        self._step_count = 0

        # Initial dummy spaces for safe VecEnv initialization
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._feat_per_tls,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(1)

        self.last_actions: Dict[str, int] = {}
        self._last_obs: np.ndarray | None = None
        self.traci_conn = None

    # --------------------------------------------------------------------------------
    # SUMO startup logic
    # --------------------------------------------------------------------------------
    def _start_sumo(self):
        """Launch SUMO with dynamic port and skip duplicate remote-port settings."""
        port = random.randint(50000, 60000)

        # Detect if .sumocfg already defines <remote-port>
        has_remote_port = False
        for arg in self.sumo_cmd:
            if arg.endswith(".sumocfg") and os.path.exists(arg):
                try:
                    with open(arg, "r", encoding="utf-8") as f:
                        text = f.read()
                        if "<remote-port" in text:
                            has_remote_port = True
                            break
                except Exception:
                    pass

        # Also check CLI args
        if any("remote-port" in arg for arg in self.sumo_cmd):
            has_remote_port = True

        # Final command construction
        cmd = list(self.sumo_cmd)
        if not has_remote_port:
            cmd += ["--remote-port", str(port)]

        if self.verbose:
            print(f"🌐 Launching SUMO on port {port}...")
            print("Command:", " ".join(cmd))

        traci.start(cmd, label="multi")
        traci_connection = traci.getConnection("multi")
        return traci_connection

    # --------------------------------------------------------------------------------
    # Environment Lifecycle
    # --------------------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        """Reset SUMO simulation and initialize all intersection agents."""
        super().reset(seed=seed)

        # Close any old TraCI sessions
        try:
            traci.close(False)
        except Exception:
            pass

        self.traci_conn = self._start_sumo()
        self._step_count = 0

        tls_ids = traci.trafficlight.getIDList()
        if self.verbose:
            print(f"[INFO] Initialized {len(tls_ids)} intersections.")

        self.intersections.clear()
        self._phases_per_tls.clear()

        # Initialize intersection agents
        for tls_id in tls_ids:
            inter = IntersectionEnv(tls_id, cfg=self.cfg)
            inter.init_from_sumo()
            self.intersections.append(inter)
            self._phases_per_tls.append(inter.num_phases)

        # Update observation/action spaces dynamically
        self._num_tls = len(self.intersections)
        total_obs_dim = self._num_tls * self._feat_per_tls
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(total_obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(self._phases_per_tls)

        # Warm-up phase to stabilize simulation before training
        for _ in range(self.warmup_steps):
            traci.simulationStep()

        obs = self._get_observation()
        self._last_obs = obs
        return obs, {}

    # --------------------------------------------------------------------------------
    def step(self, actions: np.ndarray):
        """Perform one simulation step across all intersections."""
        self._step_count += 1
        if not isinstance(actions, (list, np.ndarray)):
            actions = [actions]

        # Apply per-intersection actions safely
        for i, tls in enumerate(self.intersections):
            try:
                act = int(actions[i]) % tls.num_phases
                tls.apply_action(act)
                self.last_actions[tls.tls_id] = act
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Action failed at {tls.tls_id}: {e}")

        # Step SUMO forward
        traci.simulationStep()

        obs = self._get_observation()
        reward = self._compute_global_reward()
        terminated = self._step_count >= self.max_steps
        truncated = False
        info = {"step": self._step_count, "n_tls": self._num_tls}

        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------------------------
    # Observation & Reward
    # --------------------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        obs_all = []
        for inter in self.intersections:
            obs_all.extend(inter.observe())
        return np.array(obs_all, dtype=np.float32)

    def _compute_global_reward(self) -> float:
        rewards = [inter.compute_reward() for inter in self.intersections]
        return float(np.mean(rewards)) if rewards else 0.0

    # --------------------------------------------------------------------------------
    def close(self):
        """Graceful shutdown of SUMO and TraCI."""
        try:
            traci.close(False)
        except Exception:
            pass
        if self.verbose:
            print("[INFO] SUMO closed cleanly.")
