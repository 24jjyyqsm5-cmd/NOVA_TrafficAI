# ======================================================================
# evaluate_multi.py — Evaluate Shared PPO (All TLS) + Detector-Only Observations
# ======================================================================

import os
import time
from typing import Any, Dict, Optional

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.config import load_config
from env.multi_env import MultiIntersectionEnv


def _get_cfg_int(cfg: Dict[str, Any], keys, default: int) -> int:
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return int(cur)
    except Exception:
        return default


def _get_cfg_str(cfg: Dict[str, Any], keys, default: str) -> str:
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return str(cur) if cur is not None else default


def make_env(cfg: Dict[str, Any]):
    return MultiIntersectionEnv(cfg)


def main():
    cfg = load_config()

    # Model path: either config override, or prompt default
    default_model = _get_cfg_str(cfg, ["evaluation", "model_path"], "./models/latest_final")
    model_path = default_model

    # Evaluation steps
    eval_steps = _get_cfg_int(cfg, ["evaluation", "steps"], 5000)
    deterministic = bool(cfg.get("evaluation", {}).get("deterministic", True))

    print("────────────────────────────────────────────── 🔍 Evaluation (All TLS) ───────────────────────────────────────────────")
    print(f"[INFO] model_path     : {model_path}")
    print(f"[INFO] eval_steps     : {eval_steps}")
    print(f"[INFO] deterministic  : {deterministic}")
    print("──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────")

    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print("[WARN] Model path not found. SB3 expects a .zip. Example:")
        print("       ./models/ppo_all_tls_xxxxx_final")
        print("       (SB3 will load ./models/ppo_all_tls_xxxxx_final.zip)")
        return

    env = DummyVecEnv([lambda: make_env(cfg)])

    # Load model
    model = PPO.load(model_path, env=env)

    # Rollout
    obs = env.reset()
    ep_reward = 0.0

    # Track some aggregate info
    info_last = {}

    for t in range(eval_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)

        r = float(reward[0]) if hasattr(reward, "__len__") else float(reward)
        ep_reward += r

        # info in DummyVecEnv is list[dict]
        if isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict):
            info_last = info[0]

        if done is True or (hasattr(done, "__len__") and bool(done[0])):
            print(f"[INFO] Episode ended at step {t}. Episode reward: {ep_reward:.4f}")
            obs = env.reset()
            ep_reward = 0.0

        # light periodic prints
        if (t + 1) % 500 == 0:
            alpha = info_last.get("alpha_sim", None)
            pressure = info_last.get("pressure_total", None)
            phase_changes = info_last.get("phase_changes", None)
            print(
                f"[STEP {t+1:6d}] r_sum={ep_reward: .4f}"
                + (f" | alpha_sim={alpha:.3f}" if isinstance(alpha, (int, float)) else "")
                + (f" | pressure={pressure:.3f}" if isinstance(pressure, (int, float)) else "")
                + (f" | phase_changes={phase_changes}" if phase_changes is not None else "")
            )

    print(f"✅ Evaluation finished. Total reward accumulated (across resets): {ep_reward:.4f}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
