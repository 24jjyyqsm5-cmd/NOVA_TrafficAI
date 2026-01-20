# ======================================================================
# train_multi.py — Shared PPO Training (All TLS)
# ======================================================================

import os
import time
from typing import Any, Dict, List

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.config import load_config
from env.multi_env import MultiIntersectionEnv


def _get_cfg(cfg: Dict[str, Any], keys: List[List[str]], default: Any) -> Any:
    for path in keys:
        cur: Any = cfg
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok and cur is not None:
            return cur
    return default


def _get_cfg_int(cfg: Dict[str, Any], keys: List[List[str]], default: int) -> int:
    v = _get_cfg(cfg, keys, default)
    try:
        return int(v)
    except Exception:
        return default


def _get_cfg_float(cfg: Dict[str, Any], keys: List[List[str]], default: float) -> float:
    v = _get_cfg(cfg, keys, default)
    try:
        return float(v)
    except Exception:
        return default


def _get_cfg_str(cfg: Dict[str, Any], keys: List[List[str]], default: str) -> str:
    v = _get_cfg(cfg, keys, default)
    try:
        return str(v)
    except Exception:
        return default


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def main():
    cfg = load_config()

    log_dir = _ensure_dir(_get_cfg_str(cfg, [["logging", "log_dir"], ["training", "log_dir"]], "./runs"))
    save_dir = _ensure_dir(
        _get_cfg_str(cfg, [["logging", "model_dir"], ["training", "save_dir"], ["training", "model_dir"]], "./models")
    )

    total_timesteps = _get_cfg_int(cfg, [["training", "total_timesteps"]], 200_000)
    save_freq = _get_cfg_int(cfg, [["training", "save_freq"]], 50_000)

    learning_rate = _get_cfg_float(cfg, [["training", "learning_rate"]], 3e-4)
    n_steps = _get_cfg_int(cfg, [["training", "n_steps"]], 2048)
    batch_size = _get_cfg_int(cfg, [["training", "batch_size"]], 256)
    gamma = _get_cfg_float(cfg, [["training", "gamma"]], 0.99)
    gae_lambda = _get_cfg_float(cfg, [["training", "gae_lambda"]], 0.95)
    clip_range = _get_cfg_float(cfg, [["training", "clip_range"]], 0.2)
    ent_coef = _get_cfg_float(cfg, [["training", "ent_coef"]], 0.0)
    vf_coef = _get_cfg_float(cfg, [["training", "vf_coef"]], 0.5)
    max_grad_norm = _get_cfg_float(cfg, [["training", "max_grad_norm"]], 0.5)
    verbose = _get_cfg_int(cfg, [["logging", "verbose"], ["training", "verbose"]], 1)

    run_name = _get_cfg_str(cfg, [["training", "run_name"]], f"ppo_all_tls_{int(time.time())}")

    print("=" * 110)
    print("Shared PPO Training (All TLS)")
    print("=" * 110)
    print(f"[INFO] run_name        : {run_name}")
    print(f"[INFO] log_dir         : {log_dir}")
    print(f"[INFO] save_dir        : {save_dir}")
    print(f"[INFO] total_timesteps : {total_timesteps}")
    print(f"[INFO] save_freq       : {save_freq}")
    print("=" * 110)

    base_env = MultiIntersectionEnv(cfg)

    obs_dim = int(base_env.observation_space.shape[0])
    print(f"[INFO] env obs_dim      : {obs_dim}")

    # MultiDiscrete per TLS (should be per-TLS nvec!)
    if hasattr(base_env.action_space, "nvec"):
        n_tls = int(len(base_env.action_space.nvec))
        a_max = int(max(base_env.action_space.nvec))
        print(f"[INFO] env tls_count    : {n_tls}")
        print(f"[INFO] env A_max        : {a_max} (per-TLS MultiDiscrete nvec)")
    print("=" * 110)

    env = DummyVecEnv([lambda: base_env])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=verbose,
        tensorboard_log=log_dir,
    )

    try:
        steps_done = 0
        while steps_done < total_timesteps:
            steps_left = total_timesteps - steps_done
            chunk = min(save_freq, steps_left)

            print(f"[INFO] Learning chunk: {chunk} steps (progress {steps_done}/{total_timesteps})")
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                tb_log_name=run_name,
                progress_bar=False,
            )

            steps_done += chunk
            ckpt_path = os.path.join(save_dir, f"{run_name}_steps_{steps_done}")
            model.save(ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

        final_path = os.path.join(save_dir, f"{run_name}_final")
        model.save(final_path)
        print(f"[OK] Training complete. Final model saved: {final_path}")

    except KeyboardInterrupt:
        print("\n[WARN] KeyboardInterrupt received. Saving interrupt checkpoint...")
        interrupt_path = os.path.join(save_dir, f"{run_name}_interrupt")
        try:
            model.save(interrupt_path)
            print(f"[INFO] Saved interrupt checkpoint: {interrupt_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save interrupt checkpoint: {e}")

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
