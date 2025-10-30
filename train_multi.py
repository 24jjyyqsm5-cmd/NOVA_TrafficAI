# train_multi.py
from __future__ import annotations
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter

from utils.config import load_config
from env.multi_env import MultiIntersectionEnv


# ------------------------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """Ensure reproducibility across torch, numpy, and random."""
    import random, torch

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------------------------------
# Main training routine
# ------------------------------------------------------------------------------------
def main():
    print("🚦 Initializing Multi-Agent PPO Traffic Trainer")

    # ---------------------------------------------------------------
    # Step 1. Load configuration
    # ---------------------------------------------------------------
    cfg = load_config()
    print(f"[INFO] Loaded configuration from {cfg['_config_path']}")
    print(f"[DEBUG] Full config loaded:\n{cfg}")

    # Set random seed
    seed = int(cfg["training"].get("seed", 42))
    set_seed(seed)

    # ---------------------------------------------------------------
    # Step 2. Build SUMO Environment
    # ---------------------------------------------------------------
    print("🌐 Building SUMO environment...")

    def make_env():
        return MultiIntersectionEnv(cfg)

    env = DummyVecEnv([make_env])
    print("[INFO] Environment successfully created.")

    # ---------------------------------------------------------------
    # Step 3. PPO Agent Setup
    # ---------------------------------------------------------------
    log_dir = cfg["logging"].get("log_dir", "./runs")
    ensure_dir(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    ppo_params = cfg["ppo"]
    model = PPO(
        policy=ppo_params.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=ppo_params.get("learning_rate", 3e-4),
        n_steps=ppo_params.get("n_steps", 2048),
        batch_size=ppo_params.get("batch_size", 64),
        n_epochs=ppo_params.get("n_epochs", 10),
        gamma=ppo_params.get("gamma", 0.99),
        gae_lambda=ppo_params.get("gae_lambda", 0.95),
        clip_range=ppo_params.get("clip_range", 0.2),
        ent_coef=ppo_params.get("ent_coef", 0.01),
        vf_coef=ppo_params.get("vf_coef", 0.5),
        max_grad_norm=ppo_params.get("max_grad_norm", 0.5),
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
    )

    # ---------------------------------------------------------------
    # Step 4. Continuous Training Loop
    # ---------------------------------------------------------------
    episodes = int(cfg["training"].get("episodes", 100))
    steps_per_episode = int(cfg["training"].get("steps_per_episode", 3600))
    save_interval = int(cfg["training"].get("save_interval", 1))

    print("\n──────────────────────────────────────── 🚦 Step 4: Continuous Training ─────────────────────────────────────────\n")

    for ep in range(1, episodes + 1):
        print(f"🚦 Episode {ep}/{episodes}")

        obs, _ = env.reset()
        total_reward = 0.0

        for step in range(steps_per_episode):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)

            if done.any():
                break

            if step % cfg["logging"].get("print_freq", 50) == 0:
                print(f"[STEP {step:04d}] MeanReward={np.mean(reward):.4f} | WallTime={time.time() % 60:.2f}s")

        avg_reward = total_reward / steps_per_episode
        writer.add_scalar("Episode/MeanReward", avg_reward, ep)

        # -----------------------------------------------------------
        # Save model
        # -----------------------------------------------------------
        if ep % save_interval == 0:
            save_path = os.path.join(log_dir, f"ppo_multi_tls_ep{ep}.zip")
            model.save(save_path)
            print(f"[INFO] Model saved at {save_path}")

    # ---------------------------------------------------------------
    # Wrap up
    # ---------------------------------------------------------------
    writer.close()
    env.close()
    print("[INFO] SUMO closed cleanly.")
    print("✅ Training complete.")


# ------------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupt signal received, exiting cleanly...")
        try:
            from traci import close as traci_close
            traci_close(False)
        except Exception:
            pass
        exit(0)
