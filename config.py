# utils/config.py
import os
import yaml


_DEFAULT = {
    "sumo": {
        "binary": "sumo",
        "config": "",            # REQUIRED: absolute path to .sumocfg
        "step_length": 1.0,
        "max_steps": 3600,
        "warmup_steps": 50,
        "delay": 0,
        "no_warnings": True,
        "ignore_route_errors": True,
        "teleport_time": -1,
        "verbose": True,
    },
    "training": {
        "seed": 42,
        "episodes": 10,
        "steps_per_episode": 3600,
        "num_envs": 1,
        "save_interval": 1,
        "eval_interval": 5,
    },
    "ppo": {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.02,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
    },
    "observation": {
        "feature_dim": 12,
        "normalize": True,
    },
    "logging": {
        "log_dir": "./runs",
        "print_freq": 50,
        "save_freq": 1,
        "tensorboard": True,
    },
    "misc": {
        "debug": False,
        "save_gif": False,
    }
}


def load_config(path: str = "config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found at '{path}'. "
            "Create config.yaml with a 'sumo: {config: <abs path to osm.sumocfg>}' section."
        )
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    # Normalize historical key 'config_file' -> 'config'
    if "sumo" in user:
        if "config_file" in user["sumo"] and "config" not in user["sumo"]:
            user["sumo"]["config"] = user["sumo"]["config_file"]
    cfg = _DEFAULT.copy()
    # deep-merge
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                deep_update(d[k], v)
            else:
                d[k] = v
    deep_update(cfg, user)

    # Validation
    sumo_cfg = cfg["sumo"].get("config", "").strip()
    if not sumo_cfg or not os.path.exists(sumo_cfg):
        print("\n[ERROR] SUMO configuration not found.")
        print("Add this to your config.yaml (adjust the path to your file):\n")
        print("sumo:")
        print("  config: C:/Users/…/osm.sumocfg\n")
        raise FileNotFoundError("SUMO config missing or invalid.")

    cfg["_config_path"] = os.path.abspath(path)
    return cfg
