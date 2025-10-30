import os
import shlex

def build_sumo_cmd(cfg, sumo_binary: str):
    s = cfg.get("sumo", {})
    # support both keys: "config" and legacy "config_file"
    cfg_path = s.get("config") or s.get("config_file")
    if not cfg_path:
        raise FileNotFoundError("[SUMO] Please set 'sumo.config' in config.yaml to your *.sumocfg path.")
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.abspath(cfg_path)

    cmd = [
        sumo_binary,
        "-c", cfg_path,
        "--start",
        "--delay", str(int(s.get("delay", 0))),
        "--step-length", str(float(s.get("step_length", 1.0))),
        "--no-warnings", "true" if s.get("no_warnings", True) else "false",
        "--ignore-route-errors", "true" if s.get("ignore_route_errors", True) else "false",
        "--time-to-teleport", str(int(s.get("teleport_time", -1))),
        "--duration-log.disable", "true",
    ]
    # IMPORTANT: do NOT set --remote-port here; traci.start() will pick a free port.
    return cmd
