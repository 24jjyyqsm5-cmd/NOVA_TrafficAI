# ======================================================================
# utils/sumo_utils.py — stable SUMO helpers (Windows-friendly)
# - Provides ensure_sumo_binary()
# - Provides build_sumo_cmd() without duplicate/unsafe flags
# ======================================================================

import os
import shutil
from typing import Any, Dict, List, Optional


def ensure_sumo_binary(gui: bool = False) -> str:
    """
    Ensure a valid SUMO binary path exists and return it.
    Uses the GUI version if requested, else the CLI binary.
    """
    candidates: List[Optional[str]] = []

    # 1) Respect SUMO_HOME if set
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        bin_dir = os.path.join(sumo_home, "bin")
        candidates.extend(
            [
                os.path.join(bin_dir, "sumo-gui.exe"),
                os.path.join(bin_dir, "sumo.exe"),
            ]
        )

    # 2) Common Windows install locations
    candidates.extend(
        [
            r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe",
            r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe",
            r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe",
            r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe",
        ]
    )

    # 3) PATH fallbacks
    candidates.append(shutil.which("sumo-gui"))
    candidates.append(shutil.which("sumo"))

    # Choose the first valid match
    for cand in candidates:
        if cand and os.path.exists(cand):
            base = os.path.basename(cand).lower()
            if gui and "gui" in base:
                return cand
            if (not gui) and ("gui" not in base):
                return cand

    raise FileNotFoundError(
        "Could not locate SUMO binary. Install SUMO or set SUMO_HOME.\n"
        "Expected sumo.exe / sumo-gui.exe to be discoverable."
    )


def build_sumo_cmd(cfg: Dict[str, Any], sumo_binary: str, sumo_cfg_path: Optional[str] = None) -> List[str]:
    """
    Build the SUMO command line from config.

    IMPORTANT: We avoid forcing duration-log flags because some .sumocfg files
    already specify equivalents (which causes: 'option already set' errors).
    """
    sumo_section = cfg.get("sumo", {})

    sumo_cfg = sumo_cfg_path or sumo_section.get("config") or sumo_section.get("config_file")
    if not sumo_cfg:
        raise ValueError("[ERROR] Missing SUMO config file path in config.yaml (sumo.config or sumo.config_file).")

    sumo_cfg = os.path.expanduser(os.path.expandvars(str(sumo_cfg)))
    if not os.path.isfile(sumo_cfg):
        raise FileNotFoundError(f"[ERROR] SUMO config file not found: {sumo_cfg}")

    step_length = float(sumo_section.get("step_length", 1.0))
    delay = int(sumo_section.get("delay", 0))
    teleport_time = int(sumo_section.get("teleport_time", -1))

    # Noise control (keep end-of-run summary; just suppress warnings if requested)
    no_warnings = bool(sumo_section.get("no_warnings", True))
    no_step_log = bool(sumo_section.get("no_step_log", True))
    ignore_route_errors = bool(sumo_section.get("ignore_route_errors", True))

    cmd: List[str] = [
        sumo_binary,
        "-c", sumo_cfg,
        "--start",
        f"--step-length={step_length}",
        f"--delay={delay}",
        f"--time-to-teleport={teleport_time}",
    ]

    if no_warnings:
        cmd.append("--no-warnings")

    if no_step_log:
        cmd.append("--no-step-log")

    if ignore_route_errors:
        cmd.append("--ignore-route-errors=true")

    return cmd
