from __future__ import annotations
from typing import Dict, List
import warnings

try:
    import traci
    TRACI_AVAILABLE = True
except Exception:
    TRACI_AVAILABLE = False


def _phases_for_tls(tls_id: str) -> List:
    definitions = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
    if not definitions:
        return []
    return list(definitions[0].phases)


def build_per_tls_action_space(tls_ids: List[str]) -> Dict[str, int]:
    if not TRACI_AVAILABLE:
        raise RuntimeError("TraCI not available; start SUMO first.")

    action_space: Dict[str, int] = {}
    for tls in tls_ids:
        phases = _phases_for_tls(tls)
        n = len(phases)
        if n <= 0:
            warnings.warn(f"No phases for {tls}, defaulting to 1.")
            n = 1
        action_space[tls] = n
    return action_space


def clamp_action_to_valid_phase(tls_id: str, action: int) -> int:
    phases = _phases_for_tls(tls_id)
    max_idx = max(0, len(phases) - 1)
    if action < 0 or action > max_idx:
        warnings.warn(f"Invalid phase {action} for {tls_id}; clamped to [0,{max_idx}].")
        return min(max(action, 0), max_idx)
    return action
