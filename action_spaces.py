# ======================================================================
# action_spaces.py
# ----------------------------------------------------------------------
# Defines per-intersection (per-TLS) action space sizes based on
# the number of signal phases in SUMO.
#
# Used by:
#   - env/multi_env.py (MultiDiscrete action space)
#
# Design:
#   - Each TLS has its own number of discrete actions
#   - Action = phase index [0 .. num_phases-1]
# ======================================================================

from __future__ import annotations

from typing import Dict, List

import traci


def build_per_tls_action_space(tls_ids: List[str]) -> Dict[str, int]:
    """
    Returns a dict:
        tls_id -> number of phases (action space size)

    This guarantees each intersection has a unique discrete action space
    while still allowing a shared PPO policy.
    """
    per_tls_n: Dict[str, int] = {}

    for tls_id in tls_ids:
        try:
            defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
            if not defs or not defs[0].phases:
                per_tls_n[tls_id] = 1
            else:
                per_tls_n[tls_id] = int(len(defs[0].phases))
        except Exception:
            # Safe fallback: 1 phase (no-op controller)
            per_tls_n[tls_id] = 1

    return per_tls_n
