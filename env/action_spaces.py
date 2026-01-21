from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import warnings

try:
    import traci  # type: ignore
    TRACI_AVAILABLE = True
except Exception:
    TRACI_AVAILABLE = False


@dataclass
class TLSPhaseCache:
    """
    Cache for TLS -> number of phases in the currently-active program logic.
    """
    phase_count: Dict[str, int]

    def max_index(self, tls_id: str) -> int:
        n = int(self.phase_count.get(tls_id, 1))
        return max(0, n - 1)

    def count(self, tls_id: str) -> int:
        return int(self.phase_count.get(tls_id, 1))


def _ensure_traci() -> None:
    if not TRACI_AVAILABLE:
        raise RuntimeError("TraCI not available; start SUMO first.")
    # If not connected, any domain call will throw "Not connected."
    try:
        _ = traci.isLoaded()
    except Exception:
        # Don't hard-fail here; the next calls will raise a clearer error.
        pass


def _get_active_program_phases(tls_id: str) -> List:
    """
    Return the phases for the active TLS program.

    We prefer getCompleteRedYellowGreenDefinition (newer API).
    This returns a list of program logics; we try to pick the active program if possible,
    otherwise fall back to index 0.
    """
    defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
    if not defs:
        return []

    # Try to select the active program id
    active_prog: Optional[str] = None
    try:
        active_prog = traci.trafficlight.getProgram(tls_id)
    except Exception:
        active_prog = None

    if active_prog is not None:
        for logic in defs:
            # logic.programID exists in SUMO python objects
            pid = getattr(logic, "programID", None)
            if pid == active_prog:
                return list(getattr(logic, "phases", []) or [])

    # Fallback: first logic
    return list(getattr(defs[0], "phases", []) or [])


def build_per_tls_action_space(tls_ids: List[str]) -> TLSPhaseCache:
    """
    Build and return a cached phase-count map for each TLS.

    NOTE:
      If a TLS reports <= 0 phases, we force it to 1 (frozen).
      If a TLS reports 1 phase, it is effectively "not controllable" via setPhase.
    """
    _ensure_traci()

    phase_count: Dict[str, int] = {}
    for tls in tls_ids:
        try:
            phases = _get_active_program_phases(tls)
            n = len(phases)
        except Exception as e:
            warnings.warn(f"[action_spaces] Failed to read phases for {tls}: {e}. Defaulting to 1.")
            n = 1

        if n <= 0:
            warnings.warn(f"[action_spaces] No phases for {tls}, defaulting to 1.")
            n = 1

        phase_count[tls] = int(n)

    return TLSPhaseCache(phase_count=phase_count)


def filter_controllable_tls_ids(tls_ids: List[str], cache: TLSPhaseCache) -> List[str]:
    """
    Returns only TLS that have 2+ phases (controllable).
    This is the simplest way to eliminate the TraCI error spam.
    """
    kept: List[str] = []
    dropped: List[str] = []
    for tls in tls_ids:
        if cache.count(tls) >= 2:
            kept.append(tls)
        else:
            dropped.append(tls)

    if dropped:
        warnings.warn(
            f"[action_spaces] Dropping {len(dropped)} TLS with <2 phases (not controllable). "
            f"Examples: {dropped[:8]}"
        )
    return kept


def clamp_action_to_valid_phase(tls_id: str, action: int, cache: Optional[TLSPhaseCache] = None) -> int:
    """
    Clamp action to valid phase index for this TLS.

    IMPORTANT:
      Prefer passing the cache from build_per_tls_action_space() so clamp is stable and
      doesn't re-query TraCI every step.
    """
    if cache is not None:
        max_idx = cache.max_index(tls_id)
    else:
        # Safe fallback (slower + less stable than cache)
        phases = _get_active_program_phases(tls_id)
        max_idx = max(0, len(phases) - 1)

    if action < 0 or action > max_idx:
        return min(max(int(action), 0), max_idx)
    return int(action)
