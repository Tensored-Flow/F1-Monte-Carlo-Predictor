"""Track-level factors for strategy, overtaking difficulty, and chaos."""
from __future__ import annotations

from typing import Dict

TRACK_FACTORS: Dict[str, dict] = {
    # Example values; expand as needed.
    "Bahrain": {
        "overtaking_difficulty": 0.35,  # lower is easier
        "pit_loss_seconds": 22.0,
        "safety_car_prob": 0.35,
        "degradation_factor": 1.1,
    },
    "Monaco": {
        "overtaking_difficulty": 0.95,
        "pit_loss_seconds": 20.0,
        "safety_car_prob": 0.75,
        "degradation_factor": 0.9,
    },
    "Monza": {
        "overtaking_difficulty": 0.25,
        "pit_loss_seconds": 18.0,
        "safety_car_prob": 0.30,
        "degradation_factor": 1.2,
    },
}


def get_track_factors(event_name: str) -> dict:
    """Return factors for a given event name with sensible defaults."""
    base = {
        "overtaking_difficulty": 0.5,
        "pit_loss_seconds": 20.0,
        "safety_car_prob": 0.4,
        "degradation_factor": 1.0,
    }
    return {**base, **TRACK_FACTORS.get(event_name, {})}
