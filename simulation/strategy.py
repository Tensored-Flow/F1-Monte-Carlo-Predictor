"""Lightweight strategy and race-chaos adjustments."""
from __future__ import annotations

import numpy as np
import pandas as pd


def adjust_mu_for_strategy(mu: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    """
    Adjust mu for strategy factors: pit loss and overtaking difficulty.

    Heuristic: higher overtaking difficulty penalizes slower cars more;
    pit loss raises the cost of extra stops; degradation factor scales pace uncertainty.
    """
    overtaking = features.get("track_overtaking_difficulty", pd.Series(0.5, index=features.index))
    pit_loss = features.get("track_pit_loss_seconds", pd.Series(20.0, index=features.index))
    degradation = features.get("track_degradation_factor", pd.Series(1.0, index=features.index))

    # Normalize features
    over_penalty = (overtaking - overtaking.mean()) / (overtaking.std() + 1e-6)
    pit_penalty = (pit_loss - pit_loss.mean()) / (pit_loss.std() + 1e-6)
    deg_scale = degradation / (degradation.mean() + 1e-6)

    adjusted = mu - 0.05 * over_penalty.to_numpy() - 0.02 * pit_penalty.to_numpy()
    adjusted = adjusted / deg_scale.to_numpy()
    return adjusted


def adjust_sigma_for_chaos(sigma: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    """
    Inflate sigma based on safety car probability and track chaos.
    """
    safety = features.get("track_safety_car_prob", pd.Series(0.4, index=features.index))
    chaos = features.get("track_chaos", pd.Series(0, index=features.index)).astype(float)
    chaos_norm = (chaos - chaos.mean()) / (chaos.std() + 1e-6)
    sigma_adj = sigma * (1 + 0.3 * safety.to_numpy()) * (1 + 0.05 * chaos_norm.to_numpy())
    return sigma_adj
