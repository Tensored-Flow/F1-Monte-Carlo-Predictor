"""Monte Carlo simulation for race finishing positions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    probabilities: pd.DataFrame
    expected_finish: pd.Series
    podium_prob: pd.Series
    top10_prob: pd.Series
    head_to_head: pd.DataFrame


def _head_to_head_from_orders(orders: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Compute pairwise win probabilities from simulation orders."""
    n_sims, n_drivers = orders.shape
    wins = np.zeros((n_drivers, n_drivers), dtype=int)
    for sim in range(n_sims):
        order = orders[sim]
        # For each driver, increment wins against those finishing behind
        for rank, drv_idx in enumerate(order):
            wins[drv_idx, order[rank + 1 :]] += 1
    probs = wins / float(n_sims)
    return pd.DataFrame(probs, index=labels, columns=labels)


def simulate_probability_table(
    mu: pd.Series | np.ndarray,
    sigma: pd.Series | np.ndarray,
    n_sims: int = 10_000,
    driver_labels: list[str] | None = None,
    random_state: int | None = None,
) -> SimulationResult:
    """
    Run simulations and return position probabilities and summary metrics.

    Args:
        mu: mean performance per driver (larger = faster).
        sigma: volatility per driver (same shape as mu).
        n_sims: number of Monte Carlo draws.
        driver_labels: optional labels for DataFrame index.
        random_state: seed for reproducibility.
    """
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    if mu_arr.shape != sigma_arr.shape:
        raise ValueError("mu and sigma must have the same shape")

    n = mu_arr.shape[0]
    rng = np.random.default_rng(random_state)
    counts = np.zeros((n, n), dtype=int)  # driver x position counts
    orders = np.empty((n_sims, n), dtype=int)

    for sim in range(n_sims):
        perf = rng.normal(mu_arr, sigma_arr)
        order = np.argsort(-perf)  # best performance first
        finish_pos = np.empty(n, dtype=int)
        finish_pos[order] = np.arange(1, n + 1)
        counts[np.arange(n), finish_pos - 1] += 1
        orders[sim] = order

    probs = counts / float(n_sims)
    labels = driver_labels if driver_labels is not None else list(range(n))
    columns = [f"P{i}" for i in range(1, n + 1)]
    prob_df = pd.DataFrame(probs, index=labels, columns=columns)

    expected_finish = (prob_df.values * np.arange(1, n + 1)).sum(axis=1)
    podium_prob = prob_df[["P1", "P2", "P3"]].sum(axis=1)
    top10_prob = prob_df[[f"P{i}" for i in range(1, min(10, n) + 1)]].sum(axis=1)

    head_to_head = _head_to_head_from_orders(orders, labels)

    return SimulationResult(
        probabilities=prob_df,
        expected_finish=pd.Series(expected_finish, index=labels, name="ExpectedFinish"),
        podium_prob=pd.Series(podium_prob, index=labels, name="PodiumProb"),
        top10_prob=pd.Series(top10_prob, index=labels, name="Top10Prob"),
        head_to_head=head_to_head,
    )
