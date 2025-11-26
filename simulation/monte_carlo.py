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
    chaos_scale: float = 0.25,
    base_dnf_prob: float | np.ndarray = 0.05,
    dirichlet_smoothing: float = 0.2,
    winner_floor: float = 0.0,
) -> SimulationResult:
    """
    Run simulations and return position probabilities and summary metrics.

    Args:
        mu: mean performance per driver (larger = faster).
        sigma: volatility per driver (same shape as mu).
        n_sims: number of Monte Carlo draws.
        driver_labels: optional labels for DataFrame index.
        random_state: seed for reproducibility.
        chaos_scale: magnitude of lognormal chaos multiplier on sigma (adds heavier tails).
        base_dnf_prob: baseline per-driver probability of a DNF; scalar or array-like.
        dirichlet_smoothing: additive smoothing applied to position counts to avoid zero tails.
        winner_floor: minimum P1 probability per driver after smoothing (helps avoid overconfident zeros).
    """
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    if mu_arr.shape != sigma_arr.shape:
        raise ValueError("mu and sigma must have the same shape")

    n = mu_arr.shape[0]
    # Vectorize DNF probability; clip to keep the sim stable.
    dnf_arr = np.asarray(base_dnf_prob, dtype=float)
    if dnf_arr.shape == ():
        dnf_arr = np.full(n, float(dnf_arr))
    if dnf_arr.shape != (n,):
        raise ValueError("base_dnf_prob must be scalar or same shape as mu")
    dnf_arr = np.clip(dnf_arr, 0.0, 0.75)

    rng = np.random.default_rng(random_state)
    counts = np.zeros((n, n), dtype=int)  # driver x position counts
    orders = np.empty((n_sims, n), dtype=int)

    for sim in range(n_sims):
        # Heavy-tailed chaos on sigma to widen tails and avoid overconfidence.
        chaos = rng.lognormal(mean=0.0, sigma=chaos_scale)
        perf = rng.normal(mu_arr, sigma_arr * chaos)

        # Retirements push drivers to the back; keep at least one finisher.
        retire = rng.random(n) < dnf_arr
        if retire.all():
            retire[rng.integers(0, n)] = False
        perf = np.where(retire, -np.inf, perf)

        order = np.argsort(-perf)  # best performance first
        finish_pos = np.empty(n, dtype=int)
        finish_pos[order] = np.arange(1, n + 1)
        counts[np.arange(n), finish_pos - 1] += 1
        orders[sim] = order

    # Add light smoothing to avoid zero-probability tails.
    probs = (counts + dirichlet_smoothing) / float(n_sims + dirichlet_smoothing * n)

    # Apply a winner probability floor, then renormalize rows that were boosted.
    winner_floor = float(winner_floor)
    if winner_floor > 0:
        needs_boost = probs[:, 0] < winner_floor
        for idx in np.where(needs_boost)[0]:
            boost = winner_floor - probs[idx, 0]
            probs[idx, 0] = winner_floor
            other = probs[idx, 1:]
            remaining = other.sum()
            if remaining <= 0:
                other[:] = (1 - winner_floor) / max(len(other), 1)
            else:
                scale = max(remaining - boost, 0) / remaining
                other *= scale
            probs[idx, 1:] = other
            row_sum = probs[idx].sum()
            if row_sum > 0:
                probs[idx] /= row_sum
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
