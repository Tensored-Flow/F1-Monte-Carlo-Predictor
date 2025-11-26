"""Grid-search simulator hyperparameters for better calibration on past races."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import fastf1

from features.feature_engineering import build_features
from models.mu_model import load_model as load_mu_model
from models.mu_model import predict_mu
from models.sigma_model import (
    load_model as load_sigma_model,
    predict_sigma,
    predict_sigma_calibrated,
    heuristic_sigma,
)
from simulation.monte_carlo import simulate_probability_table
from utils.session_loader import load_session


DEFAULT_EVENTS: List[Tuple[int, str]] = [
    (2023, "Bahrain Grand Prix"),
    (2023, "Spanish Grand Prix"),
    (2023, "Monaco Grand Prix"),
    (2024, "Bahrain Grand Prix"),
    (2024, "Japanese Grand Prix"),
    (2024, "Las Vegas Grand Prix"),
    (2024, "Abu Dhabi Grand Prix"),
    (2025, "Mexico City Grand Prix"),
    (2025, "São Paulo Grand Prix"),
    (2025, "Las Vegas Grand Prix"),
]
CONFIG_PATH = Path("config/calibration.json")
RESULTS_PATH = Path("output/calibration_results.csv")


def _align_features(df: pd.DataFrame, ref_model: Any) -> pd.DataFrame:
    if ref_model is None:
        return df
    if hasattr(ref_model, "feature_names_in_"):
        cols = list(ref_model.feature_names_in_)
        aligned = df.copy()
        for col in cols:
            if col not in aligned:
                aligned[col] = 0.0
        return aligned[cols]
    return df


def _load_models() -> Tuple[Any | None, Any | None]:
    mu_model = load_mu_model("models/artifacts/mu.joblib") if Path("models/artifacts/mu.joblib").exists() else None
    sigma_bundle = joblib.load("models/artifacts/sigma.joblib") if Path("models/artifacts/sigma.joblib").exists() else None
    return mu_model, sigma_bundle


def _predict_mu_sigma(feat_df: pd.DataFrame, mu_model: Any | None, sigma_bundle: Any | None) -> Tuple[np.ndarray, np.ndarray]:
    numeric = feat_df.select_dtypes(include=[np.number])
    base_mu = -feat_df["median_lap_s"].to_numpy()
    grid_pos = feat_df.get("grid_position", pd.Series(np.nan, index=feat_df.index)).to_numpy()
    grid_adj = np.where(np.isnan(grid_pos), 0.0, (feat_df.shape[0] + 1 - grid_pos) * 0.004)
    mu_pred = base_mu + grid_adj
    if mu_model is not None:
        try:
            mu_pred = predict_mu(mu_model, _align_features(numeric, mu_model))
        except Exception:
            pass

    sigma_pred = heuristic_sigma(feat_df)
    if sigma_bundle is not None:
        try:
            if isinstance(sigma_bundle, dict):
                sigma_pred = predict_sigma_calibrated(sigma_bundle, _align_features(numeric, sigma_bundle.get("model", sigma_bundle)))
            else:
                sigma_pred = predict_sigma(sigma_bundle, _align_features(numeric, sigma_bundle))
        except Exception:
            pass
    return mu_pred, sigma_pred


def backtest_event(
    year: int,
    event: str,
    sims: int,
    chaos_scale: float,
    dirichlet_smoothing: float,
    dnf_floor: float,
    sigma_multiplier: float,
    winner_floor: float,
    mu_model: Any | None,
    sigma_bundle: Any | None,
) -> Dict[str, Any]:
    """Run a single backtest for given hyperparameters."""
    try:
        q = load_session(year, event, "Q")
        r = load_session(year, event, "R")
    except Exception as e:
        return {"year": year, "event": event, "error": str(e)}

    event_name = getattr(getattr(q.get("session"), "event", None), "EventName", event)
    session_results = q.get("results", pd.DataFrame())

    # Recent historical results (last 6 races) for form/DNF
    hist_df = pd.DataFrame()
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        row = schedule[schedule["EventName"] == str(event)]
        target_round = int(row["RoundNumber"].iloc[0]) if not row.empty else None
        if target_round:
            prev = schedule[schedule["RoundNumber"] < target_round].sort_values("RoundNumber", ascending=False).head(6)
            rows = []
            for _, ev in prev.iterrows():
                try:
                    res_session = load_session(int(ev["EventDate"].year), ev["EventName"], "R")
                    res = res_session.get("results", pd.DataFrame())
                    if res is None or res.empty:
                        continue
                    part = res[["Abbreviation", "Position", "Status"]].copy()
                    part["RacePosition"] = part["Position"]
                    part["QualiPosition"] = np.nan
                    part["DNF"] = ~part["Status"].fillna("").str.contains("Finished", case=False)
                    part["Round"] = int(ev["RoundNumber"])
                    part = part.rename(columns={"Abbreviation": "Driver"})
                    rows.append(part[["Driver", "RacePosition", "QualiPosition", "DNF", "Round"]])
                except Exception:
                    continue
            if rows:
                hist_df = pd.concat(rows, ignore_index=True)
    except Exception:
        hist_df = pd.DataFrame()

    # Grid positions from session results if available
    grid_series = None
    if session_results is not None and not session_results.empty:
        if "GridPosition" in session_results.columns:
            grid_series = session_results.set_index("Abbreviation")["GridPosition"]
        elif "Position" in session_results.columns:
            grid_series = session_results.set_index("Abbreviation")["Position"]

    feat = build_features(
        q["laps"],
        historical_results=hist_df,
        weather=q.get("weather"),
        event_name=event_name,
        grid_positions=grid_series,
    )
    mu_pred, sigma_pred = _predict_mu_sigma(feat, mu_model, sigma_bundle)
    sigma_pred = sigma_pred * sigma_multiplier

    # DNF prior: use feature column if present, otherwise floor.
    if "dnf_rate" in feat.columns:
        dnf_prior = feat["dnf_rate"].fillna(dnf_floor).clip(dnf_floor, 0.75).to_numpy()
    else:
        dnf_prior = np.full(len(feat), dnf_floor)

    sim = simulate_probability_table(
        mu_pred,
        sigma_pred,
        n_sims=sims,
        driver_labels=feat["Driver"].tolist(),
        random_state=42,
        chaos_scale=chaos_scale,
        base_dnf_prob=dnf_prior,
        dirichlet_smoothing=dirichlet_smoothing,
        winner_floor=winner_floor,
    )

    results = r.get("results")
    if results is None or results.empty:
        return {"year": year, "event": event, "error": "No race results"}
    res_df = results[["Abbreviation", "Position"]].dropna().copy()
    res_df["Position"] = res_df["Position"].astype(int)

    prob_df = sim.probabilities.reset_index().rename(columns={"index": "Driver"})
    merged = res_df.merge(prob_df, left_on="Abbreviation", right_on="Driver", how="inner")
    if merged.empty:
        return {"year": year, "event": event, "error": "No driver overlap"}

    expected = sim.expected_finish.loc[merged["Driver"]].to_numpy()
    actual = merged["Position"].to_numpy()
    mae_finish = float(np.mean(np.abs(expected - actual)))

    podium_actual = (actual <= 3).astype(int)
    top10_actual = (actual <= 10).astype(int)
    podium_prob = merged[["P1", "P2", "P3"]].sum(axis=1).to_numpy()
    top10_prob = merged[[f"P{i}" for i in range(1, 11)]].sum(axis=1).to_numpy()
    brier_podium = float(np.mean((podium_prob - podium_actual) ** 2))
    brier_top10 = float(np.mean((top10_prob - top10_actual) ** 2))

    winner_row = merged.loc[merged["Position"] == 1]
    winner_prob = float(winner_row["P1"].iloc[0]) if not winner_row.empty else np.nan

    return {
        "year": year,
        "event": event_name,
        "mae_finish": mae_finish,
        "brier_podium": brier_podium,
        "brier_top10": brier_top10,
        "winner_prob_predicted": winner_prob,
        "winner": winner_row["Driver"].iloc[0] if not winner_row.empty else "N/A",
    }


def objective_metric(records: Iterable[Dict[str, Any]]) -> float:
    """Combine metrics into a single score (lower is better)."""
    recs = [r for r in records if "error" not in r]
    if not recs:
        return float("inf")
    mae = np.mean([r["mae_finish"] for r in recs])
    b_pod = np.mean([r["brier_podium"] for r in recs])
    b_t10 = np.mean([r["brier_top10"] for r in recs])
    # weight Brier more; add small weight for MAE
    return b_pod + b_t10 + 0.05 * mae


def run_calibration(
    events: List[Tuple[int, str]] = DEFAULT_EVENTS,
    sims: int = 3000,
    chaos_grid: Iterable[float] = (0.4, 0.6),
    smoothing_grid: Iterable[float] = (0.15, 0.35),
    dnf_grid: Iterable[float] = (0.1, 0.16),
    sigma_grid: Iterable[float] = (1.0, 1.15),
    winner_floor_grid: Iterable[float] = (0.005, 0.01),
) -> Dict[str, Any]:
    mu_model, sigma_bundle = _load_models()

    results: List[Dict[str, Any]] = []
    best_cfg: Dict[str, Any] | None = None
    best_score = float("inf")

    for chaos_scale, smoothing, dnf_floor, sigma_mult, winner_floor in itertools.product(
        chaos_grid, smoothing_grid, dnf_grid, sigma_grid, winner_floor_grid
    ):
        records = []
        for year, event in events:
            rec = backtest_event(
                year=year,
                event=event,
                sims=sims,
                chaos_scale=chaos_scale,
                dirichlet_smoothing=smoothing,
                dnf_floor=dnf_floor,
                sigma_multiplier=sigma_mult,
                winner_floor=winner_floor,
                mu_model=mu_model,
                sigma_bundle=sigma_bundle,
            )
            rec.update(
                {
                    "chaos_scale": chaos_scale,
                    "dirichlet_smoothing": smoothing,
                    "dnf_floor": dnf_floor,
                    "sigma_multiplier": sigma_mult,
                    "winner_floor": winner_floor,
                }
            )
            records.append(rec)
            results.append(rec)

        score = objective_metric(records)
        if score < best_score:
            best_score = score
            best_cfg = {
                "chaos_scale": chaos_scale,
                "dirichlet_smoothing": smoothing,
                "dnf_floor": dnf_floor,
                "sigma_multiplier": sigma_mult,
                "winner_floor": winner_floor,
                "score": score,
            }

    df = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)

    if best_cfg:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w") as f:
            json.dump(best_cfg, f, indent=2)
    return {"best_config": best_cfg, "results": results}


if __name__ == "__main__":
    outcome = run_calibration()
    print("Best config:", outcome["best_config"])
    print(f"Results written to {RESULTS_PATH}")
