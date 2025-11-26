"""Pipeline entrypoint for F1 race outcome prediction."""
from __future__ import annotations

import os
from pathlib import Path

# Ensure Matplotlib can write its config/cache before importing FastF1.
_mpl_dir = Path(os.environ.get("MPLCONFIGDIR", "output/.matplotlib"))
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = _mpl_dir.as_posix()

import json
import numpy as np
import pandas as pd
import fastf1

from features.feature_engineering import build_features
from models.mu_model import load_model as load_mu_model
from models.mu_model import predict_mu, train_mu_model
from models.sigma_model import heuristic_sigma, load_model as load_sigma_model
from models.sigma_model import predict_sigma, predict_sigma_calibrated, train_sigma_model
from simulation.monte_carlo import simulate_probability_table
from utils.session_loader import load_session
from visualizations.plots import (
    expected_finish_plot,
    head_to_head_heatmap,
    plotly_finish_heatmap,
    probability_bars,
    sigma_chart,
)


def run_pipeline(
    year: int = 2023,
    event: str | int = "Bahrain",
    session_name: str = "Q",
    output_dir: str | Path = "output",
    use_trained_models: bool = False,
    historical_races: int = 6,
):
    """
    Load data, build features, predict mu/sigma, run Monte Carlo, and save outputs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load session data
    data = load_session(year, event, session_name)
    laps = data["laps"]
    session_results = data.get("results", pd.DataFrame())

    # Collect recent race results for form and DNF priors
    def _collect_recent_results(target_year: int, target_event: str | int, n_back: int) -> pd.DataFrame:
        try:
            schedule = fastf1.get_event_schedule(target_year, include_testing=False)
        except Exception:
            return pd.DataFrame()
        # Resolve round number
        target_round = None
        if isinstance(target_event, int):
            target_round = target_event
        else:
            row = schedule[schedule["EventName"] == str(target_event)]
            if not row.empty:
                target_round = int(row["RoundNumber"].iloc[0])
        if target_round is None:
            return pd.DataFrame()
        prev = schedule[schedule["RoundNumber"] < target_round].sort_values("RoundNumber", ascending=False).head(n_back)
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
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    hist_results = _collect_recent_results(year, event, historical_races)

    # Extract grid from session results if available (fallback to qualifying position).
    grid_series = None
    if session_results is not None and not session_results.empty:
        if "GridPosition" in session_results.columns:
            grid_series = session_results.set_index("Abbreviation")["GridPosition"]
        elif "Position" in session_results.columns:
            grid_series = session_results.set_index("Abbreviation")["Position"]

    # Build features with grid and historical form/DNF
    features_df = build_features(
        laps,
        historical_results=hist_results,
        weather=data.get("weather"),
        event_name=getattr(getattr(data.get("session"), "event", None), "EventName", str(event)),
        grid_positions=grid_series,
    )
    features_numeric = features_df.select_dtypes(include=[np.number])

    # Predict mu
    mu_preds: np.ndarray
    if use_trained_models and Path("models/artifacts/mu.joblib").exists():
        mu_model = load_mu_model("models/artifacts/mu.joblib")
        mu_preds = predict_mu(mu_model, features_numeric)
    else:
        # Heuristic: faster median lap => higher mu, adjust for grid position advantage.
        base_mu = -features_df["median_lap_s"].to_numpy()
        grid_pos = features_df.get("grid_position", pd.Series(np.nan, index=features_df.index)).to_numpy()
        grid_adj = np.where(np.isnan(grid_pos), 0.0, (features_df.shape[0] + 1 - grid_pos) * 0.004)
        mu_preds = base_mu + grid_adj

    # Predict sigma
    sigma_path = Path("models/artifacts/sigma.joblib")
    if use_trained_models and sigma_path.exists():
        sigma_loaded = load_sigma_model(sigma_path)
        if isinstance(sigma_loaded, dict):
            sigma_preds = predict_sigma_calibrated(sigma_loaded, features_numeric)
        else:
            sigma_preds = predict_sigma(sigma_loaded, features_numeric)
    else:
        sigma_preds = heuristic_sigma(features_df)

    drivers = features_df["Driver"].tolist()

    # Load tuned simulation hyperparameters if present.
    def _load_sim_cfg() -> dict:
        cfg_path = Path("config/calibration.json")
        base = {
            "chaos_scale": 0.25,
            "dirichlet_smoothing": 0.2,
            "dnf_floor": 0.05,
            "sigma_multiplier": 1.0,
            "winner_floor": 0.0,
        }
        if cfg_path.exists():
            try:
                loaded = json.loads(cfg_path.read_text())
                base.update({k: v for k, v in loaded.items() if k in base})
            except Exception:
                pass
        return base

    sim_cfg = _load_sim_cfg()

    sigma_preds = sigma_preds * float(sim_cfg.get("sigma_multiplier", 1.0))

    # Derive a per-driver DNF prior with a floor to widen tails when needed.
    dnf_col = "dnf_rate" if "dnf_rate" in features_df.columns else None
    dnf_prior = (
        features_df[dnf_col].fillna(sim_cfg["dnf_floor"]).clip(sim_cfg["dnf_floor"], 0.75).to_numpy()
        if dnf_col
        else np.full(len(drivers), sim_cfg["dnf_floor"])
    )

    # Run Monte Carlo with tail-widening chaos and DNF prior to avoid overconfidence.
    sim_result = simulate_probability_table(
        mu_preds,
        sigma_preds,
        n_sims=10_000,
        driver_labels=drivers,
        random_state=42,
        chaos_scale=float(sim_cfg["chaos_scale"]),
        base_dnf_prob=dnf_prior,
        dirichlet_smoothing=float(sim_cfg["dirichlet_smoothing"]),
        winner_floor=float(sim_cfg.get("winner_floor", 0.0)),
    )

    # Save tables
    sim_result.probabilities.to_csv(output_path / "probabilities.csv")
    summary = pd.DataFrame(
        {
            "Driver": drivers,
            "ExpectedFinish": sim_result.expected_finish.values,
            "PodiumProb": sim_result.podium_prob.values,
            "Top10Prob": sim_result.top10_prob.values,
        }
    ).sort_values("ExpectedFinish")
    summary.to_csv(output_path / "summary.csv", index=False)
    sim_result.head_to_head.to_csv(output_path / "head_to_head.csv")

    # Generate and save plots (HTML for interactivity)
    finish_fig = plotly_finish_heatmap(sim_result.probabilities)
    finish_fig.write_html(output_path / "finish_heatmap.html")
    p1_fig = probability_bars(sim_result.probabilities, position="P1")
    p1_fig.write_html(output_path / "p1_probabilities.html")
    exp_fig = expected_finish_plot(sim_result.expected_finish)
    exp_fig.write_html(output_path / "expected_finish.html")
    h2h_fig = head_to_head_heatmap(sim_result.head_to_head)
    h2h_fig.write_html(output_path / "head_to_head.html")
    sigma_fig = sigma_chart(pd.Series(sigma_preds, index=drivers))
    sigma_fig.write_html(output_path / "sigma.html")

    # Terminal summary
    display_cols = ["Driver", "ExpectedFinish", "PodiumProb", "Top10Prob"]
    print("=== Monte Carlo Summary ===")
    print(summary[display_cols].to_string(index=False, formatters={"ExpectedFinish": "{:.2f}".format}))

    return {
        "features": features_df,
        "probabilities": sim_result.probabilities,
        "summary": summary,
    }


if __name__ == "__main__":
    run_pipeline()
