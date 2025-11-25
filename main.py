"""Pipeline entrypoint for F1 race outcome prediction."""
from __future__ import annotations

import os
from pathlib import Path

# Ensure Matplotlib can write its config/cache before importing FastF1.
_mpl_dir = Path(os.environ.get("MPLCONFIGDIR", "output/.matplotlib"))
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = _mpl_dir.as_posix()

import numpy as np
import pandas as pd

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
):
    """
    Load data, build features, predict mu/sigma, run Monte Carlo, and save outputs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load session data
    data = load_session(year, event, session_name)
    laps = data["laps"]

    # Build features
    features_df = build_features(laps)
    features_numeric = features_df.select_dtypes(include=[np.number])

    # Predict mu
    mu_preds: np.ndarray
    if use_trained_models and Path("models/artifacts/mu.joblib").exists():
        mu_model = load_mu_model("models/artifacts/mu.joblib")
        mu_preds = predict_mu(mu_model, features_numeric)
    else:
        # Heuristic: faster median lap => higher mu
        mu_preds = -features_df["median_lap_s"].to_numpy()

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

    # Run Monte Carlo
    sim_result = simulate_probability_table(
        mu_preds, sigma_preds, n_sims=10_000, driver_labels=drivers, random_state=42
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
