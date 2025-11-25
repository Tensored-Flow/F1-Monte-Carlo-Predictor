"""Train cross-validated mu and calibrated sigma models using historical sessions."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from features.feature_engineering import build_features
from models.mu_model import cross_validated_mu, save_model as save_mu
from models.sigma_model import (
    _build_sigma_pipeline,
    calibrate_sigma,
    cross_validated_sigma,
    save_model as save_sigma,
)
from utils.session_loader import load_session


TRAIN_RACES: list[Tuple[int, str, str]] = [
    (2023, "Bahrain", "Q"),
    (2023, "Saudi Arabia", "Q"),
    (2023, "Australia", "Q"),
    (2023, "Azerbaijan", "Q"),
    (2023, "Miami", "Q"),
    (2023, "Monaco", "Q"),
    (2023, "Spain", "Q"),
    (2023, "Canada", "Q"),
    (2023, "Austria", "Q"),
]


def collect_training_data(races: Iterable[Tuple[int, str, str]]):
    rows = []
    for year, event, session_name in races:
        data = load_session(year, event, session_name)
        laps = data["laps"]
        event_name = data["session"].event.EventName
        feats = build_features(laps, event_name=event_name)
        feats["event"] = event_name
        # Targets: mu -> rank by fastest lap (lower rank => faster); sigma -> lap variance
        feats = feats.sort_values("fastest_lap").reset_index(drop=True)
        feats["mu_target"] = -feats["median_lap_s"]
        feats["sigma_target"] = np.sqrt(feats["lap_var"].fillna(0.5))
        rows.append(feats)
    full = pd.concat(rows, axis=0, ignore_index=True)
    feature_cols = full.select_dtypes(include=[np.number]).columns.drop(
        ["mu_target", "sigma_target"]
    )
    return full, feature_cols


def main():
    print("Collecting training data...")
    data, feature_cols = collect_training_data(TRAIN_RACES)
    X = data[feature_cols]
    y_mu = data["mu_target"]
    y_sigma = data["sigma_target"]

    print("Training mu model with CV...")
    mu_model, mu_cv = cross_validated_mu(X, y_mu, cv_splits=5, random_state=42)
    print(f"Mu CV (neg MAE): {mu_cv:.4f}")

    print("Training sigma model with CV and calibration...")
    sigma_base, sigma_cv = cross_validated_sigma(X, y_sigma, cv_splits=5, random_state=42)
    sigma_bundle = calibrate_sigma(_build_sigma_pipeline(X, model=sigma_base.named_steps['model']), X, y_sigma)
    print(f"Sigma CV (neg MAE): {sigma_cv:.4f}")

    artifacts = Path("models/artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    save_mu(mu_model, artifacts / "mu.joblib")
    # Save calibrated bundle
    import joblib
    joblib.dump(sigma_bundle, artifacts / "sigma.joblib")
    print("Saved models to models/artifacts/")


if __name__ == "__main__":
    main()
