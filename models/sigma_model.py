"""Model to predict volatility (sigma)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_sigma_pipeline(feature_df: pd.DataFrame, model=None) -> Pipeline:
    model = model or RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    )
    numeric_cols = feature_df.columns
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_cols)], remainder="drop"
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_sigma_model(
    features: pd.DataFrame, target: pd.Series, model: Any | None = None
) -> Pipeline:
    """Train a baseline model estimating variance/volatility."""
    if features.empty:
        raise ValueError("features DataFrame is empty")
    pipeline = _build_sigma_pipeline(features, model=model)
    pipeline.fit(features, target)
    return pipeline


def predict_sigma(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Generate sigma predictions for each driver."""
    return model.predict(features)


def heuristic_sigma(features: pd.DataFrame) -> np.ndarray:
    """
    Lightweight sigma estimate when no trained model exists.
    Uses lap variance, deleted laps, and track chaos as inputs.
    """
    base = features.get("lap_var", pd.Series(0, index=features.index)).fillna(0)
    deleted = features.get("deleted_laps", pd.Series(0, index=features.index)).fillna(0)
    chaos = features.get("track_chaos", pd.Series(0, index=features.index)).fillna(0)
    sigma = np.sqrt(base + 0.05 * deleted + 0.01 * chaos)
    return sigma.to_numpy()


def save_model(model: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)


def cross_validated_sigma(
    features: pd.DataFrame,
    target: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
    model: Any | None = None,
) -> tuple[Pipeline, float]:
    """Train with cross-validation and return fitted model and mean CV score (neg MAE)."""
    pipeline = _build_sigma_pipeline(features, model=model)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, features, target, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    pipeline.fit(features, target)
    return pipeline, float(scores.mean())


def calibrate_sigma(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict:
    """
    Fit isotonic regression on out-of-fold predictions to calibrate sigma magnitude.
    Returns a dict with 'model' and 'calibrator'.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, features, target, cv=cv, n_jobs=-1)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof, target)
    fitted_model = model.fit(features, target)
    return {"model": fitted_model, "calibrator": calibrator}


def predict_sigma_calibrated(bundle: dict, features: pd.DataFrame) -> np.ndarray:
    """Predict sigma using a calibrated regressor bundle."""
    base_pred = bundle["model"].predict(features)
    return bundle["calibrator"].transform(base_pred)


def get_feature_importance(model: Pipeline, feature_names: Iterable[str]) -> pd.Series:
    mdl = model.named_steps.get("model")
    if mdl is None:
        return pd.Series(dtype=float)
    if hasattr(mdl, "feature_importances_"):
        return pd.Series(mdl.feature_importances_, index=feature_names)
    return pd.Series(dtype=float)
