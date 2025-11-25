"""Model to predict expected performance (mu)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _default_model(random_state: int = 42):
    """Build a default model, preferring XGBoost if available."""
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
        )
    except Exception:
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )


def _build_pipeline(feature_df: pd.DataFrame, model=None) -> Pipeline:
    """Create preprocessing + model pipeline."""
    model = model or _default_model()
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


def train_mu_model(
    features: pd.DataFrame, target: pd.Series, model: Any | None = None
) -> Pipeline:
    """Train a baseline model for expected performance (mu)."""
    if features.empty:
        raise ValueError("features DataFrame is empty")
    pipeline = _build_pipeline(features, model=model)
    pipeline.fit(features, target)
    return pipeline


def predict_mu(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Generate mu predictions for each driver."""
    return model.predict(features)


def save_model(model: Pipeline, path: str | Path) -> None:
    """Persist the trained model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Pipeline:
    """Load a persisted model."""
    return joblib.load(path)


def cross_validated_mu(
    features: pd.DataFrame,
    target: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
    model: Any | None = None,
) -> tuple[Pipeline, float]:
    """Train with cross-validation and return the fitted model and mean CV score (negative MAE)."""
    pipeline = _build_pipeline(features, model=model)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, features, target, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    pipeline.fit(features, target)
    return pipeline, float(scores.mean())


def get_feature_importance(model: Pipeline, feature_names: Iterable[str]) -> pd.Series:
    """Extract feature importance from underlying model if available."""
    mdl = model.named_steps.get("model")
    if mdl is None:
        return pd.Series(dtype=float)
    if hasattr(mdl, "feature_importances_"):
        return pd.Series(mdl.feature_importances_, index=feature_names)
    if hasattr(mdl, "get_booster"):
        booster = mdl.get_booster()
        score = booster.get_score(importance_type="weight")
        return pd.Series(score)
    return pd.Series(dtype=float)
