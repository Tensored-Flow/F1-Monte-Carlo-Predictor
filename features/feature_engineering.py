"""Feature engineering helpers for race-day predictions."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from utils.track_factors import get_track_factors


def _safe_mean(series: pd.Series, default=np.nan):
    return series.mean() if len(series) else default


def _rolling_metric(
    df: pd.DataFrame, group_col: str, value_col: str, window: int
) -> pd.Series:
    """Compute rolling mean per group, aligned to the last available entry."""
    return (
        df.sort_values([group_col, "Date" if "Date" in df.columns else "Round"])
        .groupby(group_col)[value_col]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def build_features(
    laps: pd.DataFrame,
    historical_results: Optional[pd.DataFrame] = None,
    fp2_laps: Optional[pd.DataFrame] = None,
    weather: Optional[pd.DataFrame] = None,
    event_name: str | None = None,
) -> pd.DataFrame:
    """
    Transform session data into driver-level features.

    Args:
        laps: FastF1 laps DataFrame for the target session (e.g., qualifying).
        historical_results: optional DataFrame with columns Driver, QualiPosition, RacePosition, Date/Round.
        fp2_laps: optional FP2 laps DataFrame for long-run pace.
        weather: optional weather data to build weather-influenced pace.
    """
    if laps is None or laps.empty:
        raise ValueError("laps DataFrame is required and must be non-empty")

    # Base per-driver pace and consistency
    pace = (
        laps.groupby("Driver")
        .agg(
            fastest_lap=("LapTime", "min"),
            median_lap=("LapTime", "median"),
            lap_count=("LapTime", "size"),
            lap_var=("LapTime", lambda s: np.var(s.dt.total_seconds())),
            deleted_laps=("Deleted", "sum") if "Deleted" in laps.columns else ("LapTime", "size"),
        )
        .reset_index()
    )
    pace["fastest_lap_s"] = pace["fastest_lap"].dt.total_seconds()
    pace["median_lap_s"] = pace["median_lap"].dt.total_seconds()
    pace["consistency"] = 1 / (1 + pace["lap_var"].fillna(0))

    # Team context for teammate delta
    if "Team" in laps.columns:
        team_fast = (
            laps.groupby(["Team", "Driver"])["LapTime"]
            .min()
            .reset_index()
            .rename(columns={"LapTime": "driver_fastest"})
        )
        team_fast["team_best"] = team_fast.groupby("Team")["driver_fastest"].transform("min")
        team_fast["delta_to_teammate"] = (
            team_fast["driver_fastest"] - team_fast["team_best"]
        ).dt.total_seconds()
        pace = pace.merge(team_fast[["Driver", "delta_to_teammate"]], on="Driver", how="left")
    else:
        pace["delta_to_teammate"] = np.nan

    # Long-run pace proxy from FP2 (median)
    if fp2_laps is not None and not fp2_laps.empty:
        fp2_pace = (
            fp2_laps.groupby("Driver")["LapTime"]
            .median()
            .reset_index()
            .rename(columns={"LapTime": "fp2_median"})
        )
        fp2_pace["fp2_median_s"] = fp2_pace["fp2_median"].dt.total_seconds()
        pace = pace.merge(fp2_pace[["Driver", "fp2_median_s"]], on="Driver", how="left")
    else:
        pace["fp2_median_s"] = np.nan

    # Driver form and reliability from historical results
    if historical_results is not None and not historical_results.empty:
        hist = historical_results.copy()
        if "QualiPosition" in hist.columns:
            hist["qual_rolling"] = _rolling_metric(hist, "Driver", "QualiPosition", 5)
        if "RacePosition" in hist.columns:
            hist["race_rolling"] = _rolling_metric(hist, "Driver", "RacePosition", 5)
        form = hist.groupby("Driver").agg(
            qual_form=("qual_rolling", lambda s: s.iloc[-1] if len(s) else np.nan),
            race_form=("race_rolling", lambda s: s.iloc[-1] if len(s) else np.nan),
            dnf_rate=("DNF", lambda s: s.mean() if "DNF" in hist.columns else np.nan),
        )
        form = form.reset_index()
        pace = pace.merge(form, on="Driver", how="left")
    else:
        pace["qual_form"] = np.nan
        pace["race_form"] = np.nan
        pace["dnf_rate"] = np.nan

    # Track chaos factor from track status changes (number of incidents)
    if "TrackStatus" in laps.columns:
        chaos = (
            laps.groupby("Driver")["TrackStatus"]
            .apply(lambda s: s.fillna("0").str.count("[2-9]").sum())
            .reset_index(name="track_chaos")
        )
        pace = pace.merge(chaos, on="Driver", how="left")
    else:
        pace["track_chaos"] = 0

    # Tyre mix: fraction of laps per compound
    if "Compound" in laps.columns:
        compound_share = (
            laps.groupby(["Driver", "Compound"]).size().unstack(fill_value=0)
        )
        compound_share = compound_share.div(compound_share.sum(axis=1), axis=0).reset_index()
        pace = pace.merge(compound_share, on="Driver", how="left")

    # Weather influence: mean track temp per driver session
    if weather is not None and not weather.empty and "AirTemp" in weather.columns:
        weather_feats = weather[["AirTemp", "TrackTemp"]].mean().to_dict()
        for key, val in weather_feats.items():
            pace[f"avg_{key.lower()}"] = val

    # Final clean-up
    # Track-level factors
    track_info = get_track_factors(event_name or "Generic")
    for key, val in track_info.items():
        pace[f"track_{key}"] = val

    pace = pace.fillna(0)
    return pace
