"""Session loading helpers using FastF1 with caching."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import fastf1
import pandas as pd


def enable_cache(cache_dir: str | Path = "data/cache") -> None:
    """Enable FastF1 on-disk cache, creating the directory if needed."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_path.as_posix())


def load_session(
    year: int,
    event: str | int,
    session_name: str,
    cache_dir: str | Path = "data/cache",
    load_telemetry: bool = False,
) -> Dict[str, Any]:
    """
    Load a FastF1 session (FP1/FP2/FP3/Q/R) with caching enabled.

    Returns a dict containing the session object and key DataFrames:
    - session: raw FastF1 session object
    - laps: laps DataFrame
    - weather: weather data
    - drivers: driver info table
    - results: results table (if available)
    - telemetry (optional): per-driver telemetry merged into a dict
    """
    enable_cache(cache_dir)
    session = fastf1.get_session(year, event, session_name)
    session.load()

    data: Dict[str, Any] = {
        "session": session,
        "laps": session.laps.copy(),
        "weather": session.weather_data.copy()
        if hasattr(session, "weather_data")
        else pd.DataFrame(),
        "drivers": session.driver_info.copy()
        if hasattr(session, "driver_info")
        else pd.DataFrame(),
        "results": getattr(session, "results", pd.DataFrame()),
    }

    if load_telemetry:
        telemetry: Dict[str, pd.DataFrame] = {}
        for drv in session.drivers:
            try:
                drv_tel = session.laps.pick_driver(drv).get_telemetry()
                telemetry[drv] = drv_tel
            except Exception:
                continue
        data["telemetry"] = telemetry

    return data
