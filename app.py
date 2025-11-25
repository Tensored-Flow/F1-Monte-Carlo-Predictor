"""F1 Monte Carlo Race Outcome Predictor - Streamlit dashboard with demo data."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ACCENT_RED = "#d3131b"
NAVY = "#0b1220"
CARD = "#111a2b"
TEXT = "#e8eef5"
MUTED = "#9fb3c8"
SHADOW = "0 18px 38px rgba(0,0,0,0.35)"
DATA_DICTIONARY = {
    "Finish Position Probability Heatmap": "Rows are drivers, columns are P1–P20; cell shows probability of finishing in that position.",
    "Expected Position": "Monte Carlo average finishing position (lower is better).",
    "5th–95th": "Central interval of the simulated finishing position distribution.",
    "Beat Teammate": "Probability this driver finishes ahead of their teammate.",
    "Volatility (σ)": "Higher σ = more spread in outcomes; chaos/noise factor in simulations.",
    "Head-to-Head": "Pairwise probability that the row driver finishes ahead of the column driver.",
    "Team Summary": "Team-level averages of expected positions and teammate edge.",
    "Mixture Regimes": "Weights for on-form vs off-form latent performance states.",
    "Race Events": "Lap-level simulation outputs like average safety cars and retirements.",
    "Scenario Knobs": "Track grip, randomness, pit delta, and retirement rates used in the sim.",
    "Model Choice": "Heuristic, Bayesian μ/σ, Mixture, or Lap-level simulator powering results.",
}
FALLBACK_CIRCUITS = [
    "Bahrain",
    "Jeddah",
    "Melbourne",
    "Suzuka",
    "Baku",
    "Miami",
    "Imola",
    "Monaco",
    "Barcelona",
    "Montreal",
    "Red Bull Ring",
    "Silverstone",
    "Hungaroring",
    "Spa",
    "Zandvoort",
    "Monza",
    "Singapore",
    "Lusail",
    "COTA",
    "Mexico City",
    "Interlagos",
    "Las Vegas",
    "Abu Dhabi",
]


def set_styles() -> None:
    """Inject global styles for dark theme cards and typography."""
    st.set_page_config(
        page_title="F1 Monte Carlo Race Outcome Predictor",
        page_icon="🏁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        f"""
        <style>
        :root {{
            --navy: {NAVY};
            --card: {CARD};
            --accent: {ACCENT_RED};
            --text: {TEXT};
            --muted: {MUTED};
        }}
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(145deg, #0b1220 0%, #0f192d 60%, #0b1220 100%);
            color: var(--text);
        }}
        [data-testid="stSidebar"] {{
            background: #0a1020;
            color: var(--text);
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: var(--text);
            font-weight: 800;
            letter-spacing: 0.4px;
        }}
        .subtitle {{
            color: var(--muted);
            font-size: 0.95rem;
            margin-top: -0.3rem;
        }}
        .hero {{
            background: rgba(17, 26, 43, 0.6);
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            box-shadow: {SHADOW};
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .hero strong {{
            color: var(--text);
        }}
        .placeholder-text {{
            color: var(--muted);
            padding: 1rem;
        }}
        .card-title {{
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
            color: var(--text);
        }}
        .card-subtitle {{
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 0.6rem;
        }}
        .stPlotlyChart, div[data-testid="stDataFrame"] {{
            background: var(--card);
            padding: 1.1rem;
            border-radius: 16px;
            box-shadow: {SHADOW};
            border: 1px solid rgba(255,255,255,0.06);
        }}
        div[data-testid="stDataFrame"] {{
            padding: 1rem;
        }}
        .stMarkdown.card {{
            background: var(--card);
            padding: 1rem 1.2rem;
            border-radius: 16px;
            box-shadow: {SHADOW};
            border: 1px solid rgba(255,255,255,0.06);
        }}
        .metric-inline {{
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }}
        .chip {{
            background: rgba(255,255,255,0.06);
            padding: 0.35rem 0.65rem;
            border-radius: 12px;
            color: var(--muted);
            font-size: 0.85rem;
        }}
        .stSlider, .stSelectbox, .stNumberInput, .stCheckbox, .stButton {{
            color: var(--text);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_f1_session(year: int, event: str, session_name: str) -> Dict[str, List[str]]:
    """Return mock session data with drivers and simple team mapping."""
    drivers = [
        "Verstappen",
        "Perez",
        "Leclerc",
        "Sainz",
        "Hamilton",
        "Russell",
        "Norris",
        "Piastri",
        "Alonso",
        "Stroll",
        "Ocon",
        "Gasly",
        "Albon",
        "Sargeant",
        "Bottas",
        "Zhou",
        "Magnussen",
        "Hulkenberg",
        "Tsunoda",
        "Ricciardo",
    ]
    team_pairs = [
        ("Verstappen", "Perez"),
        ("Leclerc", "Sainz"),
        ("Hamilton", "Russell"),
        ("Norris", "Piastri"),
        ("Alonso", "Stroll"),
        ("Ocon", "Gasly"),
        ("Albon", "Sargeant"),
        ("Bottas", "Zhou"),
        ("Magnussen", "Hulkenberg"),
        ("Tsunoda", "Ricciardo"),
    ]
    team_names = [
        "Red Bull",
        "Ferrari",
        "Mercedes",
        "McLaren",
        "Aston Martin",
        "Alpine",
        "Williams",
        "Stake Sauber",
        "Haas",
        "RB",
    ]
    team_map: Dict[str, str] = {}
    driver_team: Dict[str, str] = {}
    for a, b in team_pairs:
        team_map[a] = b
        team_map[b] = a
    for (a, b), team in zip(team_pairs, team_names):
        driver_team[a] = team
        driver_team[b] = team

    return {
        "year": year,
        "event": event,
        "session": session_name,
        "drivers": drivers,
        "team_map": team_map,
        "driver_team": driver_team,
        "teams": team_names,
    }


def try_fastf1_session(year: int, event: str, session_name: str) -> Dict[str, Any] | None:
    """Attempt to load real session data via FastF1; return None if unavailable."""
    try:
        from utils.session_loader import load_session
        from features.feature_engineering import build_features as build_real_features
    except Exception:
        return None

    try:
        data = load_session(year, event, session_name)
        laps = data.get("laps", pd.DataFrame())
        if laps is None or laps.empty:
            return None
        session_obj = data.get("session")
        event_name = getattr(getattr(session_obj, "event", None), "EventName", str(event))
        features = build_real_features(laps, event_name=event_name)

        drivers = features["Driver"].tolist()
        # team mapping
        team_map: Dict[str, str] = {}
        driver_team: Dict[str, str] = {}
        if "Team" in laps.columns:
            team_pairs = laps[["Driver", "Team"]].dropna().drop_duplicates()
            driver_team = dict(zip(team_pairs["Driver"], team_pairs["Team"]))
            for team, grp in team_pairs.groupby("Team"):
                drv = grp["Driver"].tolist()
                if len(drv) == 2:
                    team_map[drv[0]] = drv[1]
                    team_map[drv[1]] = drv[0]

        # start positions from results if available
        results = data.get("results", pd.DataFrame())
        start_map: Dict[str, float] = {}
        if isinstance(results, pd.DataFrame) and not results.empty:
            if "Abbreviation" in results.columns:
                if "GridPosition" in results.columns:
                    start_map = results.set_index("Abbreviation")["GridPosition"].to_dict()
                elif "Position" in results.columns:
                    start_map = results.set_index("Abbreviation")["Position"].to_dict()
        if start_map:
            features["start_pos"] = features["Driver"].map(start_map).fillna(len(drivers)).astype(float)
        elif "start_pos" not in features.columns:
            features["start_pos"] = np.arange(1, len(drivers) + 1, dtype=float)

        return {
            "year": year,
            "event": event_name,
            "session": session_name,
            "drivers": drivers,
            "team_map": team_map,
            "driver_team": driver_team,
            "features": features,
            "source": "fastf1",
        }
    except Exception:
        return None


def build_features(session_data: Dict[str, List[str]]) -> pd.DataFrame:
    """Create a demo feature matrix with pace and volatility signals (mock)."""
    rng = np.random.default_rng(session_data["year"] + len(session_data["event"]))
    drivers = session_data["drivers"]
    base_pace = np.linspace(1.0, 0.2, num=len(drivers))  # quicker drivers first
    pace_jitter = rng.normal(0, 0.05, size=len(drivers))
    reliability = rng.uniform(0.8, 1.05, size=len(drivers))
    track_chaos = rng.uniform(0.9, 1.1)

    # richer, mock feature set
    sector1 = base_pace + rng.normal(0, 0.03, size=len(drivers))
    sector2 = base_pace + rng.normal(0, 0.025, size=len(drivers))
    sector3 = base_pace + rng.normal(0, 0.02, size=len(drivers))
    tyre_deg = rng.uniform(0.8, 1.2, size=len(drivers))  # higher = more degradation
    aero_eff = rng.uniform(0.9, 1.1, size=len(drivers))
    pit_skill = rng.uniform(0.9, 1.1, size=len(drivers))
    drs_gain = rng.uniform(0.05, 0.15, size=len(drivers))
    start_pos = rng.integers(1, 21, size=len(drivers))

    df = pd.DataFrame(
        {
            "Driver": drivers,
            "pace_score": base_pace + pace_jitter,
            "sector1": sector1,
            "sector2": sector2,
            "sector3": sector3,
            "reliability": reliability,
            "track_chaos": track_chaos,
            "tyre_deg": tyre_deg,
            "aero_eff": aero_eff,
            "pit_skill": pit_skill,
            "drs_gain": drs_gain,
            "start_pos": start_pos,
        }
    )
    return df


def predict_mu_sigma(features: pd.DataFrame, use_trained: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Generate mock μ (pace) and σ (volatility) vectors."""
    rng = np.random.default_rng(features.shape[0] + (11 if use_trained else 3))
    grid_bonus = np.interp(features["start_pos"], (1, 20), (0.06, -0.06))
    base_mu = (
        0.45 * features["pace_score"].to_numpy()
        + 0.2 * features["aero_eff"].to_numpy()
        + 0.1 * features["drs_gain"].to_numpy()
        - 0.05 * features["tyre_deg"].to_numpy()
        + grid_bonus
    )
    mu = base_mu + rng.normal(0, 0.02, size=len(features))
    sigma = (1.05 - features["reliability"].to_numpy()) * 1.1
    sigma += 0.25 * (features["tyre_deg"].to_numpy() - 1)
    sigma += rng.normal(0.015, 0.008, size=len(features))
    sigma = np.clip(sigma, 0.04, None)
    if use_trained:
        mu *= 1.03
        sigma *= 0.9
    return mu, sigma


def predict_mu_sigma_bayesian(features: pd.DataFrame, scenarios: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Mock Bayesian/hierarchical adjustment for μ/σ."""
    rng = np.random.default_rng(123)
    base_mu = predict_mu_sigma(features, use_trained=True)[0]
    track_effect = scenarios.get("grip_factor", 1.0)
    chaos = scenarios.get("chaos_amp", 0.0)
    mu_pool = np.mean(base_mu)
    mu = 0.85 * (base_mu * track_effect) + 0.15 * mu_pool + rng.normal(0, 0.012, size=len(features))
    sigma_prior = 0.9 * (1.1 - features["reliability"].to_numpy()) + 0.15 * (features["tyre_deg"].to_numpy() - 1)
    sigma = 0.75 * sigma_prior + 0.25 * sigma_prior.mean()
    sigma += 0.2 * chaos
    sigma += rng.normal(0.012, 0.004, size=len(features))
    sigma = np.clip(sigma, 0.04, None)
    return mu, sigma


def predict_mu_sigma_mixture(features: pd.DataFrame, scenarios: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mock mixture/latent form model with two regimes (on-form vs off-form)."""
    rng = np.random.default_rng(456)
    base_mu = predict_mu_sigma(features, use_trained=True)[0]
    chaos = scenarios.get("chaos_amp", 0.0)
    weights = rng.dirichlet(np.ones(2) * (1 + chaos * 1.5))
    # regime offsets
    regime_offsets = np.array([0.025, -0.04])
    chosen_regime = rng.choice(2, size=len(features), p=weights)
    mu = base_mu + regime_offsets[chosen_regime] + rng.normal(0, 0.02, size=len(features))
    reliability = features["reliability"].to_numpy()
    sigma = 0.07 + 0.35 * (chosen_regime == 1) + 0.18 * (features["tyre_deg"].to_numpy() - 1) + 0.15 * (1.05 - reliability)
    sigma = np.clip(sigma, 0.045, None)
    return mu, sigma, weights


def run_monte_carlo(mu: np.ndarray, sigma: np.ndarray, sims: int, scenarios: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate finishing positions using a simple Gaussian performance model with scenario knobs."""
    rng = np.random.default_rng(42 + sims + len(mu))
    sims = max(sims, 200)
    mu_adj = mu * scenarios.get("grip_factor", 1.0)
    sigma_adj = sigma * (1 + scenarios.get("chaos_amp", 0.0))
    performance = rng.normal(loc=mu_adj, scale=sigma_adj, size=(sims, len(mu)))
    # Pit delta: randomly apply a penalty to a subset of drivers
    pit_delta = scenarios.get("pit_delta", 0.0)
    if pit_delta > 0:
        pit_mask = rng.random((sims, len(mu))) < 0.2
        performance -= pit_mask * (pit_delta * 0.01)
    # Retirements push drivers to the back
    retire_prob = scenarios.get("retirement_prob", 0.0)
    if retire_prob > 0:
        retirements = rng.random((sims, len(mu))) < retire_prob
        performance = np.where(retirements, -np.inf, performance)
    order = np.argsort(-performance, axis=1)
    finish_positions = np.argsort(order, axis=1) + 1  # 1 = winner

    prob_matrix = np.zeros((len(mu), 20))
    max_pos = min(20, len(mu))
    for pos in range(1, max_pos + 1):
        prob_matrix[:, pos - 1] = (finish_positions == pos).mean(axis=0)
    return prob_matrix, finish_positions


def simulate_race_events(features: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, sims: int, scenarios: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Lap-level event simulator with safety cars, tyre deg, and pit loss.

    Optimized for stability:
    - Interpret retirement slider as per-race probability; convert to per-lap hazard.
    - Precompute tyre degradation matrix instead of repeating per lap.
    - Keep DNF order tied to retirement lap rather than array index.
    """
    rng = np.random.default_rng(777 + sims + len(mu))
    sims = max(sims, 200)
    n_drivers = len(mu)
    laps = 58
    chaos = scenarios.get("chaos_amp", 0.0)
    retirement_race_prob = scenarios.get("retirement_prob", 0.0)
    pit_delta = scenarios.get("pit_delta", 0.0)
    grip = scenarios.get("grip_factor", 1.0)

    tyre_deg = features["tyre_deg"].to_numpy()
    deg_slope = 0.002 * tyre_deg
    lap_idx = np.arange(laps)
    deg_matrix = 1 - np.outer(lap_idx, deg_slope)
    deg_matrix = np.clip(deg_matrix, 0.6, None)  # prevent negative pace late in the race

    # Convert per-race retirement probability to a per-lap hazard and ramp slightly with lap count
    if retirement_race_prob <= 0:
        lap_retire_prob = np.zeros(laps)
    else:
        base_hazard = 1 - (1 - retirement_race_prob) ** (1 / laps)
        lap_retire_prob = np.clip(base_hazard * (1 + lap_idx / laps), 0, 0.9)

    finish_positions = np.zeros((sims, n_drivers), dtype=int)
    sc_counts: List[int] = []
    retire_counts: List[int] = []

    for s in range(sims):
        safety_car_laps = rng.binomial(1, 0.02 + chaos * 0.5, size=laps)
        sc_counts.append(int(safety_car_laps.sum()))

        lap_mu = mu * grip * deg_matrix
        lap_sigma = sigma * (1 + 0.5 * safety_car_laps[:, None]) * (1 + chaos)
        lap_perf = rng.normal(lap_mu, lap_sigma)

        pit_prob = min(0.04 + chaos * 0.05, 0.25)  # expected ~2–2.5 stops over 58 laps
        pit_mask = rng.random((laps, n_drivers)) < pit_prob
        lap_perf -= pit_mask * (pit_delta * 0.015)

        if retirement_race_prob > 0:
            retire_draws = rng.random((laps, n_drivers)) < lap_retire_prob[:, None]
            retired_any = retire_draws.any(axis=0)
            first_retire_lap = np.where(retired_any, retire_draws.argmax(axis=0), laps)
            # apply a steep penalty from the retirement lap onwards
            retire_mask = lap_idx[:, None] >= first_retire_lap
            lap_perf[retire_mask] = -1e6  # harsh penalty keeps DNF order tied to retirement lap
            retire_counts.append(int(retired_any.sum()))
        else:
            retire_counts.append(0)

        total_perf = lap_perf.sum(axis=0)
        order = np.argsort(-total_perf)
        finish_positions[s] = np.argsort(order) + 1

    prob_matrix = np.zeros((n_drivers, 20))
    max_pos = min(20, n_drivers)
    for pos in range(1, max_pos + 1):
        prob_matrix[:, pos - 1] = (finish_positions == pos).mean(axis=0)
    summary = {
        "avg_safety_cars": float(np.mean(sc_counts)),
        "avg_retirements": float(np.mean(retire_counts)),
    }
    return prob_matrix, finish_positions, summary


@st.cache_resource(show_spinner=False)
def load_trained_bundles() -> Tuple[Any | None, Any | None]:
    """Load persisted μ and σ models if present."""
    artifacts = Path("models/artifacts")
    mu_path = artifacts / "mu.joblib"
    sigma_path = artifacts / "sigma.joblib"
    mu_model = joblib.load(mu_path) if mu_path.exists() else None
    sigma_bundle = joblib.load(sigma_path) if sigma_path.exists() else None
    return mu_model, sigma_bundle


def predict_with_trained(features: pd.DataFrame, mu_model: Any | None, sigma_bundle: Any | None) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Predict μ/σ using trained models, handling calibrated sigma bundles."""
    if mu_model is None and sigma_bundle is None:
        return None, None
    feat_matrix = features.drop(columns=["Driver"], errors="ignore")

    def _align(feat_df: pd.DataFrame, ref_model: Any) -> pd.DataFrame:
        if hasattr(ref_model, "feature_names_in_"):
            ref_cols = list(ref_model.feature_names_in_)
            aligned = feat_df.copy()
            for col in ref_cols:
                if col not in aligned:
                    aligned[col] = 0.0
            return aligned[ref_cols]
        return feat_df

    mu_pred = None
    if mu_model is not None:
        aligned = _align(feat_matrix, mu_model)
        mu_pred = mu_model.predict(aligned)

    sigma_pred = None
    if sigma_bundle is not None:
        sigma_model = sigma_bundle["model"] if isinstance(sigma_bundle, dict) and "model" in sigma_bundle else sigma_bundle
        aligned_sigma = _align(feat_matrix, sigma_model)
        try:
            if isinstance(sigma_bundle, dict) and "calibrator" in sigma_bundle:
                from models.sigma_model import predict_sigma_calibrated

                sigma_pred = predict_sigma_calibrated(sigma_bundle, aligned_sigma)
            else:
                sigma_pred = sigma_bundle.predict(aligned_sigma)
        except Exception:
            sigma_pred = None
    if sigma_pred is not None:
        sigma_pred = np.clip(sigma_pred, 0.02, None)
    return mu_pred, sigma_pred


def heuristic_from_real_features(features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback heuristic μ/σ when trained models are missing but real laps are available."""
    mu = -features.get("median_lap_s", pd.Series(0, index=features.index)).to_numpy()
    lap_var = features.get("lap_var", pd.Series(0.25, index=features.index)).fillna(0.25)
    sigma = np.sqrt(lap_var).to_numpy()
    sigma = np.clip(sigma, 0.03, None)
    return mu, sigma


def ensure_sim_defaults(features: pd.DataFrame) -> pd.DataFrame:
    """Add missing columns expected by simulation with reasonable defaults."""
    feats = features.copy()
    if "tyre_deg" not in feats.columns:
        feats["tyre_deg"] = 1.0
    if "reliability" not in feats.columns:
        feats["reliability"] = 0.95
    if "start_pos" not in feats.columns:
        feats["start_pos"] = np.arange(1, len(feats) + 1, dtype=float)
    return feats


def compute_heatmap(prob_matrix: np.ndarray, drivers: List[str]) -> go.Figure:
    """Build a Plotly heatmap for finish position probabilities."""
    positions = list(range(1, 21))
    z = (prob_matrix[:, :20] * 100).round(1)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=positions,
            y=drivers,
            text=z,
            texttemplate="%{text}%",
            colorscale=[[0, "#0c3c72"], [0.35, "#235c9f"], [0.7, "#f25c54"], [1, ACCENT_RED]],
            colorbar=dict(
                title=dict(text="%", font=dict(color=TEXT)),
                tickcolor=TEXT,
                tickfont=dict(color=TEXT),
            ),
            hovertemplate="Driver: %{y}<br>Pos %{x}: %{z}%<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Finish Position", font=dict(color=TEXT)),
            gridcolor="#1f2b3d",
            tickfont=dict(color=TEXT),
        ),
        yaxis=dict(title="", gridcolor="#1f2b3d", tickfont=dict(color=TEXT)),
    )
    return fig


def compute_expected_finish(
    prob_matrix: np.ndarray,
    drivers: List[str],
    finish_samples: np.ndarray,
    team_map: Dict[str, str],
) -> pd.DataFrame:
    """Summarize expected finishes and teammate comparison."""
    positions = np.arange(1, 21)
    expected = (prob_matrix[:, :20] * positions).sum(axis=1)
    p5 = np.percentile(finish_samples, 5, axis=0)
    p95 = np.percentile(finish_samples, 95, axis=0)

    idx = {d: i for i, d in enumerate(drivers)}
    beat_prob = []
    for driver in drivers:
        teammate = team_map.get(driver)
        if teammate and teammate in idx:
            di = idx[driver]
            ti = idx[teammate]
            beat_prob.append(float((finish_samples[:, di] < finish_samples[:, ti]).mean()))
        else:
            beat_prob.append(np.nan)

    df = pd.DataFrame(
        {
            "Driver": drivers,
            "Expected Position": expected,
            "5th–95th": [f"{low:.1f} – {high:.1f}" for low, high in zip(p5, p95)],
            "Beat Teammate": beat_prob,
        }
    ).sort_values("Expected Position")
    return df


def compute_volatility_chart(drivers: List[str], sigma: np.ndarray) -> go.Figure:
    """Render horizontal bar chart for driver volatility (σ)."""
    max_idx = int(np.argmax(sigma))
    colors = [ACCENT_RED if i == max_idx else "#4b5c73" for i in range(len(drivers))]
    fig = go.Figure(
        go.Bar(
            x=sigma,
            y=drivers,
            orientation="h",
            marker_color=colors,
            hovertemplate="Driver: %{y}<br>σ: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Volatility (σ)", color=TEXT, gridcolor="#1f2b3d"),
        yaxis=dict(title="", color=TEXT),
    )
    return fig


def compute_head_to_head(drivers: List[str], finish_samples: np.ndarray) -> pd.DataFrame:
    """Pairwise win probabilities: probability row driver beats column driver."""
    n = len(drivers)
    wins = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                wins[i, j] = 0.5
            else:
                wins[i, j] = float((finish_samples[:, i] < finish_samples[:, j]).mean())
    return pd.DataFrame(wins, index=drivers, columns=drivers)


def compute_team_summary(expected_df: pd.DataFrame, driver_team: Dict[str, str]) -> pd.DataFrame:
    """Aggregate expected finish by team."""
    df = expected_df.copy()
    df["Team"] = df["Driver"].map(driver_team)
    grouped = (
        df.groupby("Team")
        .agg(
            Drivers=("Driver", "count"),
            AvgExpected=("Expected Position", "mean"),
            AvgBeatTeammate=("Beat Teammate", "mean"),
        )
        .reset_index()
        .sort_values("AvgExpected")
    )
    return grouped


def render_h2h_heatmap(h2h_df: pd.DataFrame) -> go.Figure:
    """Plotly heatmap for head-to-head probabilities."""
    z = (h2h_df.values * 100).round(1)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=h2h_df.columns.tolist(),
            y=h2h_df.index.tolist(),
            colorscale=[[0, "#0c3c72"], [0.5, "#235c9f"], [0.7, "#f25c54"], [1, ACCENT_RED]],
            text=z,
            texttemplate="%{text}%",
            hovertemplate="Row beats Column<br>%{y} vs %{x}: %{z}%<extra></extra>",
            colorbar=dict(title=dict(text="%", font=dict(color=TEXT)), tickfont=dict(color=TEXT), tickcolor=TEXT),
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#1f2b3d", tickfont=dict(color=TEXT)),
        yaxis=dict(gridcolor="#1f2b3d", tickfont=dict(color=TEXT)),
    )
    return fig


@st.cache_data(show_spinner=False)
def fetch_fastf1_calendars() -> Tuple[List[int], Dict[int, List[str]]]:
    """Discover available years and circuits from fastf1, with fallback."""
    years: List[int] = []
    calendars: Dict[int, List[str]] = {}
    try:
        import fastf1  # type: ignore

        current = dt.datetime.utcnow().year
        for year in range(2014, current + 2):  # include next season if published
            try:
                sched = fastf1.get_event_schedule(year, include_testing=False)
                if sched is None or len(sched) == 0:
                    continue
                circuits = (
                    sched["EventName"].dropna().astype(str).str.strip().tolist()
                    if "EventName" in sched.columns
                    else []
                )
                if circuits:
                    years.append(year)
                    calendars[year] = circuits
            except Exception:
                continue
        years = sorted(list(set(years)))
    except Exception:
        pass

    if not years:
        years = list(range(2015, 2026))
    if not calendars:
        calendars = {y: FALLBACK_CIRCUITS for y in years}
    return years, calendars


def build_prepared_notes(results: Dict[str, Any]) -> List[str]:
    """Create a set of prepared insights based on current results."""
    if not results:
        return ["Run a simulation to get tailored notes about favorites, volatility, and teams."]

    notes: List[str] = []
    expected_df: pd.DataFrame = results["expected_df"]
    sigma: np.ndarray = results["sigma"]
    team_df: pd.DataFrame = results["team_df"]
    model_choice = results.get("model_choice", "Heuristic")
    mixture_weights = results.get("mixture_weights")
    event_summary = results.get("event_summary")

    top = expected_df.iloc[0]
    notes.append(f"{top['Driver']} is the favorite with expected P{top['Expected Position']:.1f}.")

    # Teammate edges
    beat = expected_df.sort_values("Beat Teammate", ascending=False).iloc[0]
    if not np.isnan(beat["Beat Teammate"]) and beat["Beat Teammate"] > 0.6:
        notes.append(f"Strong teammate edge: {beat['Driver']} beats teammate in {beat['Beat Teammate']*100:.1f}% of sims.")

    # High volatility
    vol_idx = int(np.argmax(sigma))
    vol_driver = expected_df.iloc[vol_idx]["Driver"]
    vol_val = sigma[vol_idx]
    if vol_val > 0.25:
        notes.append(f"Highest volatility: {vol_driver} with σ≈{vol_val:.2f}; outcomes are wide.")

    # Tight midfield check
    mid = expected_df.iloc[6:12]["Expected Position"]
    if len(mid) > 3 and mid.max() - mid.min() < 3:
        notes.append("Midfield is tightly packed (P7–P12 within ~3 positions).")

    # Team dominance
    best_team = team_df.iloc[0]
    notes.append(f"Team average leader: {best_team['Team']} with avg expected P{best_team['AvgExpected']:.2f}.")

    # Model choice
    notes.append(f"Model used: {model_choice} simulation.")

    # Mixture regimes
    if mixture_weights is not None:
        notes.append(f"Mixture regimes — On-form: {mixture_weights[0]*100:.1f}%, Off-form: {mixture_weights[1]*100:.1f}%.")

    # Event summary
    if event_summary is not None:
        notes.append(
            f"Lap-level sim hints: avg safety cars {event_summary.get('avg_safety_cars', 0):.2f}, avg retirements {event_summary.get('avg_retirements', 0):.2f}."
        )

    return notes[:8]


def build_sidebar() -> Tuple[int, str, str, int, bool, bool, Dict[str, float]]:
    """Render sidebar controls and scenario toggles."""
    st.sidebar.markdown("### Session Controls")
    years_available, calendars = fetch_fastf1_calendars()
    default_year = years_available.index(min(max(years_available), dt.datetime.utcnow().year)) if years_available else 0
    year = st.sidebar.selectbox("Year", years_available, index=default_year if default_year >= 0 else 0)

    circuits = calendars.get(year, FALLBACK_CIRCUITS)
    event = st.sidebar.selectbox("Circuit / Event", circuits, index=0)

    session_name = st.sidebar.selectbox("Session", ["FP1", "FP2", "FP3", "Q", "R"], index=3)
    sims = st.sidebar.slider("Monte Carlo simulations", min_value=500, max_value=20000, value=5000, step=500)
    st.sidebar.markdown("### Scenarios")
    grip_factor = st.sidebar.slider("Track grip / pace factor", min_value=0.9, max_value=1.1, value=1.0, step=0.01)
    chaos_amp = st.sidebar.slider("Safety car / randomness", min_value=0.0, max_value=0.5, value=0.15, step=0.01)
    retirement_prob = st.sidebar.slider("Retirement probability", min_value=0.0, max_value=0.3, value=0.08, step=0.01)
    pit_delta = st.sidebar.slider("Pit stop delta (s)", min_value=0, max_value=30, value=18, step=1)
    use_trained = st.sidebar.checkbox("Use trained models if available", value=True)
    st.sidebar.markdown(
        "<span style='font-size:0.85rem;color:var(--muted);'>Trained μ/σ models use real laps + calibration; they’re more stable than the mock Monte Carlo heuristics.</span>",
        unsafe_allow_html=True,
    )
    run_clicked = st.sidebar.button("Run Simulation", use_container_width=True)

    scenarios = {
        "grip_factor": grip_factor,
        "chaos_amp": chaos_amp,
        "retirement_prob": retirement_prob,
        "pit_delta": pit_delta,
    }

    if run_clicked and not event:
        st.sidebar.error("Please load data first.")
    return year, event, session_name, sims, use_trained, run_clicked, scenarios


def hero_header(year: int, event: str, session_name: str, sims: int) -> None:
    """Top hero headline and context chips."""
    st.markdown(
        f"""
        <div class="hero">
            <h1>F1 Monte Carlo Race Outcome Predictor</h1>
            <div class="subtitle">FastF1 data → feature engineering → μ/σ → Monte Carlo → interactive visuals</div>
            <div class="metric-inline" style="margin-top:0.75rem;">
                <div class="chip">Year: <strong>{year}</strong></div>
                <div class="chip">Event: <strong>{event}</strong></div>
                <div class="chip">Session: <strong>{session_name}</strong></div>
                <div class="chip">Simulations: <strong>{sims:,}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    set_styles()
    year, event, session_name, sims, use_trained, run_clicked, scenarios = build_sidebar()
    model_options = ["Heuristic", "Bayesian μ/σ", "Mixture/latent form", "Lap-level race simulator"]
    model_choice = st.sidebar.radio("Model type", model_options, index=1)
    model_specs = {
        "Heuristic": {
            "does": "Pace/reliability heuristic without explicit uncertainty.",
            "good": "Fastest option; good for quick scans.",
            "bad": "Underestimates uncertainty vs Bayesian/mixture; no event-level shocks.",
        },
        "Bayesian μ/σ": {
            "does": "Adds shrinkage and uncertainty to μ/σ estimates.",
            "good": "More calibrated than heuristic; balanced speed/quality.",
            "bad": "Slower than heuristic; lacks regime shifts and lap events.",
        },
        "Mixture/latent form": {
            "does": "Blends on-form/off-form regimes to reflect streakiness.",
            "good": "Captures upside/downside tails and form swings.",
            "bad": "Regime assignment can be noisy; no lap-level SC/pit chaos.",
        },
        "Lap-level race simulator": {
            "does": "Lap-wise SC/VSC, pits, tyre deg, retirements.",
            "good": "Most realistic event dynamics; surfaces chaos-driven variance.",
            "bad": "Heaviest and highest variance; less stable for quick comparisons.",
        },
    }

    if "results" not in st.session_state:
        st.session_state["results"] = None

    hero_header(year, event, session_name, sims)
    st.markdown("<br>", unsafe_allow_html=True)

    if run_clicked and event:
        with st.spinner("Running Monte Carlo simulation..."):
            session_data = try_fastf1_session(year, event, session_name)
            if session_data is None:
                session_data = load_f1_session(year, event, session_name)
                features = build_features(session_data)
                session_data["features"] = features
                session_data["source"] = "mock"
            features = ensure_sim_defaults(session_data["features"])
            mixture_weights = None
            event_summary: Dict[str, float] | None = None
            mu_model, sigma_bundle = load_trained_bundles() if use_trained else (None, None)
            mu_pred, sigma_pred = predict_with_trained(features, mu_model, sigma_bundle) if use_trained else (None, None)

            if model_choice == "Heuristic":
                if mu_pred is None or sigma_pred is None:
                    if session_data.get("source") == "fastf1":
                        mu, sigma = heuristic_from_real_features(features)
                    else:
                        mu, sigma = predict_mu_sigma(features, use_trained)
                else:
                    mu, sigma = mu_pred, sigma_pred
                prob_matrix, finish_samples = run_monte_carlo(mu, sigma, sims, scenarios)
            elif model_choice == "Bayesian μ/σ":
                if mu_pred is None or sigma_pred is None:
                    if session_data.get("source") == "fastf1":
                        mu_base, sigma_base = heuristic_from_real_features(features)
                    else:
                        mu_base, sigma_base = predict_mu_sigma(features, use_trained)
                else:
                    mu_base, sigma_base = mu_pred, sigma_pred
                # light pooling toward field mean for stability
                mu = 0.9 * (mu_base * scenarios.get("grip_factor", 1.0)) + 0.1 * mu_base.mean()
                sigma = np.clip(0.8 * sigma_base + 0.2 * sigma_base.mean() + 0.15 * scenarios.get("chaos_amp", 0.0), 0.03, None)
                prob_matrix, finish_samples = run_monte_carlo(mu, sigma, sims, scenarios)
            elif model_choice == "Mixture/latent form":
                if mu_pred is None or sigma_pred is None:
                    if session_data.get("source") == "fastf1":
                        base_mu, base_sigma = heuristic_from_real_features(features)
                    else:
                        base_mu, base_sigma, mixture_weights = predict_mu_sigma_mixture(features, scenarios)
                        mu, sigma = base_mu, base_sigma
                if mu_pred is not None and sigma_pred is not None:
                    base_mu, base_sigma = mu_pred, sigma_pred
                if mixture_weights is None:
                    chaos = scenarios.get("chaos_amp", 0.0)
                    mixture_weights = np.array([0.6, 0.4]) if chaos > 0.2 else np.array([0.75, 0.25])
                regime_offsets = np.array([0.02, -0.04])
                rng = np.random.default_rng(456)
                chosen_regime = rng.choice(2, size=len(features), p=mixture_weights / mixture_weights.sum())
                mu = base_mu + regime_offsets[chosen_regime]
                sigma = np.clip(base_sigma * (1 + 0.15 * (chosen_regime == 1)), 0.04, None)
                prob_matrix, finish_samples = run_monte_carlo(mu, sigma, sims, scenarios)
            else:
                if mu_pred is None or sigma_pred is None:
                    if session_data.get("source") == "fastf1":
                        mu, sigma = heuristic_from_real_features(features)
                    else:
                        mu, sigma = predict_mu_sigma(features, use_trained)
                else:
                    mu, sigma = mu_pred, sigma_pred
                prob_matrix, finish_samples, event_summary = simulate_race_events(features, mu, sigma, sims, scenarios)

            heatmap_fig = compute_heatmap(prob_matrix, session_data["drivers"])
            expected_df = compute_expected_finish(prob_matrix, session_data["drivers"], finish_samples, session_data["team_map"])
            volatility_fig = compute_volatility_chart(session_data["drivers"], sigma)
            h2h_df = compute_head_to_head(session_data["drivers"], finish_samples)
            team_df = compute_team_summary(expected_df, session_data["driver_team"])

        st.session_state["results"] = {
            "heatmap_fig": heatmap_fig,
            "expected_df": expected_df,
            "volatility_fig": volatility_fig,
            "h2h_df": h2h_df,
            "team_df": team_df,
            "model_choice": model_choice,
            "mixture_weights": mixture_weights,
            "event_summary": event_summary,
            "mu": mu,
            "sigma": sigma,
            "drivers": session_data["drivers"],
        }

    results = st.session_state.get("results")

    tab_outcome, tab_h2h = st.tabs(["Race Outcome", "Head-to-Head & Teams"])

    with tab_outcome:
        if results:
            st.markdown(
                f'<div class="card-subtitle">Model: {results.get("model_choice", "Heuristic")}</div>',
                unsafe_allow_html=True,
            )
        col_heatmap, col_right = st.columns([1.35, 1], gap="large")

        with col_heatmap:
            st.markdown('<div class="card-title">Finish Position Probability Heatmap</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="card-subtitle">Probability by driver across positions 1–20. Dark blues favor podium; F1 red highlights stronger win chances.</div>',
                unsafe_allow_html=True,
            )
            if results:
                st.plotly_chart(results["heatmap_fig"], use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown(
                    '<div class="stMarkdown card"><div class="placeholder-text">Set year/event/session and click <strong>Run Simulation</strong> to generate results.</div></div>',
                    unsafe_allow_html=True,
                )

        with col_right:
            st.markdown('<div class="card-title">Expected Finish Table</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="card-subtitle">Expected position, uncertainty band, and probability of beating their teammate.</div>',
                unsafe_allow_html=True,
            )
            if results:
                styled_df = results["expected_df"].style.format(
                    {"Expected Position": "{:.2f}", "Beat Teammate": lambda v: "-" if np.isnan(v) else f"{v*100:.1f}%"}
                )
                st.dataframe(styled_df, use_container_width=True, hide_index=True, height=450)
            else:
                st.markdown(
                    '<div class="stMarkdown card"><div class="placeholder-text">Run a simulation to reveal expected finish distributions.</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="card-title" style="margin-top:1rem;">Volatility Index</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-subtitle">Higher σ implies more chaotic outcomes; the most volatile driver is highlighted in F1 red.</div>',
            unsafe_allow_html=True,
        )
        if results:
            st.plotly_chart(results["volatility_fig"], use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown(
                '<div class="stMarkdown card"><div class="placeholder-text">Monte Carlo results will populate the volatility view.</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="card-title" style="margin-top:1rem;">Data Dictionary</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-subtitle">Quick reference plus prepared insights based on the current simulation.</div>',
            unsafe_allow_html=True,
        )
        dd_items = "<br>".join([f"• <strong>{k}</strong>: {v}" for k, v in DATA_DICTIONARY.items()])
        st.markdown(f'<div class="stMarkdown card"><div class="placeholder-text">{dd_items}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="card-title" style="margin-top:1rem;">Prepared Outcome Notes</div>', unsafe_allow_html=True)
        if results:
            notes = build_prepared_notes(results)
            note_text = "<br>".join([f"• {n}" for n in notes])
            st.markdown(f'<div class="stMarkdown card"><div class="placeholder-text">{note_text}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="stMarkdown card"><div class="placeholder-text">Run a simulation to see tailored takeaways on favorites, volatility, and team trends.</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="card-title" style="margin-top:1rem;">Model Notes</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-subtitle">What each model does, its strengths, and trade-offs.</div>',
            unsafe_allow_html=True,
        )
        notes_html = []
        for name in model_options:
            spec = model_specs[name]
            notes_html.append(
                f"<div class='stMarkdown card' style='margin-bottom:0.5rem;'>"
                f"<strong>{name}</strong><br>"
                f"<span style='color:var(--muted);'>Does:</span> {spec['does']}<br>"
                f"<span style='color:var(--muted);'>Good at:</span> {spec['good']}<br>"
                f"<span style='color:var(--muted);'>Less good:</span> {spec['bad']}"
                f"</div>"
            )
        st.markdown("".join(notes_html), unsafe_allow_html=True)

    with tab_h2h:
        st.markdown('<div class="card-title">Head-to-Head Matrix</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-subtitle">Pairwise probability a driver (row) beats another (column). Values are derived from the same Monte Carlo runs.</div>',
            unsafe_allow_html=True,
        )
        if results:
            st.plotly_chart(render_h2h_heatmap(results["h2h_df"]), use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown(
                '<div class="stMarkdown card"><div class="placeholder-text">Run a simulation to reveal head-to-head probabilities.</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="card-title" style="margin-top:1rem;">Team Summary</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-subtitle">Team-level averages of expected position and intra-team edge.</div>',
            unsafe_allow_html=True,
        )
        if results:
            team_df = results["team_df"].copy()
            st.dataframe(
                team_df.style.format(
                    {"AvgExpected": "{:.2f}", "AvgBeatTeammate": "{:.2f}", "Drivers": "{:.0f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.markdown(
                '<div class="stMarkdown card"><div class="placeholder-text">Team aggregates appear after your first simulation.</div></div>',
                unsafe_allow_html=True,
            )

        if results and results.get("mixture_weights") is not None:
            mix = results["mixture_weights"]
            st.markdown('<div class="card-title" style="margin-top:1rem;">Mixture Regime Weights</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="card-subtitle">Relative weight of on-form vs off-form regimes in the latent mixture model.</div>',
                unsafe_allow_html=True,
            )
            mix_df = pd.DataFrame({"Regime": ["On-form", "Off-form"], "Weight": mix})
            st.bar_chart(mix_df.set_index("Regime"))

        if results and results.get("event_summary") is not None:
            st.markdown('<div class="card-title" style="margin-top:1rem;">Race Event Summary</div>', unsafe_allow_html=True)
            summary = results["event_summary"]
            st.markdown(
                f'<div class="stMarkdown card"><div class="placeholder-text">Avg safety cars: {summary.get("avg_safety_cars", 0):.2f} • Avg retirements: {summary.get("avg_retirements", 0):.2f}</div></div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
