"""F1 Monte Carlo Race Outcome Predictor - Streamlit dashboard with demo data."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Tuple

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


def build_features(session_data: Dict[str, List[str]]) -> pd.DataFrame:
    """Create a demo feature matrix with pace and volatility signals."""
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
    base_mu = (
        0.45 * features["pace_score"].to_numpy()
        + 0.2 * features["aero_eff"].to_numpy()
        + 0.1 * features["drs_gain"].to_numpy()
        - 0.05 * features["tyre_deg"].to_numpy()
    )
    mu = base_mu + rng.normal(0, 0.025, size=len(features))
    sigma = (1.1 - features["reliability"].to_numpy()) * 1.3
    sigma += 0.3 * (features["tyre_deg"].to_numpy() - 1)
    sigma += rng.normal(0.02, 0.01, size=len(features))
    sigma = np.clip(sigma, 0.05, None)
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
    mu = base_mu * track_effect + rng.normal(0, 0.015, size=len(features))
    sigma = 0.9 * (1.1 - features["reliability"].to_numpy()) + 0.25 * chaos
    sigma += rng.normal(0.015, 0.005, size=len(features))
    sigma = np.clip(sigma, 0.04, None)
    return mu, sigma


def predict_mu_sigma_mixture(features: pd.DataFrame, scenarios: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mock mixture/latent form model with two regimes (on-form vs off-form)."""
    rng = np.random.default_rng(456)
    base_mu = predict_mu_sigma(features, use_trained=True)[0]
    weights = rng.dirichlet(np.ones(2) * (1 + scenarios.get("chaos_amp", 0.0)))
    # regime offsets
    regime_offsets = np.array([0.02, -0.03])
    chosen_regime = rng.choice(2, size=len(features), p=weights)
    mu = base_mu + regime_offsets[chosen_regime] + rng.normal(0, 0.02, size=len(features))
    sigma = 0.08 + 0.4 * (chosen_regime == 1) + 0.2 * (features["tyre_deg"].to_numpy() - 1)
    sigma = np.clip(sigma, 0.05, None)
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
    """Lap-level event simulator with safety cars, tyre deg, and pit loss."""
    rng = np.random.default_rng(777 + sims + len(mu))
    sims = max(sims, 200)
    n_drivers = len(mu)
    laps = 58
    chaos = scenarios.get("chaos_amp", 0.0)
    retire_prob = scenarios.get("retirement_prob", 0.0)
    pit_delta = scenarios.get("pit_delta", 0.0)
    grip = scenarios.get("grip_factor", 1.0)

    finish_positions = np.zeros((sims, n_drivers), dtype=int)
    sc_counts = []
    retire_counts = []
    for s in range(sims):
        # track safety car events
        safety_car_laps = rng.binomial(1, 0.02 + chaos * 0.5, size=laps)
        sc_counts.append(int(safety_car_laps.sum()))
        lap_perf = np.zeros((laps, n_drivers))
        retired = np.zeros(n_drivers, dtype=bool)
        pit_mask = rng.random((laps, n_drivers)) < 0.02
        for lap in range(laps):
            deg_factor = 1 - 0.002 * features["tyre_deg"].to_numpy() * lap
            lap_mu = mu * grip * deg_factor
            lap_sigma = sigma * (1 + 0.5 * safety_car_laps[lap]) * (1 + chaos)
            lap_perf[lap] = rng.normal(lap_mu, lap_sigma)
            # pits
            lap_perf[lap] -= pit_mask[lap] * (pit_delta * 0.015)
            # retirements
            retire_now = rng.random(n_drivers) < retire_prob * (1 + lap / laps)
            retired = retired | retire_now
            lap_perf[lap][retired] = -np.inf
        retire_counts.append(int(retired.sum()))
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
    st.sidebar.caption("Toggle to see how a model-driven μ/σ would shift results.")
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
            session_data = load_f1_session(year, event, session_name)
            features = build_features(session_data)
            mixture_weights = None
            event_summary: Dict[str, float] | None = None
            if model_choice == "Heuristic":
                mu, sigma = predict_mu_sigma(features, use_trained)
                prob_matrix, finish_samples = run_monte_carlo(mu, sigma, sims, scenarios)
            elif model_choice == "Bayesian μ/σ":
                mu, sigma = predict_mu_sigma_bayesian(features, scenarios)
                prob_matrix, finish_samples = run_monte_carlo(mu, sigma, sims, scenarios)
            elif model_choice == "Mixture/latent form":
                mu, sigma, mixture_weights = predict_mu_sigma_mixture(features, scenarios)
                prob_matrix, finish_samples = run_monte_carlo(mu, sigma, sims, scenarios)
            else:
                mu, sigma = predict_mu_sigma(features, use_trained)
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
