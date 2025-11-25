# F1 Monte Carlo Race Outcome Predictor

An end-to-end F1 analytics project: FastF1 data ingestion → feature engineering → ML estimates for mean pace (μ) and volatility (σ) → Monte Carlo simulation → interactive visuals and backtests. Produces full finishing-position distributions, expected finish, podium/top-10 odds, and head-to-head matrices, with a Streamlit dashboard for exploration and calibration. The latest dashboard is a dark, card-based “bento” layout with scenario sliders, multiple model flavors, and auto-detected calendars from FastF1.

## Project Layout
```
f1_monte_carlo/
    README.md
    requirements.txt
    main.py
    notebooks/
        exploration.ipynb
    data/
        cache/
    features/
    models/
    simulation/
    visualizations/
    utils/
```

## Quickstart
1. Create and activate a virtualenv:
   - `python3 -m venv .venv && source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the pipeline (example: 2023 Bahrain qualifying):
   - `python main.py`
   - Outputs go to `output/` (probability tables + interactive HTML plots).
   - Dashboard: `streamlit run app.py` (pick year/event/session, run sims, view plots, backtest/calibration, feature importances). The app auto-discovers available years/circuits from FastF1 and updates the selectors.
   - CLI: `python -m cli --year 2024 --event Monaco --session Q --sims 15000`
4. Explore the data and features in the notebook:
   - Open `notebooks/exploration.ipynb` in VS Code/Jupyter and run cells.
5. Backtest and calibration:
   - Run `notebooks/backtest.ipynb` for Brier/MAE metrics over sample races.
6. Train models (optional):
   - `python train_models.py` to fit cross-validated XGBoost/forest μ and calibrated σ models; artifacts saved to `models/artifacts/`.

## How It Works
- **Data**: FastF1 timing/telemetry with local cache (`data/cache/`). Dashboard auto-detects which years/circuits FastF1 exposes (fallback lists if offline).
- **Features**: Driver/team form, pace metrics, teammate deltas, FP2 long-run pace, reliability, deleted laps, tyre mix, track chaos, weather, track factors (overtaking difficulty, pit loss, safety-car probability). The live demo also adds mock sector pace, tyre deg, aero efficiency, pit skill, DRS gain, and start positions so the UI is rich even without real data.
- **Models (pick per run)**:
  - **Heuristic**: quick-and-dirty μ/σ from pace/reliability heuristics.
  - **Bayesian μ/σ**: adds shrinkage and uncertainty for better-calibrated probabilities.
  - **Mixture/latent form**: on-form/off-form regimes to capture streakiness and upside tails.
  - **Lap-level race simulator**: lap-wise safety cars, pits, tyre deg, retirements for event-driven chaos.
- **Scenarios**: Sliders for track grip, randomness/safety-car amplification, retirement probability, and pit delta feed directly into the sims.
- **Simulation**: Draw performance by driver, rank to finish order, aggregate to full distributions, expected finish, podium/top-10, head-to-head, team aggregates, and volatility.
- **Visuals**: Dark, card-based layout with probability heatmap, expected finish table, volatility index, head-to-head matrix, team summary, prepared notes (auto takeaways), and model notes at the bottom of the page.

## Running with Trained Models
- Default uses heuristics for μ/σ (or richer mock versions). To use trained models, run `python train_models.py` (downloads sessions, fits CV models) and then set `use_trained_models=True` (CLI/Dashboard/Main). If models aren’t found, the app falls back gracefully.

## Future Improvements
- Strategy branches (1-stop vs 2-stop) and safety-car scenario sampling.
- Per-track transfer learning and live updates as sessions progress.
- Publishing live dashboard + adding CI (lint/format/tests) and small synthetic dataset for offline demos.
