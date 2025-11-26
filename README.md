# F1 Monte Carlo Race Outcome Predictor

An end-to-end F1 analytics project: FastF1 data ingestion → feature engineering → ML estimates for mean pace (μ) and volatility (σ) → Monte Carlo simulation → interactive visuals and backtests. Produces full finishing-position distributions, expected finish, podium/top-10 odds, and head-to-head matrices, with a Streamlit dashboard for exploration and calibration. The latest dashboard is a dark, card-based “bento” layout with scenario sliders, multiple model flavors, auto-detected calendars from FastF1, grid-aware μ, recent-form/DNF priors, and tuned chaos/winner floors for fatter tails.

Public demo: https://f1-monte-carlo-predictor.streamlit.app/

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

## Recent Updates
- FastF1 auto-load: the app now attempts to pull real session laps (with caching) and builds features from them; it falls back to mock data only if FastF1 is unavailable.
- Trained μ/σ pickup: if `models/artifacts/mu.joblib` and `sigma.joblib` exist (from `python train_models.py`), the dashboard automatically uses them when “Use trained models” is checked; otherwise it uses a real-data heuristic.
- Lap-level realism: retirement slider is treated as per-race probability (converted to per-lap hazard) and pits/safety-car effects are more stable to prevent runaway DNFs.
- Training pipeline robustness: feature engineering now converts all lap-time deltas to numeric seconds before model training, preventing dtype errors in cross-validation.
- Grid/form-aware features: feature builder now ingests grid positions and rolling race form/DNF rates from recent events; heuristics apply a grid advantage when trained models are absent.
- Tail tuning and calibration: the Monte Carlo layer supports chaos scaling, DNF floors, Dirichlet smoothing, and winner floors; `calibrate_params.py` grid-searches these over historical races and writes the best config to `config/calibration.json`.
- Backtests: recent calibration runs on ~20 races show improved MAE/Brier and top-5 precision (e.g., 3–4/5 correct in several recent races), but results still depend heavily on μ/σ quality.

## Motivation & Notes from the Build
- Motivation: I wanted a fast way to sanity-check paddock narratives with numbers—“Is the midfield really that tight?”—without firing up a notebook each race week.
- Difficulties: FastF1 can be brittle across seasons, and calibrating σ so tails are believable but not cartoonish required repeated backtests. Lap-level DNFs were especially tricky; naïve per-lap hazards exploded retirements. Small calibration windows make tail tuning noisy; shock winners can still be underweighted. Accuracy is bottlenecked by μ/σ inputs: richer features (grid, penalties, long-run pace, weather) and broader training data are needed to lift MAE/top-5 precision further.
- Takeaways: Shrinkage and small priors beat bespoke heuristics for stability, translating user-facing knobs (like “retirement probability”) into well-behaved hazards matters more than fancy visuals, and calibration + grid-aware features noticeably improve tails and top-5 hit rates but don’t replace the need for stronger μ/σ models.

## Future Improvements
- Strategy branches (1-stop vs 2-stop) and safety-car scenario sampling.
- Per-track transfer learning and live updates as sessions progress.
- Publishing live dashboard + adding CI (lint/format/tests) and small synthetic dataset for offline demos.
