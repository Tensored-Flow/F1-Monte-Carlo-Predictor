"""Command-line interface for running the F1 Monte Carlo pipeline."""
from __future__ import annotations

import argparse

from main import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run F1 Monte Carlo predictor")
    parser.add_argument("--year", type=int, default=2023, help="Season year (e.g., 2023)")
    parser.add_argument("--event", type=str, default="Bahrain", help="Event name or round number")
    parser.add_argument("--session", type=str, default="Q", help="Session (FP1/FP2/FP3/Q/R)")
    parser.add_argument("--sims", type=int, default=10000, help="Number of Monte Carlo simulations")
    parser.add_argument("--use-trained", action="store_true", help="Use trained mu/sigma models if available")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        year=args.year,
        event=args.event,
        session_name=args.session,
        output_dir=args.output,
        use_trained_models=args.use_trained,
    )


if __name__ == "__main__":
    main()
