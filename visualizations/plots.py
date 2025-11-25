"""Plotting helpers for race outcome probabilities."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns


def heatmap_finish_probs(prob_table: pd.DataFrame, figsize=(10, 6), cmap="magma"):
    """Static heatmap of driver vs finishing position probabilities."""
    plt.figure(figsize=figsize)
    sns.heatmap(prob_table, annot=False, cmap=cmap)
    plt.xlabel("Finishing Position")
    plt.ylabel("Driver")
    plt.tight_layout()
    return plt.gca()


def plotly_finish_heatmap(prob_table: pd.DataFrame, title: str = "Finish Probabilities"):
    """Interactive heatmap using Plotly."""
    melted = prob_table.reset_index().melt(
        id_vars="index", var_name="Position", value_name="Prob"
    )
    melted = melted.rename(columns={"index": "Driver"})
    fig = px.density_heatmap(
        melted,
        x="Position",
        y="Driver",
        z="Prob",
        color_continuous_scale="magma",
        title=title,
    )
    fig.update_layout(height=600)
    return fig


def probability_bars(prob_table: pd.DataFrame, position: str = "P1"):
    """Interactive bar chart for a specific finishing position probability."""
    df = prob_table[[position]].reset_index().rename(columns={"index": "Driver"})
    fig = px.bar(
        df.sort_values(position, ascending=False),
        x="Driver",
        y=position,
        title=f"{position} Probability",
    )
    fig.update_layout(xaxis_title="Driver", yaxis_title="Probability")
    return fig


def expected_finish_plot(expected_finish: pd.Series):
    """Bar plot of expected finishing position with ordering."""
    df = expected_finish.sort_values().reset_index()
    df.columns = ["Driver", "ExpectedFinish"]
    fig = px.bar(
        df,
        x="Driver",
        y="ExpectedFinish",
        title="Expected Finish (lower is better)",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def head_to_head_heatmap(h2h: pd.DataFrame):
    """Interactive head-to-head win probability matrix."""
    fig = px.imshow(
        h2h,
        x=h2h.columns,
        y=h2h.index,
        color_continuous_scale="viridis",
        aspect="auto",
        labels=dict(color="Win Prob"),
        title="Head-to-Head Win Probabilities",
    )
    return fig


def sigma_chart(sigmas: pd.Series):
    """Bar chart of sigma (volatility) per driver."""
    df = sigmas.reset_index()
    df.columns = ["Driver", "Sigma"]
    fig = px.bar(
        df.sort_values("Sigma", ascending=False),
        x="Driver",
        y="Sigma",
        title="Volatility (σ) by Driver",
    )
    return fig
