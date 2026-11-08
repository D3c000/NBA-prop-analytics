
import numpy as np
import pandas as pd
import streamlit as st

from backtesting import run_backtest


def validate_model(df):
    results = run_backtest(df).copy()

    if results.empty:
        st.warning("No backtest results available.")
        return

    st.header("🧪 NBA Model Validation")

    st.caption(
        "Out-of-sample analysis of the 2025-26 "
        "season. Results are not evidence of "
        "betting profitability."
    )

    market = st.selectbox(
        "Select market to validate",
        ["PTS", "REB", "AST"],
        key="validation_market"
    )

    data = results[
        results["market"] == market
    ].copy()

    data = data.sort_values(
        ["GAME_DATE", "PLAYER_ID"]
    )

    data["signed_error"] = (
        data["prediction"] - data["actual"]
    )

    data["error_advantage"] = (
        data["baseline_error"]
        - data["model_error"]
    )

    st.subheader("Overall Performance")

    model_mae = data["model_error"].mean()
    baseline_mae = data["baseline_error"].mean()
    bias = data["signed_error"].mean()

    c1, c2 = st.columns(2)

    c1.metric(
        "Model MAE",
        f"{model_mae:.3f}"
    )

    c2.metric(
        "Baseline MAE",
        f"{baseline_mae:.3f}"
    )

    c1, c2 = st.columns(2)

    improvement = (
        100 * (baseline_mae - model_mae)
        / baseline_mae
        if baseline_mae > 0
        else float("nan")
    )

    c1.metric(
        "MAE Improvement",
        f"{improvement:+.2f}%"
    )

    c2.metric(
        "Prediction Bias",
        f"{bias:+.3f}"
    )

    st.caption(
        "Positive bias means the model "
        "overpredicts on average. Negative "
        "bias means it underpredicts."
    )

    st.subheader("Monthly Performance")

    data["month"] = (
        pd.to_datetime(data["GAME_DATE"])
        .dt.to_period("M")
        .astype(str)
    )

    monthly = data.groupby("month").agg(
        games=("actual", "size"),
        model_mae=("model_error", "mean"),
        baseline_mae=("baseline_error", "mean"),
        advantage=("error_advantage", "mean")
    ).reset_index()

    st.dataframe(
        monthly.round(3),
        hide_index=True,
        use_container_width=True
    )

    st.line_chart(
        monthly.set_index("month")[
            ["model_mae", "baseline_mae"]
        ]
    )

    st.subheader("Player-Level Performance")

    players = data.groupby(
        ["PLAYER_ID", "PLAYER_NAME"]
    ).agg(
        games=("actual", "size"),
        model_mae=("model_error", "mean"),
        baseline_mae=("baseline_error", "mean"),
        bias=("signed_error", "mean"),
        advantage=("error_advantage", "mean")
    ).reset_index()

    min_games = st.slider(
        "Minimum games per player",
        min_value=5,
        max_value=60,
        value=20,
        key="validation_min_games"
    )

    players = players[
        players["games"] >= min_games
    ]

    st.dataframe(
        players.sort_values(
            "advantage", ascending=False
        ).round(3),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Player rankings are descriptive only. "
        "Selecting players based on test-season "
        "performance can overfit the results."
    )

    st.subheader("Uncertainty Estimate")

    # Aggregate by game date so players in
    # the same game-day are not treated as
    # fully independent observations.
    daily = data.groupby("GAME_DATE")[
        "error_advantage"
    ].mean().sort_index()

    if len(daily) < 30:
        st.warning(
            "Not enough dates for this "
            "uncertainty calculation."
        )
        return

    rng = np.random.default_rng(42)
    values = daily.to_numpy()

    # Moving-block bootstrap to retain some
    # short-term dependence between dates.
    block_length = 7
    n = len(values)
    simulations = []

    for _ in range(1000):
        indices = []

        while len(indices) < n:
            start = rng.integers(
                0, max(1, n - block_length + 1)
            )
            indices.extend(
                range(start, min(start + block_length, n))
            )

        sampled = values[
            np.array(indices[:n])
        ]

        simulations.append(
            sampled.mean()
        )

    low, high = np.percentile(
        simulations, [2.5, 97.5]
    )

    st.metric(
        "Average Daily MAE Advantage",
        f"{daily.mean():+.4f}"
    )

    st.write(
        "Approximate 95% bootstrap interval:",
        f"{low:+.4f} to {high:+.4f}"
    )

    if low > 0:
        st.success(
            "The estimated interval is above "
            "zero. This suggests improvement "
            "over the baseline under the "
            "bootstrap assumptions."
        )
    elif high < 0:
        st.warning(
            "The estimated interval is below "
            "zero. The model may perform worse "
            "than the baseline."
        )
    else:
        st.info(
            "The interval includes zero. "
            "There is insufficient evidence "
            "of a consistent improvement."
        )

    st.caption(
        "This is an approximate uncertainty "
        "check using daily averages and a "
        "7-day block bootstrap. It is not "
        "a formal proof of predictive edge."
    )
