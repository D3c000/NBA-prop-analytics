
import numpy as np
import pandas as pd
import streamlit as st

TEST_SEASON = "2025-26"


def run_backtest(df):
    data = df.copy()

    required = [
        "PLAYER_ID", "PLAYER_NAME",
        "GAME_DATE", "SEASON",
        "PTS", "REB", "AST"
    ]

    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data["GAME_DATE"] = pd.to_datetime(
        data["GAME_DATE"], errors="coerce"
    )

    data = data.dropna(
        subset=["PLAYER_ID", "GAME_DATE"]
    )

    # Prevent ambiguous same-day ordering
    duplicate_dates = data.duplicated(
        ["PLAYER_ID", "GAME_DATE"]
    )
    if duplicate_dates.any():
        raise ValueError(
            "Duplicate player/date records need review."
        )

    data = data.sort_values(
        ["PLAYER_ID", "GAME_DATE"]
    ).reset_index(drop=True)

    results = []

    for stat in ["PTS", "REB", "AST"]:
        temp = data[
            ["PLAYER_ID", "PLAYER_NAME",
             "GAME_DATE", "SEASON", stat]
        ].copy()

        temp[stat] = pd.to_numeric(
            temp[stat], errors="coerce"
        )

        # Actual outcome for this market
        temp["actual"] = temp[stat]

        # Previous games only
        prior = temp.groupby(
            "PLAYER_ID"
        )[stat].shift(1)

        grouped_prior = prior.groupby(
            temp["PLAYER_ID"]
        )

        temp["last5"] = grouped_prior.transform(
            lambda x: x.rolling(
                5, min_periods=5
            ).mean()
        )

        temp["last10"] = grouped_prior.transform(
            lambda x: x.rolling(
                10, min_periods=10
            ).mean()
        )

        temp["last20"] = grouped_prior.transform(
            lambda x: x.rolling(
                20, min_periods=20
            ).mean()
        )

        # Experimental weighted model
        temp["prediction"] = (
            0.40 * temp["last5"]
            + 0.35 * temp["last10"]
            + 0.25 * temp["last20"]
        )

        # Simple comparison model
        temp["baseline"] = temp["last20"]

        temp["market"] = stat

        results.append(
            temp[
                ["PLAYER_ID", "PLAYER_NAME",
                 "GAME_DATE", "SEASON",
                 "market", "actual",
                 "prediction", "baseline"]
            ]
        )

    results = pd.concat(
        results, ignore_index=True
    )

    # Evaluate on the held-out season
    results = results[
        results["SEASON"] == TEST_SEASON
    ].dropna(
        subset=["actual", "prediction", "baseline"]
    ).copy()

    results["model_error"] = (
        results["prediction"] - results["actual"]
    ).abs()

    results["baseline_error"] = (
        results["baseline"] - results["actual"]
    ).abs()

    return results.sort_values(
        ["GAME_DATE", "PLAYER_ID", "market"]
    ).reset_index(drop=True)


def show_backtest(df):
    st.header("📊 NBA Backtesting Dashboard")

    with st.spinner("Evaluating historical predictions..."):
        results = run_backtest(df)

    if results.empty:
        st.warning("No eligible historical games.")
        return

    summary = results.groupby("market").agg(
        games=("actual", "size"),
        model_mae=("model_error", "mean"),
        baseline_mae=("baseline_error", "mean")
    ).reset_index()

    summary["improvement_pct"] = np.where(
        summary["baseline_mae"] > 0,
        100 * (
            summary["baseline_mae"]
            - summary["model_mae"]
        ) / summary["baseline_mae"],
        np.nan
    )

    st.subheader("2025-26 Historical Test")

    st.dataframe(
        summary.round(3),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "MAE = mean absolute error. "
        "Lower is better. Positive improvement "
        "means the weighted model performed "
        "better than the 20-game baseline."
    )

    market = st.selectbox(
        "Choose market to inspect",
        ["PTS", "REB", "AST"],
        key="backtest_market"
    )

    selected = results[
        results["market"] == market
    ].copy()

    st.subheader("Prediction Error")

    c1, c2 = st.columns(2)

    c1.metric(
        "Model MAE",
        f"{selected['model_error'].mean():.2f}"
    )

    c2.metric(
        "Baseline MAE",
        f"{selected['baseline_error'].mean():.2f}"
    )

    st.subheader("Predicted vs Actual")

    # Use a single player so the chart is meaningful
    players = sorted(
        selected["PLAYER_NAME"].dropna().unique()
    )

    player = st.selectbox(
        "Select player",
        players,
        key="backtest_player"
    )

    player_results = selected[
        selected["PLAYER_NAME"] == player
    ].sort_values("GAME_DATE")

    chart = player_results[
        ["GAME_DATE", "prediction", "actual"]
    ].set_index("GAME_DATE")

    st.line_chart(chart)

    st.subheader("Historical Predictions")

    st.dataframe(
        player_results[
            ["GAME_DATE", "prediction",
             "actual", "model_error"]
        ].round(2),
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "This backtest measures prediction error, "
        "not betting profitability. The model weights "
        "are experimental, and the data still requires "
        "independent verification."
    )
