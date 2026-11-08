
import pandas as pd
import streamlit as st

def show_data_audit(df):
    st.header("🔍 NBA Data Accuracy Checker")

    required = [
        "PLAYER_ID", "PLAYER_NAME",
        "GAME_DATE", "PTS", "REB",
        "AST", "MIN", "SEASON"
    ]

    missing_columns = [
        c for c in required if c not in df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing columns: {missing_columns}"
        )
        return

    data = df.copy()
    data["GAME_DATE"] = pd.to_datetime(
        data["GAME_DATE"], errors="coerce"
    )

    for col in ["PTS", "REB", "AST", "MIN"]:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    st.subheader("Dataset Overview")

    c1, c2 = st.columns(2)
    c1.metric("Player-game records", len(data))
    c2.metric(
        "Unique players",
        data["PLAYER_ID"].nunique()
    )

    c1, c2 = st.columns(2)
    c1.metric(
        "Earliest game",
        str(data["GAME_DATE"].min().date())
        if data["GAME_DATE"].notna().any()
        else "Unknown"
    )
    c2.metric(
        "Latest game",
        str(data["GAME_DATE"].max().date())
        if data["GAME_DATE"].notna().any()
        else "Unknown"
    )

    st.subheader("Data Quality Checks")

    duplicates = data.duplicated(
        subset=["PLAYER_ID", "GAME_DATE"],
        keep=False
    ).sum()

    missing = data[required].isna().sum()

    negative = (
        (data[["PTS", "REB", "AST", "MIN"]] < 0)
        .any(axis=1)
        .sum()
    )

    st.metric(
        "Possible duplicate records",
        int(duplicates)
    )
    st.metric(
        "Records with negative statistics",
        int(negative)
    )

    st.write("Missing values by column")
    st.dataframe(
        missing.rename("Missing values"),
        use_container_width=True
    )

    st.subheader("Season Coverage")

    coverage = (
        data.groupby("SEASON")
        .agg(
            records=("PLAYER_ID", "size"),
            players=("PLAYER_ID", "nunique"),
            first_game=("GAME_DATE", "min"),
            last_game=("GAME_DATE", "max")
        )
        .reset_index()
    )

    st.dataframe(
        coverage,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "These are automated quality checks, "
        "not proof that every box score is correct. "
        "Duplicate dates may include legitimate "
        "same-day games and need manual review."
    )
