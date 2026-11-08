
import pandas as pd
import streamlit as st

BASE_URL = (
    "https://huggingface.co/datasets/"
    "cbratkovics/nba-game-logs/resolve/main/"
    "game_logs/"
)

@st.cache_data(ttl=21600)
def load_nba_data():
    seasons = [
        "2023-24",
        "2024-25",
        "2025-26"
    ]

    frames = []

    for season in seasons:
        url = (
            BASE_URL
            + f"game_logs_{season}.parquet"
        )

        data = pd.read_parquet(url)

        data["SEASON"] = season

        frames.append(data)

    df = pd.concat(
        frames,
        ignore_index=True
    )

    # Convert the dataset's column names
    # to the format our existing app expects.
    df = df.rename(columns={
        "game_date": "GAME_DATE",
        "player_id": "PLAYER_ID",
        "player_name": "PLAYER_NAME",
        "minutes": "MIN",
        "pts": "PTS",
        "reb": "REB",
        "ast": "AST"
    })

    df["GAME_DATE"] = pd.to_datetime(
        df["GAME_DATE"]
    )

    # Create matchup descriptions
    df["MATCHUP"] = (
        df["team"].astype(str)
        + " vs "
        + df["opponent"].astype(str)
    )

    return df.sort_values(
        ["PLAYER_ID", "GAME_DATE"]
    )
