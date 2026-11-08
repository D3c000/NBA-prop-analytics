
import pandas as pd
import streamlit as st
from nba_api.stats.endpoints import leaguegamelog

@st.cache_data(ttl=21600)
def load_nba_data():
    seasons = ["2023-24", "2024-25", "2025-26"]
    frames = []

    for season in seasons:
        response = leaguegamelog.LeagueGameLog(
            player_or_team_abbreviation="P",
            season=season,
            season_type_all_star="Regular Season",
            timeout=30
        )

        data = response.get_data_frames()[0]
        data["SEASON"] = season
        frames.append(data)

    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    return df.sort_values(
        ["PLAYER_ID", "GAME_DATE"]
    )
