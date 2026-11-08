
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

NZ = ZoneInfo("Pacific/Auckland")
MARKETS = {
    "Points": "PTS",
    "Rebounds": "REB",
    "Assists": "AST",
}

st.set_page_config(
    page_title="NBA Daily Analyst",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 NBA Daily Prop Analyst")
st.caption("TAB NZ | Experimental statistical research")

today = datetime.now(NZ).date()
season_start = today.year if today.month >= 10 else today.year - 1
season = f"{season_start}-{str(season_start + 1)[-2:]}"

st.info(
    "Enter current TAB NZ odds manually. "
    "Predictions are statistical estimates, not "
    "validated betting recommendations."
)

@st.cache_data(ttl=3600, show_spinner=False)
def load_player(name, season):
    matches = players.find_players_by_full_name(name)
    exact = [
        p for p in matches
        if p["full_name"].casefold() == name.casefold()
    ]

    if len(exact) != 1:
        raise ValueError(
            "Enter a player's full name exactly. "
            "No unique player match was found."
        )

    player = exact[0]

    result = playergamelog.PlayerGameLog(
        player_id=player["id"],
        season=season,
        season_type_all_star="Regular Season",
        timeout=30
    )

    df = result.get_data_frames()[0]

    if df.empty:
        raise ValueError("No games found for this season.")

    df["GAME_DATE"] = pd.to_datetime(
        df["GAME_DATE"], errors="coerce"
    )

    for col in ["PTS", "REB", "AST", "MIN"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df.dropna(
        subset=["GAME_DATE", "PTS", "REB", "AST"]
    )

    df = df.sort_values(
        "GAME_DATE", ascending=False
    )

    df = df.drop_duplicates(
        subset=["Game_ID"]
        if "Game_ID" in df.columns
        else ["GAME_DATE", "MATCHUP"]
    )

    return player["full_name"], df


def analyse(df, market, line, odds, side):
    col = MARKETS[market]

    # Only completed games before today's NZ date.
    historical = df[
        df["GAME_DATE"].dt.date < today
    ].copy()

    values = historical[col].head(20).to_numpy(
        dtype=float
    )

    if len(values) < 15:
        raise ValueError(
            "At least 15 prior games are required."
        )

    if not np.isfinite(values).all():
        raise ValueError("Invalid statistics found.")

    # Recent games receive greater weight.
    weights = np.exp(
        -np.arange(len(values)) / 12.0
    )
    weights /= weights.sum()

    projection = float(np.dot(weights, values))
    variance = float(
        np.dot(weights, (values - projection) ** 2)
    )
    sd = math.sqrt(variance)

    # Effective sample size for weighted data.
    effective_n = 1 / float(np.sum(weights ** 2))

    # Normal approximation with conservative
    # minimum spread; not a calibrated probability.
    sd = max(sd, 1.5 if col != "PTS" else 3.0)

    from scipy.stats import norm

    # Count statistics are integers.
    # For a whole-number line, equality is a push.
    if float(line).is_integer():
        threshold = int(line)
        over = float(
            norm.sf(threshold + 0.5, projection, sd)
        )
        under = float(
            norm.cdf(threshold - 0.5, projection, sd)
        )
        push = max(0.0, 1 - over - under)
    else:
        threshold = math.floor(line)
        over = float(
            norm.sf(threshold + 0.5, projection, sd)
        )
        under = 1 - over
        push = 0.0

    win = over if side == "Over" else under
    lose = under if side == "Over" else over

    # Expected net return per $1 staked:
    # win profit, loss of stake, push refund.
    ev = win * (odds - 1) - lose

    last5 = values[:5]
    recent_mean = float(np.mean(last5))
    season_sample_mean = float(np.mean(values))

    if side == "Over":
        hits = int(np.sum(values > line))
    else:
        hits = int(np.sum(values < line))

    return {
        "projection": projection,
        "recent": recent_mean,
        "sample_mean": season_sample_mean,
        "sd": sd,
        "win": win,
        "lose": lose,
        "push": push,
        "ev": ev,
        "hits": hits,
        "games": len(values),
        "effective_n": effective_n,
        "history": historical,
    }


def explain(name, market, side, line, odds, result):
    projection = result["projection"]
    recent = result["recent"]
    ev = result["ev"]

    if recent > projection:
        trend = "above"
    elif recent < projection:
        trend = "below"
    else:
        trend = "close to"

    if ev > 0:
        assessment = (
            "The experimental model estimates "
            "positive expected value."
        )
    else:
        assessment = (
            "The experimental model does not "
            "estimate positive expected value."
        )

    return (
        f"{name} projects for approximately "
        f"{projection:.1f} {market.lower()}. "
        f"The last-five-game average is "
        f"{recent:.1f}, {trend} the projection. "
        f"For {side.lower()} {line:g} at TAB odds "
        f"of {odds:.2f}, the estimated probability "
        f"of winning is {result['win']:.1%}. "
        f"{assessment} "
        "This estimate does not adjust for "
        "confirmed injuries, opponent strength, "
        "lineup changes or expected minutes. "
        "Its probability calibration has not "
        "been established."
    )


st.subheader("📅 Today's research")
st.write(f"New Zealand date: **{today:%d %B %Y}**")
st.write(f"NBA season: **{season}**")

st.markdown("### Enter a TAB player prop")

with st.form("prop_form"):
    name = st.text_input(
        "NBA player full name",
        placeholder="Nikola Jokic"
    )

    market = st.selectbox(
        "Player prop market",
        list(MARKETS)
    )

    side = st.selectbox(
        "Selection",
        ["Over", "Under"]
    )

    line = st.number_input(
        "TAB line",
        min_value=0.0,
        value=24.5,
        step=0.5
    )

    odds = st.number_input(
        "TAB decimal odds",
        min_value=1.01,
        max_value=100.0,
        value=1.90,
        step=0.01
    )

    checked = st.checkbox(
        "I have checked the current TAB line, "
        "odds and player's upcoming game."
    )

    submitted = st.form_submit_button(
        "🔍 Analyse Prop",
        use_container_width=True
    )

if submitted:
    if not name.strip():
        st.warning("Enter a player name.")
    elif not checked:
        st.warning(
            "Confirm the TAB market before analysing."
        )
    else:
        try:
            with st.spinner("Loading NBA game logs..."):
                full_name, df = load_player(
                    name.strip(), season
                )

            result = analyse(
                df, market, line, odds, side
            )

            st.success("Analysis completed")

            st.subheader("📊 Statistical analysis")
            st.write(
                f"**{full_name} — {side} "
                f"{line:g} {market}**"
            )

            a, b = st.columns(2)

            with a:
                st.metric(
                    "Projected total",
                    f"{result['projection']:.2f}"
                )
                st.metric(
                    "Estimated win probability",
                    f"{result['win']:.1%}"
                )

            with b:
                st.metric(
                    "TAB odds",
                    f"{odds:.2f}"
                )
                st.metric(
                    "Experimental EV",
                    f"{result['ev']:+.2%}"
                )

            st.write(
                f"Break-even probability without "
                f"pushes: **{1 / odds:.1%}**"
            )

            st.write(
                f"Historical hits: "
                f"**{result['hits']} / "
                f"{result['games']}**"
            )

            if result["push"] > 0.001:
                st.write(
                    "Estimated push probability: "
                    f"**{result['push']:.1%}**"
                )

            st.subheader("🤖 Analyst explanation")
            st.info(
                explain(
                    full_name, market, side,
                    line, odds, result
                )
            )

            st.subheader("📈 Recent performances")
            chart = (
                result["history"]
                .sort_values("GAME_DATE")
                .tail(20)
                .set_index("GAME_DATE")[
                    MARKETS[market]
                ]
            )
            st.line_chart(chart)

            st.subheader("Recent game log")
            st.dataframe(
                result["history"][
                    ["GAME_DATE", "MATCHUP",
                     "MIN", "PTS", "REB", "AST"]
                ].head(10),
                hide_index=True,
                use_container_width=True
            )

            st.warning(
                "Research only. These probabilities "
                "use a simplified distribution and "
                "are not calibrated against future "
                "NBA results. A positive estimated "
                "EV is not proof of an actual edge."
            )

        except Exception as exc:
            st.error(
                "Unable to complete analysis: "
                f"{type(exc).__name__}: {exc}"
            )
            st.info(
                "NBA.com may be unavailable or "
                "the player may have insufficient "
                "current-season games. Try again "
                "later or check the full name."
            )

st.divider()
st.caption(
    "TAB odds are entered manually. "
    "No wagers are placed. "
    "The app does not verify market availability."
)
