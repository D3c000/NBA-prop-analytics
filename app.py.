
import streamlit as st
import numpy as np
from scipy.stats import nbinom

st.set_page_config(
    page_title="NBA Prop Analytics",
    page_icon="🏀",
    layout="centered"
)

st.title("🏀 NBA Prop Analytics")
st.caption("Personal basketball probability research")

st.warning(
    "Research prototype: enter your own projections. "
    "Results are not validated betting predictions."
)

st.subheader("Player Prop Calculator")

player = st.text_input(
    "Player name",
    value="Example Player"
)

market = st.selectbox(
    "Prop market",
    ["Points", "Rebounds", "Assists"]
)

projection = st.number_input(
    "Projected stat average",
    min_value=0.1,
    max_value=100.0,
    value=24.0,
    step=0.5
)

dispersion = st.number_input(
    "Negative binomial dispersion",
    min_value=0.1,
    max_value=500.0,
    value=20.0,
    step=1.0,
    help="Illustrative value until fitted to real data"
)

line = st.number_input(
    "Bookmaker prop line",
    min_value=0.0,
    max_value=100.0,
    value=24.5,
    step=0.5
)

over_odds = st.number_input(
    "Over decimal odds",
    min_value=1.01,
    value=1.91,
    step=0.01
)

under_odds = st.number_input(
    "Under decimal odds",
    min_value=1.01,
    value=1.91,
    step=0.01
)

# Negative binomial count model
p = dispersion / (dispersion + projection)

# Strictly above or below the line
p_over = float(
    nbinom.sf(np.floor(line), dispersion, p)
)
p_under = float(
    nbinom.cdf(np.ceil(line) - 1, dispersion, p)
)
p_push = max(0.0, 1 - p_over - p_under)

ev_over = (
    p_over * (over_odds - 1) - p_under
)
ev_under = (
    p_under * (under_odds - 1) - p_over
)

# Fair odds including push probability
fair_over = (
    (1 - p_push) / p_over
    if p_over > 0 else float("inf")
)
fair_under = (
    (1 - p_push) / p_under
    if p_under > 0 else float("inf")
)

st.divider()
st.subheader("Model Results")

st.write(f"**{player} — {market}**")

c1, c2 = st.columns(2)
c1.metric("Projected Average", f"{projection:.1f}")
c2.metric("Push Probability", f"{p_push:.1%}")

c1, c2 = st.columns(2)
c1.metric("Over Probability", f"{p_over:.1%}")
c2.metric("Under Probability", f"{p_under:.1%}")

c1, c2 = st.columns(2)
c1.metric("Over Fair Odds", f"{fair_over:.2f}")
c2.metric("Under Fair Odds", f"{fair_under:.2f}")

st.subheader("Expected Value")

c1, c2 = st.columns(2)
c1.metric("Over EV", f"{ev_over:+.2%}")
c2.metric("Under EV", f"{ev_under:+.2%}")

if max(ev_over, ev_under) <= 0:
    st.info("Neither side has positive estimated EV.")
else:
    side = "Over" if ev_over >= ev_under else "Under"
    st.info(
        f"{side} has the higher model-estimated EV. "
        "This is not a validated betting recommendation."
    )

st.divider()
st.caption(
    "The distribution parameters are user-entered. "
    "Model accuracy and profitability have not been tested."
)
