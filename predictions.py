
import numpy as np
import pandas as pd
from scipy.stats import nbinom

def predict_player(player_df, stat, line):
    df = player_df.sort_values(
        "GAME_DATE"
    ).copy()

    # Use recent history only
    recent = df.tail(20)

    if len(recent) < 10:
        return None

    values = pd.to_numeric(
        recent[stat], errors="coerce"
    ).dropna()

    if len(values) < 10:
        return None

    # Weighted recent averages
    last5 = values.tail(5).mean()
    last10 = values.tail(10).mean()
    last20 = values.mean()

    projection = (
        0.40 * last5
        + 0.35 * last10
        + 0.25 * last20
    )

    # Estimate dispersion from past results
    mean = max(values.mean(), 0.01)
    variance = values.var(ddof=1)

    # Negative binomial requires variance > mean
    variance = max(variance, mean + 0.1)

    dispersion = mean**2 / (variance - mean)

    # Bound extreme estimates
    dispersion = np.clip(
        dispersion, 0.5, 500
    )

    p = dispersion / (
        dispersion + projection
    )

    p_over = nbinom.sf(
        np.floor(line), dispersion, p
    )

    p_under = nbinom.cdf(
        np.ceil(line) - 1, dispersion, p
    )

    return {
        "projection": projection,
        "over": p_over,
        "under": p_under,
        "last5": last5,
        "last10": last10,
        "last20": last20
    }
