"""Swing highs and lows, with a confirmation time so nothing looks into the future."""
import numpy as np
import pandas as pd

import config


def find_swings(df, left=config.SWING_LEFT, right=config.SWING_RIGHT):
    """A swing high has a High strictly above the `left` candles before it and the
    `right` candles after it (swing low is the same with Lows). We only know it once
    the `right` candles after it have closed, so confirm_time = pivot + right candles.
    """
    high, low = df["High"], df["Low"]
    is_high = pd.Series(True, index=df.index)
    is_low = pd.Series(True, index=df.index)
    for k in range(1, left + 1):
        is_high &= high > high.shift(k)
        is_low &= low < low.shift(k)
    for k in range(1, right + 1):
        is_high &= high > high.shift(-k)
        is_low &= low < low.shift(-k)

    rows = []
    for kind, mask, prices in (("HIGH", is_high, high), ("LOW", is_low, low)):
        for i in np.where(mask.values)[0]:
            rows.append({
                "kind": kind,
                "pivot_idx": int(i),
                "pivot_time": df.index[i],
                "price": float(prices.iloc[i]),
                "confirm_idx": int(i + right),
                "confirm_time": df.index[i + right],
            })
    cols = ["kind", "pivot_idx", "pivot_time", "price", "confirm_idx", "confirm_time"]
    swings = pd.DataFrame(rows, columns=cols)
    return swings.sort_values("confirm_idx").reset_index(drop=True)


def swings_known_at(swings, idx):
    """Only the swings that were already confirmed on or before candle number idx."""
    return swings[swings["confirm_idx"] <= idx]


def swing_levels(df, swings):
    """For every candle: the last confirmed swing high price and swing low price.
    A swing only shows up from its confirmation candle onwards."""
    out = pd.DataFrame(np.nan, index=df.index, columns=["swing_high", "swing_low"])
    for kind, col in (("HIGH", "swing_high"), ("LOW", "swing_low")):
        s = swings[swings["kind"] == kind]
        out.iloc[s["confirm_idx"].values, out.columns.get_loc(col)] = s["price"].values
    return out.ffill()
