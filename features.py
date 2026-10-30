"""Indicators and model features. Each indicator is written once, here."""
import numpy as np
import pandas as pd

import config


# ---------------- indicators ----------------

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def macd(close, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def rsi(close, period=14, method=None):
    method = method or config.RSI_METHOD
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    if method == "wilder":
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    else:
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def bollinger(close, period=20, k=2, ddof=None):
    ddof = config.BB_DDOF if ddof is None else ddof
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=ddof)
    upper = mid + k * sd
    lower = mid - k * sd
    pct_b = (close - lower) / (upper - lower)
    width = (upper - lower) / mid
    return mid, upper, lower, pct_b, width


def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(df, rsi_method=None, bb_ddof=None):
    """Returns a copy of df with all the indicator columns added."""
    d = df.copy()
    close = d["Close"]
    d["ema12"] = ema(close, 12)
    d["ema26"] = ema(close, 26)
    d["ema20"] = ema(close, 20)
    d["ema50"] = ema(close, 50)
    d["macd"], d["macd_signal"], d["macd_hist"] = macd(close)
    d["rsi"] = rsi(close, method=rsi_method)
    d["bb_mid"], d["bb_upper"], d["bb_lower"], d["pct_b"], d["bb_width"] = bollinger(close, ddof=bb_ddof)
    d["atr"] = atr(d["High"], d["Low"], close)
    d["rel_volume"] = d["Volume"] / d["Volume"].rolling(20).mean()
    return d


# ---------------- model features ----------------

# set A: only things derived from OHLCV
FEATURES_A = ["ret", "hl_range", "body", "vol_change", "rel_volume"]
# set B: A plus the indicators of Table 4.2 (scale-free versions)
FEATURES_B = FEATURES_A + [
    "ema12_dist", "ema26_dist", "ema20_dist", "ema50_dist",
    "macd_n", "macd_signal_n", "macd_hist_n",
    "rsi", "pct_b", "bb_width", "atr_pct",
]
FEATURE_SETS = {"A": FEATURES_A, "B": FEATURES_B}


def build_features(df):
    """Feature table for the models. Row t only uses candles up to t."""
    d = add_indicators(df)
    close = d["Close"]
    out = pd.DataFrame(index=d.index)
    out["ret"] = close.pct_change()
    out["hl_range"] = (d["High"] - d["Low"]) / close
    out["body"] = (close - d["Open"]) / d["Open"]
    out["vol_change"] = d["Volume"].pct_change()
    out["rel_volume"] = d["rel_volume"]
    for n in (12, 26, 20, 50):
        out[f"ema{n}_dist"] = close / d[f"ema{n}"] - 1
    out["macd_n"] = d["macd"] / close
    out["macd_signal_n"] = d["macd_signal"] / close
    out["macd_hist_n"] = d["macd_hist"] / close
    out["rsi"] = d["rsi"]
    out["pct_b"] = d["pct_b"]
    out["bb_width"] = d["bb_width"]
    out["atr_pct"] = d["atr"] / close
    return out.replace([np.inf, -np.inf], np.nan)


def make_target(df, horizon=config.HORIZON):
    """y(t) = Close(t+h) / Close(t) - 1. The last h rows have no target (NaN)."""
    return df["Close"].shift(-horizon) / df["Close"] - 1
