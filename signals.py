"""Indicator signal (Fig. 6.3) and the final confirmation step."""
import numpy as np
import pandas as pd

import config
from features import add_indicators

NEEDED = ["ema12", "ema26", "macd", "macd_signal", "rsi", "pct_b"]


def indicator_score(d):
    """Four votes added together, so the score is between -4 and +4."""
    v_ema = np.sign(d["ema12"] - d["ema26"])
    v_macd = np.sign(d["macd"] - d["macd_signal"])
    v_rsi = np.where(d["rsi"] < 30, 1, np.where(d["rsi"] > 70, -1, 0))
    v_bb = np.where(d["pct_b"] < 0.2, 1, np.where(d["pct_b"] > 0.8, -1, 0))
    return v_ema + v_macd + v_rsi + v_bb


def score_to_signal(score):
    if score >= 2:
        return "BUY"
    if score <= -2:
        return "SELL"
    return "HOLD"


def signal_table(df, rsi_method=None, bb_ddof=None):
    """Close, Score and Signal for every candle that has a full set of indicators."""
    d = add_indicators(df, rsi_method=rsi_method, bb_ddof=bb_ddof)
    d = d.dropna(subset=NEEDED)
    out = pd.DataFrame({"Close": d["Close"]})
    out["Score"] = indicator_score(d).astype(int)
    out["Signal"] = out["Score"].map(score_to_signal)
    return out


def signal_counts(table):
    counts = table["Signal"].value_counts()
    return {k: int(counts.get(k, 0)) for k in ("BUY", "SELL", "HOLD")}


def last_signals(df, n=10):
    return signal_table(df).tail(n)


# ---------------- confirmation ----------------

def lstm_direction(pred, threshold=config.SIGNAL_THRESHOLD):
    if pred is None or np.isnan(pred):
        return "NONE"
    if pred > threshold:
        return "BUY"
    if pred < -threshold:
        return "SELL"
    return "NONE"


def confirm_signal(pred, ind_signal, swing_stop, ref_price, threshold=config.SIGNAL_THRESHOLD):
    """LSTM direction, indicator signal and swing stop must all agree, else NO TRADE.

    ref_price is the price we expect to enter near (the last close, since the next
    open is not known yet). Returns (decision, reason).
    """
    lstm = lstm_direction(pred, threshold)
    if lstm == "NONE":
        return "NO TRADE", "LSTM has no clear direction"
    if ind_signal not in ("BUY", "SELL"):
        return "NO TRADE", f"indicator signal is {ind_signal}"
    if lstm != ind_signal:
        return "NO TRADE", f"LSTM says {lstm} but indicators say {ind_signal}"
    if swing_stop is None or np.isnan(swing_stop):
        return "NO TRADE", "no confirmed swing level to use as stop"
    if ind_signal == "BUY" and swing_stop >= ref_price:
        return "NO TRADE", "last confirmed swing low is not below the price"
    if ind_signal == "SELL" and swing_stop <= ref_price:
        return "NO TRADE", "last confirmed swing high is not above the price"
    return ind_signal, "LSTM, indicators and swing stop agree"
