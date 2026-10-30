"""Run with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import features
import models
import signals
import swings as sw
import trade_planner as tp
from fake_data import make_fake_candles


@pytest.fixture(scope="module")
def candles():
    return make_fake_candles(n=2471, seed=1)


# ---------- swings ----------

def test_swing_not_available_before_confirmation(candles):
    df = candles.iloc[:400]
    swings = sw.find_swings(df)
    assert len(swings) > 10
    for _, s in swings.iterrows():
        assert s["confirm_idx"] == s["pivot_idx"] + config.SWING_RIGHT
        for idx in range(s["confirm_idx"]):
            known = sw.swings_known_at(swings, idx)
            assert not ((known["pivot_idx"] == s["pivot_idx"]) & (known["kind"] == s["kind"])).any()
        lvl = sw.swing_levels(df, swings)
        before = lvl.iloc[s["confirm_idx"] - 1]
        col = "swing_high" if s["kind"] == "HIGH" else "swing_low"
        # the level on the day before confirmation cannot be this swing's price unless an
        # older swing happens to have the same price
        older = swings[(swings["kind"] == s["kind"]) & (swings["confirm_idx"] < s["confirm_idx"])]
        if not (older["price"] == s["price"]).any():
            assert not np.isclose(before[col], s["price"], equal_nan=False) or np.isnan(before[col])


def test_swings_same_when_future_is_cut_off(candles):
    full = candles.iloc[:300]
    all_swings = sw.find_swings(full)
    for k in (60, 120, 200, 299):
        part = sw.find_swings(full.iloc[:k])
        expected = sw.swings_known_at(all_swings, k - 1)
        a = part[["kind", "pivot_idx", "price"]].reset_index(drop=True)
        b = expected[["kind", "pivot_idx", "price"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


def test_swing_levels_ignore_future(candles):
    full = candles.iloc[:300]
    lvl_full = sw.swing_levels(full, sw.find_swings(full))
    for i in (50, 100, 250):
        part = full.iloc[: i + 1]
        lvl_part = sw.swing_levels(part, sw.find_swings(part)).iloc[-1]
        np.testing.assert_allclose(lvl_full.iloc[i].values, lvl_part.values, equal_nan=True)


def test_swing_needs_strict_extreme():
    idx = pd.bdate_range("2026-01-01", periods=7)
    # the middle candle has the same high as its neighbour, so it is not a swing high
    df = pd.DataFrame({"High": [1, 2, 5, 5, 2, 1, 1], "Low": [0.5, 1, 4, 4, 1, 0.5, 0.5],
                       "Open": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)
    assert (sw.find_swings(df)["kind"] == "HIGH").sum() == 0


# ---------- indicators / features ----------

def test_features_ignore_future(candles):
    df = candles.iloc[:500]
    full = features.build_features(df)
    for i in (120, 300, 499):
        part = features.build_features(df.iloc[: i + 1]).iloc[-1]
        np.testing.assert_allclose(full.iloc[i].values, part.values, equal_nan=True)


def test_score_to_signal():
    assert signals.score_to_signal(2) == "BUY"
    assert signals.score_to_signal(4) == "BUY"
    assert signals.score_to_signal(1) == "HOLD"
    assert signals.score_to_signal(-1) == "HOLD"
    assert signals.score_to_signal(-2) == "SELL"


def test_signal_counts_add_up(candles):
    t = signals.signal_table(candles)
    assert sum(signals.signal_counts(t).values()) == len(t)


# ---------- split ----------

def test_split_sizes_and_gaps(candles):
    data = models.prepare_data(candles)
    tr, va, te = data["train"], data["val"], data["test"]
    assert tr.stop - tr.start == config.N_TRAIN
    assert va.stop - va.start == config.N_VAL
    assert te.stop - te.start == config.N_TEST
    assert va.start - tr.stop == config.GAP
    assert te.start - va.stop == config.GAP
    # the last train target looks 4 candles ahead and must stay before the validation start
    assert (tr.stop - 1) + config.HORIZON < va.start
    assert (va.stop - 1) + config.HORIZON < te.start
    assert te.stop == len(data["feats"])


def test_windows_stay_inside_part():
    X = np.arange(100 * 3, dtype=float).reshape(100, 3)
    y = np.arange(100, dtype=float)
    w, yy, pos = models.make_windows(X, y, lookback=64)
    assert w.shape == (37, 64, 3)
    assert pos[0] == 63 and yy[0] == 63
    np.testing.assert_array_equal(w[0, -1], X[63])


# ---------- trade planner ----------

def test_plan_matches_table_8_2():
    p = tp.make_plan("BUY", 1000, 970, equity=50_000, risk_fraction=0.005)
    assert p["status"] == "TRADE"
    assert p["risk_per_share"] == 30
    assert p["quantity"] == 8
    assert p["target"] == 1060
    assert p["trigger_1r"] == 1030


def test_plan_sell_side():
    p = tp.make_plan("SELL", 1000, 1030, equity=50_000, risk_fraction=0.005)
    assert p["quantity"] == 8 and p["target"] == 940 and p["trigger_1r"] == 970


def test_no_trade_if_quantity_below_one():
    p = tp.make_plan("BUY", 1000, 100, equity=50_000, risk_fraction=0.005)
    assert p["status"] == "NO TRADE"


def test_no_trade_if_stop_on_wrong_side():
    assert tp.make_plan("BUY", 1000, 1010)["status"] == "NO TRADE"
    assert tp.make_plan("SELL", 1000, 990)["status"] == "NO TRADE"


def test_quantity_capped_by_capital():
    p = tp.make_plan("BUY", 1000, 995, equity=50_000, risk_fraction=0.005)   # risk formula says 50
    assert p["quantity_by_risk"] == 50 and p["quantity"] == 50
    p = tp.make_plan("BUY", 1000, 999, equity=50_000, risk_fraction=0.005)   # formula says 250 shares
    assert p["quantity_by_risk"] == 250 and p["quantity"] == 50 and p["capped_by_capital"]


def _candles(rows):
    idx = pd.bdate_range("2026-03-02", periods=len(rows))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


def test_stop_moves_to_entry_after_plus_1r():
    plan = tp.make_plan("BUY", 100, 90)                  # R=10, trigger 110, target 120
    c = _candles([[100, 112, 101, 111],                  # reaches +1R, stop moves from next candle
                  [110, 111, 99, 100]])                  # falls back through entry
    res = tp.run_trade(plan, c)
    assert res["exit_reason"] == "BE" and res["exit_price"] == 100


def test_stop_first_when_candle_touches_both():
    plan = tp.make_plan("BUY", 100, 90)
    c = _candles([[100, 125, 85, 100]])
    res = tp.run_trade(plan, c)
    assert res["exit_reason"] == "SL" and res["exit_price"] == 90


def test_target_hit():
    plan = tp.make_plan("BUY", 100, 90)
    c = _candles([[100, 105, 95, 104], [104, 121, 103, 120]])
    res = tp.run_trade(plan, c)
    assert res["exit_reason"] == "TP" and res["exit_price"] == 120


def test_gap_through_stop_fills_at_open():
    plan = tp.make_plan("BUY", 100, 90)
    c = _candles([[100, 102, 99, 101], [85, 88, 84, 86]])
    res = tp.run_trade(plan, c)
    assert res["exit_price"] == 85


def test_short_trade_break_even():
    plan = tp.make_plan("SELL", 100, 110)                # trigger 90, target 80
    c = _candles([[100, 101, 88, 89], [90, 101, 89, 100]])
    res = tp.run_trade(plan, c)
    assert res["exit_reason"] == "BE" and res["exit_price"] == 100


# ---------- confirmation ----------

def test_confirmation_needs_everything_to_agree():
    ok, _ = signals.confirm_signal(0.01, "BUY", 95.0, 100.0)
    assert ok == "BUY"
    assert signals.confirm_signal(-0.01, "BUY", 95.0, 100.0)[0] == "NO TRADE"     # LSTM disagrees
    assert signals.confirm_signal(0.01, "HOLD", 95.0, 100.0)[0] == "NO TRADE"     # no indicator signal
    assert signals.confirm_signal(0.01, "BUY", np.nan, 100.0)[0] == "NO TRADE"    # no swing
    assert signals.confirm_signal(0.01, "BUY", 105.0, 100.0)[0] == "NO TRADE"     # stop above price
    assert signals.confirm_signal(-0.01, "SELL", 105.0, 100.0)[0] == "SELL"
    assert signals.confirm_signal(np.nan, "BUY", 95.0, 100.0)[0] == "NO TRADE"
