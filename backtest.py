"""Minimal daily backtest. PRELIMINARY, rules are not tuned.

Uses the confirmation rule (LSTM + indicators + swing stop must agree), enters at the
next candle's open, stop at the last confirmed swing, target 2R, stop to entry at +1R.
Only the scored test candles are used, because that is the only period where the
LSTM predictions are out of sample.

    python backtest.py            (run run_experiment.py first)
    python backtest.py --model A
"""
import argparse
import json

import pandas as pd

import config
import data_loader
import plots
import signals
import swings as sw
import trade_planner as tp


def run_backtest(df, preds, equity=config.EQUITY):
    """df: all candles. preds: Series of LSTM predicted returns indexed by date (out of sample)."""
    sig = signals.signal_table(df)
    swing_tbl = sw.find_swings(df)
    levels = sw.swing_levels(df, swing_tbl)
    pos_of = {d: i for i, d in enumerate(df.index)}

    slip, cost = config.SLIPPAGE_PER_SIDE, config.COST_PER_SIDE
    cash = float(equity)
    free_from = 0                     # a new trade can only start from a signal on/after the last exit
    ledger = []

    for date, pred in preds.items():
        t = pos_of[date]
        if t < free_from or t + 1 >= len(df) or date not in sig.index:
            continue
        ind = sig.loc[date, "Signal"]
        if ind == "SELL" and not config.ALLOW_SHORT:
            continue
        lvl = levels.iloc[t]
        stop = lvl["swing_low"] if ind == "BUY" else lvl["swing_high"]
        decision, why = signals.confirm_signal(pred, ind, stop, df["Close"].iloc[t])
        if decision == "NO TRADE":
            continue

        entry = float(df["Open"].iloc[t + 1])             # next candle's open
        plan = tp.make_plan(decision, entry, float(stop), equity=cash)
        if plan["status"] != "TRADE":
            continue

        res = tp.run_trade(plan, df.iloc[t + 1:])
        sign = 1 if decision == "BUY" else -1
        qty = plan["quantity"]
        entry_fill = entry * (1 + sign * slip)
        exit_fill = res["exit_price"] * (1 - sign * slip)
        gross = (exit_fill - entry_fill) * qty * sign
        costs = cost * (entry_fill + exit_fill) * qty
        net = gross - costs
        cash += net
        exit_pos = t + 1 + res["exit_pos"]
        free_from = exit_pos
        ledger.append({
            "trade_id": len(ledger) + 1, "side": decision, "signal_date": date,
            "entry_date": df.index[t + 1], "entry": round(entry, 2), "stop": round(plan["stop"], 2),
            "target": round(plan["target"], 2), "quantity": qty,
            "exit_date": res["exit_date"], "exit_price": round(res["exit_price"], 2),
            "exit_reason": res["exit_reason"], "gross_pnl": round(gross, 2), "costs": round(costs, 2),
            "net_pnl": round(net, 2), "r_multiple": round(net / plan["risk_amount"], 2),
            "equity_after": round(cash, 2), "_entry_pos": t + 1, "_exit_pos": exit_pos,
            "_entry_fill": entry_fill, "_sign": sign,
        })

    ledger = pd.DataFrame(ledger)
    curve = equity_curve(df, preds.index[0], ledger, equity, cost)
    return ledger, curve


def equity_curve(df, start_date, ledger, equity, cost):
    """Daily equity, with open trades marked at the close. Entry cost is taken on the entry day."""
    days = df.loc[start_date:].index
    eq = pd.Series(float(equity), index=days)
    if ledger.empty:
        return eq
    realized = pd.Series(0.0, index=days)
    open_pnl = pd.Series(0.0, index=days)
    for _, t in ledger.iterrows():
        exit_day = df.index[t["_exit_pos"]]
        if exit_day in realized.index:
            realized.loc[exit_day] += t["net_pnl"]
        for pos in range(t["_entry_pos"], t["_exit_pos"]):          # days still open at the close
            day = df.index[pos]
            if day in open_pnl.index:
                unreal = (df["Close"].iloc[pos] - t["_entry_fill"]) * t["quantity"] * t["_sign"]
                open_pnl.loc[day] += unreal - cost * t["_entry_fill"] * t["quantity"]
    return float(equity) + realized.cumsum() + open_pnl


def summarize(ledger, curve, equity=config.EQUITY):
    if ledger.empty:
        return {"status": "PRELIMINARY", "trades": 0, "note": "no trades passed the confirmation rule"}
    wins = ledger[ledger["net_pnl"] > 0]["net_pnl"].sum()
    losses = -ledger[ledger["net_pnl"] < 0]["net_pnl"].sum()
    dd = (curve / curve.cummax() - 1).min()
    return {
        "status": "PRELIMINARY",
        "period": f"{curve.index[0].date()} to {curve.index[-1].date()}",
        "trades": int(len(ledger)),
        "win_rate_pct": round(float((ledger["net_pnl"] > 0).mean() * 100), 1),
        "profit_factor": None if losses == 0 else round(float(wins / losses), 2),
        "max_drawdown_pct": round(float(dd * 100), 2),
        "net_profit": round(float(ledger["net_pnl"].sum()), 2),
        "return_pct": round(float((curve.iloc[-1] / equity - 1) * 100), 2),
        "avg_r": round(float(ledger["r_multiple"].mean()), 2),
        "costs_paid": round(float(ledger["costs"].sum()), 2),
    }


def print_summary(s):
    print("\n==== BACKTEST SUMMARY (PRELIMINARY) ====")
    for k, v in s.items():
        print(f"  {k:<18} {v}")
    print("  Preliminary: short window, few trades, rules were not tuned. Not evidence of profit.")


def main(model="B"):
    df = data_loader.load_ohlcv()
    pred_file = config.RESULT_DIR / "lstm_test_predictions.csv"
    if not pred_file.exists():
        raise SystemExit("results/lstm_test_predictions.csv not found. Run run_experiment.py first.")
    preds = pd.read_csv(pred_file, index_col=0, parse_dates=True)[f"pred_{model}"]

    ledger, curve = run_backtest(df, preds)
    summary = summarize(ledger, curve)
    print_summary(summary)

    out = ledger.drop(columns=[c for c in ledger.columns if c.startswith("_")]) if not ledger.empty else ledger
    out.to_csv(config.RESULT_DIR / "trade_ledger.csv", index=False)
    curve.rename("equity").to_csv(config.RESULT_DIR / "equity_curve.csv")
    (config.RESULT_DIR / "backtest_summary.json").write_text(json.dumps(summary, indent=2))
    plots.equity_chart(curve, config.RESULT_DIR / "equity_curve.png")
    print("\nSaved trade_ledger.csv, equity_curve.csv, equity_curve.png, backtest_summary.json in results/")
    return ledger, curve, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="B", choices=["A", "B"])
    main(ap.parse_args().model)
