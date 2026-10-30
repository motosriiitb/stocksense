"""One worked trade-plan example on a real recent candle (needs the saved LSTM).

    python worked_example.py

Looks back over the last 120 candles for the most recent indicator BUY/SELL where the
LSTM and the swing stop also agree. If there is none, it shows the most recent
BUY/SELL and why it ends in NO TRADE. This is an illustration, not a result.
"""
import config
import data_loader
import models
import signals
import swings as sw
import trade_planner as tp


def main(model="B", look_back=120):
    df = data_loader.load_ohlcv()
    sig = signals.signal_table(df)
    levels = sw.swing_levels(df, sw.find_swings(df))
    pos_of = {d: i for i, d in enumerate(df.index)}

    latest = sig.iloc[-1]
    print(f"Newest candle {sig.index[-1].date()}: indicator score {latest['Score']:+d} -> {latest['Signal']}")
    print("(entry = next candle's open, which does not exist yet for this candle,")
    print(" so the worked example below uses the most recent candle that has a next candle.)\n")

    candidates = sig.iloc[-look_back:]
    candidates = candidates[candidates["Signal"].isin(["BUY", "SELL"])]
    candidates = [d for d in candidates.index if pos_of[d] + 1 < len(df)][::-1]
    if not candidates:
        print("No BUY/SELL indicator signal in the last", look_back, "candles with a next candle.")
        return

    chosen = None
    for date in candidates:
        t = pos_of[date]
        pred = models.predict_latest(model, df=df.iloc[: t + 1], verbose=False)   # only data up to t
        ind = sig.loc[date, "Signal"]
        stop = levels.iloc[t]["swing_low" if ind == "BUY" else "swing_high"]
        decision, why = signals.confirm_signal(pred, ind, stop, df["Close"].iloc[t])
        entry = float(df["Open"].iloc[t + 1])
        plan = tp.make_plan(decision, entry, float(stop)) if decision != "NO TRADE" else None
        cand = (date, t, pred, ind, stop, decision, why, entry, plan)
        if chosen is None:
            chosen = cand                                   # fallback = most recent signal
        if plan is not None and plan["status"] == "TRADE":
            chosen = cand
            break

    date, t, pred, ind, stop, decision, why, entry, plan = chosen
    print(f"Candle used      : {date.date()}  close {df['Close'].iloc[t]:.2f}")
    print(f"Indicator signal : {ind} (score {sig.loc[date, 'Score']:+d})")
    print(f"LSTM {model} pred    : {pred * 100:+.2f}% over the next {config.HORIZON} candles")
    print(f"Last confirmed swing {'low' if ind == 'BUY' else 'high'} : {stop:.2f}")
    print(f"Confirmation     : {decision} ({why})")
    print(f"Next open ({df.index[t + 1].date()}) : {entry:.2f}")

    if plan is None:
        # still show the arithmetic so the planner can be followed
        plan = tp.make_plan(ind, entry, float(stop))
        print("\nConfirmation says NO TRADE, so this would not be taken. The planner arithmetic anyway:")
    print()
    tp.print_plan(plan, f"Worked example, {date.date()}")

    if plan["status"] == "TRADE":
        res = tp.run_trade(plan, df.iloc[t + 1:])
        print(f"\nWhat happened next: exit {res['exit_reason']} at {res['exit_price']:.2f} "
              f"on {res['exit_date'].date()} (before costs and slippage)")
        pnl = (res["exit_price"] - plan["entry"]) * plan["quantity"] * (1 if ind == "BUY" else -1)
        print(f"P&L before costs: {pnl:+.2f}  = {pnl / plan['risk_amount']:+.2f} R")
        print("SL = stop-loss, BE = stop had moved to entry, TP = target, OPEN = still open at the last candle")


if __name__ == "__main__":
    main()
