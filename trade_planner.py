"""Trade planner (entry, stop, target, quantity) and the +1R break-even rule."""
import math

import config


def make_plan(side, entry, stop, equity=config.EQUITY, risk_fraction=config.RISK_FRACTION,
              reward_risk=config.REWARD_RISK, cap_by_capital=True):
    """Returns a dict. plan['status'] is 'TRADE' or 'NO TRADE' (with a reason)."""
    plan = {"side": side, "entry": entry, "stop": stop, "status": "NO TRADE", "reason": ""}

    if side == "BUY":
        risk = entry - stop
    elif side == "SELL":
        risk = stop - entry
    else:
        plan["reason"] = "no BUY/SELL signal"
        return plan
    if not risk > 0:
        plan["reason"] = "stop is on the wrong side of the entry"
        return plan

    sign = 1 if side == "BUY" else -1
    budget = equity * risk_fraction
    qty = math.floor(budget / risk + 1e-9)
    if qty < 1:
        plan["reason"] = f"risk budget {budget:.2f} is smaller than the risk per share {risk:.2f}"
        return plan

    qty_by_risk = qty
    capped = False
    if cap_by_capital:                      # cannot hold more shares than the account can pay for
        max_qty = math.floor(equity / entry)
        if qty > max_qty:
            qty, capped = max_qty, True
        if qty < 1:
            plan["reason"] = "not enough capital for one share"
            return plan

    plan.update({
        "status": "TRADE",
        "risk_per_share": risk,
        "target": entry + sign * reward_risk * risk,
        "trigger_1r": entry + sign * risk,
        "quantity": qty,
        "quantity_by_risk": qty_by_risk,
        "risk_amount": qty * risk,
        "capped_by_capital": capped,
    })
    return plan


def run_trade(plan, candles):
    """Walk a planned trade through the candles. First candle is the entry candle
    (we enter at its open).

    Rules (daily candles do not tell us what happened first inside a candle):
    - if a candle touches both the stop and the target, the stop is assumed to hit first
    - at +1R the stop moves to entry, but only from the NEXT candle
    - a gap through the stop fills at the open (worse than the stop)
    Returns a dict with the exit.
    """
    side = plan["side"]
    entry, target, trigger = plan["entry"], plan["target"], plan["trigger_1r"]
    stop = plan["stop"]
    moved = False

    for i in range(len(candles)):
        row = candles.iloc[i]
        o, h, l = row["Open"], row["High"], row["Low"]
        stop_name = "BE" if moved else "SL"

        if side == "BUY":
            if o <= stop:
                return _exit(candles, i, o, stop_name)
            if l <= stop:
                return _exit(candles, i, stop, stop_name)
            if h >= target:
                return _exit(candles, i, target, "TP")
            if not moved and h >= trigger:
                stop, moved = entry, True
        else:
            if o >= stop:
                return _exit(candles, i, o, stop_name)
            if h >= stop:
                return _exit(candles, i, stop, stop_name)
            if l <= target:
                return _exit(candles, i, target, "TP")
            if not moved and l <= trigger:
                stop, moved = entry, True

    last = len(candles) - 1
    return _exit(candles, last, candles.iloc[last]["Close"], "OPEN")


def _exit(candles, i, price, reason):
    return {"exit_pos": i, "exit_date": candles.index[i], "exit_price": float(price), "exit_reason": reason}


def print_plan(plan, title="Trade plan"):
    print(f"--- {title} ---")
    if plan["status"] != "TRADE":
        print(f"NO TRADE: {plan['reason']}")
        return
    print(f"Side             : {plan['side']}")
    print(f"Entry            : {plan['entry']:.2f}")
    print(f"Stop-loss        : {plan['stop']:.2f}")
    print(f"Risk per share R : {plan['risk_per_share']:.2f}")
    print(f"Target (2R)      : {plan['target']:.2f}")
    note = "  (capped by capital)" if plan["capped_by_capital"] else ""
    print(f"Quantity         : {plan['quantity']}{note}   (risk formula alone gives {plan['quantity_by_risk']})")
    print(f"Money at risk    : {plan['risk_amount']:.2f}")
    print(f"+1R trigger      : {plan['trigger_1r']:.2f}  -> stop moves to entry {plan['entry']:.2f}")
