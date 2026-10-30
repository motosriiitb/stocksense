"""Runs the whole first-stage experiment and checks it against the report.

    python run_experiment.py

Nothing in the report's numbers is changed by this script. If a value is different
it just says DIFFERENT, so you can look into it before editing anything.
"""
import json

import pandas as pd

import config
import data_loader
import models
import signals

problems = []


def check(label, got, expected, ok=None):
    if ok is None:
        ok = got == expected
    print(f"  {label:<34} got {got!s:<14} report {expected!s:<14} {'OK' if ok else 'DIFFERENT'}")
    if not ok:
        problems.append(label)


def step_data():
    print("\n[1] Data")
    df = data_loader.load_ohlcv()
    s = data_loader.data_summary(df)
    check("number of candles", s["candles"], config.REPORT_N_CANDLES)
    check("first date", s["first_date"], config.REPORT_FIRST_DATE)
    check("last date", s["last_date"], config.REPORT_LAST_DATE)
    check("first close", s["first_close"], config.REPORT_FIRST_CLOSE)
    check("last close", s["last_close"], config.REPORT_LAST_CLOSE)
    return df


def step_signals(df):
    print("\n[2] Indicator signals (Section 8.1)")
    table = signals.signal_table(df)
    counts = signals.signal_counts(table)
    check("candles with all indicators", len(table), config.REPORT_SIGNAL_ROWS)
    for k in ("BUY", "SELL", "HOLD"):
        check(f"{k} signals", counts[k], config.REPORT_SIGNAL_COUNTS[k])

    print("\n  Last ten candles (Table 8.1):")
    last = table.tail(10)
    print(last.to_string(float_format=lambda v: f"{v:,.2f}"))
    for (date, close, score, sig), (d, row) in zip(config.REPORT_TABLE_8_1, last.iterrows()):
        same = (d.strftime("%Y-%m-%d") == date and round(row["Close"], 2) == close
                and int(row["Score"]) == score and row["Signal"] == sig)
        if not same:
            problems.append(f"Table 8.1 row {date}")
            print(f"  Table 8.1 row {date} is DIFFERENT")
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.RESULT_DIR / "signals_all.csv")
    last.to_csv(config.RESULT_DIR / "last_ten_signals.csv")
    return table


def step_split(df):
    print("\n[3] Split")
    data = models.prepare_data(df)
    info = models.split_info(data)
    for name in ("train", "val", "test"):
        i = info[name]
        print(f"  {name:<5} {i['n']:>5} candles   {i['start']} to {i['end']}")
    check("train size", info["train"]["n"], config.N_TRAIN)
    check("validation size", info["val"]["n"], config.N_VAL)
    check("test size", info["test"]["n"], config.N_TEST)
    check("test start date", info["test"]["start"], config.REPORT_TEST_START)
    check("test end date", info["test"]["end"], config.REPORT_TEST_END)
    return data


def step_models(data):
    print("\n[4] Models (seed 42)")
    rows = {}

    base_preds, y_true = models.run_baselines(data)
    for name, pred in base_preds.items():
        rows[name] = models.evaluate(y_true, pred, has_direction=(name != "Zero-return baseline"))

    lstm = {}
    for key, label in (("A", "LSTM A: OHLCV only"), ("B", "LSTM B: OHLCV + indicators")):
        print(f"  training LSTM {key} ...")
        res = models.train_lstm(data, key)
        print(f"    stopped after {res['epochs']} epochs, best val loss {res['best_val_loss']:.6f}")
        models.save_lstm(res)
        rows[label] = models.evaluate(res["y_true"], res["pred"])
        lstm[key] = res

    print(f"\n  scored test candles: {len(y_true)}")
    check("scored test candles", len(y_true), config.REPORT_N_SCORED)

    # save predictions so the backtest and the notebook can use them
    pred_table = pd.DataFrame({"y_true": lstm["B"]["y_true"],
                               "pred_A": lstm["A"]["pred"],
                               "pred_B": lstm["B"]["pred"]}, index=lstm["B"]["dates"])
    pred_table.index.name = "Date"
    pred_table.to_csv(config.RESULT_DIR / "lstm_test_predictions.csv")
    return rows


def step_table(rows):
    print("\n[5] Table 8.3 (ours vs report)")
    out = []
    for name, (mae, rmse, acc) in config.REPORT_TABLE_8_3.items():
        r = rows[name]
        ok_mae = round(r["MAE"], 5) == mae
        ok_rmse = round(r["RMSE"], 5) == rmse
        ok_acc = (acc is None and r["DirAcc"] is None) or (r["DirAcc"] is not None and acc is not None
                                                           and round(r["DirAcc"], 1) == acc)
        acc_txt = "-" if r["DirAcc"] is None else f"{r['DirAcc']:.1f}"
        print(f"  {name:<28} MAE {r['MAE']:.5f} ({'OK' if ok_mae else 'DIFFERENT'})   "
              f"RMSE {r['RMSE']:.5f} ({'OK' if ok_rmse else 'DIFFERENT'})   "
              f"Dir {acc_txt} ({'OK' if ok_acc else 'DIFFERENT'})")
        for what, ok in (("MAE", ok_mae), ("RMSE", ok_rmse), ("Directional accuracy", ok_acc)):
            if not ok:
                problems.append(f"Table 8.3 {name} {what}")
        out.append({"Model": name, "MAE": round(r["MAE"], 5), "RMSE": round(r["RMSE"], 5),
                    "Directional accuracy (%)": None if r["DirAcc"] is None else round(r["DirAcc"], 1),
                    "Report MAE": mae, "Report RMSE": rmse, "Report DirAcc": acc})
    pd.DataFrame(out).to_csv(config.RESULT_DIR / "table_8_3.csv", index=False)


def main():
    df = step_data()
    step_signals(df)
    data = step_split(df)
    rows = step_models(data)
    step_table(rows)

    print("\n" + "=" * 60)
    if not problems:
        print("Everything matches the report.")
    else:
        print(f"{len(problems)} value(s) are DIFFERENT from the report:")
        for p in problems:
            print("  -", p)
        print("Nothing has been changed. Tell your teammate/guide before editing any number.")
        print("If it is the signal counts, try:  python check_variants.py")
    (config.RESULT_DIR / "check_summary.json").write_text(json.dumps({"differences": problems}, indent=2))
    return problems


if __name__ == "__main__":
    main()
