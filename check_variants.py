"""The report does not say exactly how RSI is smoothed or which std the Bollinger Bands use.
This tries every combination against the report's signal counts (271 / 325 / 1856).
It only prints, it does not change config.py.
"""
import itertools

import config
import data_loader
import signals


def main():
    df = data_loader.load_ohlcv()
    want = config.REPORT_SIGNAL_COUNTS
    print(f"report: {want}   rows: {config.REPORT_SIGNAL_ROWS}\n")
    found = False
    for rsi_method, ddof in itertools.product(["simple", "wilder"], [1, 0]):
        t = signals.signal_table(df, rsi_method=rsi_method, bb_ddof=ddof)
        counts = signals.signal_counts(t)
        same = counts == want and len(t) == config.REPORT_SIGNAL_ROWS
        found = found or same
        print(f"RSI={rsi_method:<7} BB ddof={ddof}  ->  {counts}  rows={len(t)}  {'<-- MATCH' if same else ''}")
    if found:
        print("\nSet RSI_METHOD and BB_DDOF in config.py to the matching row, then re-run run_experiment.py.")
    else:
        print("\nNo combination matches. Check the data (adjusted prices, date range) before touching the rules.")


if __name__ == "__main__":
    main()
