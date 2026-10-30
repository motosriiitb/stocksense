# StockSense

AI-assisted stock analysis and trade research for RELIANCE.NS (daily candles).
Minor project, IIIT Bhopal. Research / study only, **not financial advice**.

## What is in here

| File | What it does |
| --- | --- |
| `config.py` | every fixed setting, plus the numbers copied from the report (do not edit those) |
| `data_loader.py` | downloads RELIANCE.NS with yfinance, cleans it, caches it in `data/` |
| `features.py` | EMA, MACD, RSI, Bollinger Bands, ATR (each written once) and the LSTM features |
| `signals.py` | 4-vote indicator signal (Fig. 6.3) and the BUY / SELL / NO TRADE confirmation |
| `swings.py` | swing highs / lows with a confirmation date |
| `trade_planner.py` | entry, stop, target, quantity and the +1R stop rule |
| `models.py` | split, LSTM A and B, Random Forest / XGBoost baselines, save / load, `predict_latest()` |
| `run_experiment.py` | runs everything and checks it against the report |
| `check_variants.py` | helper if the signal counts do not match (see below) |
| `backtest.py` | minimal daily backtest (preliminary) |
| `worked_example.py` | prints one trade plan on a real recent candle |
| `plots.py`, `app.py` | charts and the small Streamlit page |
| `StockSense_demo.ipynb` | Colab demo |
| `tests/` | pytest tests |

## Requirements

- Python 3.10+
- `pip install -r requirements.txt` (numpy, pandas, scikit-learn, xgboost, tensorflow, yfinance, matplotlib, plotly, streamlit, joblib, pytest)
- internet for the first run (Yahoo Finance). After that the data is read from `data/RELIANCE_NS_daily.csv`
- tested with: Python 3.12, TensorFlow 2.21, pandas 3.0, scikit-learn 1.9, xgboost 3.4

## How to run

```bash
pip install -r requirements.txt
python run_experiment.py        # data, signals, split, 2 LSTMs + baselines, Table 8.3 check (a few minutes, CPU is fine)
python worked_example.py        # one trade plan on a recent candle
python backtest.py              # preliminary backtest
python models.py                # predict_latest(): 4-day return for the newest candle
streamlit run app.py            # chart + latest signal
pytest -q                       # tests
```

Colab: open `StockSense_demo.ipynb` and use Runtime > Run all. It clones this repo by itself.

After the first successful `run_experiment.py`, commit `data/RELIANCE_NS_daily.csv` so everybody (and the notebook) uses exactly the same candles. Yahoo can slightly change adjusted prices later.

## Fixed settings

- data: RELIANCE.NS daily, 28 Oct 2016 to 28 Oct 2026, adjusted prices (2,471 candles in the report)
- seed **42**, horizon **4** candles, lookback **64**
- split in time order: **1,706 train / 363 validation / 367 test**, **4-candle gap** after train and after validation
- the split is counted back from the last candle that has a target, so the test set ends on 22 Oct 2026 (the last 4 candles have no target)
- the first 63 test candles are only history for the first 64-candle window, so the models are scored on **304** test candles
- scaler (StandardScaler) is fitted on the training part only
- LSTM: LSTM(64) > dropout > LSTM(32) > Dense(16, ReLU) > Dense(1), MSE, Adam 0.001, batch 64, early stopping patience 5
- Random Forest 300 trees, XGBoost 300 trees / depth 3 / learning rate 0.05, both on the indicator features of the current candle

## Checking against the report

`run_experiment.py` prints `OK` or `DIFFERENT` next to every number from the report: candle count, signal counts (271 / 325 / 1,856), Table 8.1, split sizes and dates, and Table 8.3. It never edits those numbers.

- if the signal counts are different, run `python check_variants.py`. The report does not say whether RSI uses a simple or Wilder average, or which std the Bollinger Bands use, so it tries all four combinations
- LSTM numbers can move a little between machines / TensorFlow versions even with seed 42. If only the LSTM rows differ, say so in the report instead of changing the numbers

## Things the report does not say (my choices)

- dropout 0.2, max 50 epochs, StandardScaler, target is not scaled
- Random Forest / XGBoost are trained on all 1,706 train rows
- LSTM A uses `ret, hl_range, body, vol_change, rel_volume`; LSTM B adds the indicators of Table 4.2 as scale-free values (distance to EMA, MACD / close, RSI, %B, band width, ATR / close)
- `predict_latest()` uses LSTM B by default (`predict_latest("A")` for the other one)
- RSI default is the simple 14-day average (`RSI_METHOD` in `config.py`)

## Swings, planner, confirmation

- **swing**: a high (low) strictly above (below) the 2 candles on each side. `confirm_time` = pivot + 2 candles. A swing is used only from its confirmation candle onwards. `tests/` checks that cutting off the future does not change any swing
- **planner**: entry = next candle's open, stop = last confirmed swing low (BUY) / high (SELL), target = entry +/- 2 x risk, quantity = floor(equity x risk fraction / risk per share), NO TRADE if it is below 1. The quantity is also limited to what the equity can pay for (needed when the stop is very close). Defaults: equity 50,000, risk 0.5%
- **+1R rule**: when price reaches entry +/- 1R the stop moves to entry, from the next candle on. If one daily candle touches both stop and target, the stop is assumed to hit first. A gap through the stop fills at the open
- **confirmation**: the LSTM direction, the indicator signal and the swing stop must all agree, otherwise NO TRADE

## Backtest (preliminary)

- costs 0.10% per side and slippage 0.05% per side (assumptions, change in `config.py`), one position at a time, SELL trades allowed (a real SELL needs futures or intraday)
- only the out-of-sample test candles are used (that is where the LSTM predictions are honest), so it is a short window with few trades
- outputs in `results/`: `trade_ledger.csv`, `equity_curve.csv`, `equity_curve.png`, `backtest_summary.json` (win rate, profit factor, max drawdown, ...)
- always labelled **PRELIMINARY**. The rules were not tuned to improve it

## Limits

- one stock, one split, one training run per model (the report says the same)
- Yahoo data is not a real-time feed and the report's daily candles are only a proxy for the 15-minute / 1-hour design
- no profit is claimed anywhere
