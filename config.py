"""All the fixed settings for StockSense in one place."""
import os
from pathlib import Path

# STOCKSENSE_HOME is only used by the tests so they don't touch the real data/models folders
ROOT = Path(os.environ.get("STOCKSENSE_HOME", Path(__file__).resolve().parent))
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
DATA_FILE = DATA_DIR / "RELIANCE_NS_daily.csv"

# ---- data ----
SYMBOL = "RELIANCE.NS"
START_DATE = "2016-10-28"
END_DATE = "2026-10-29"      # yfinance end date is exclusive, so this keeps 28 Oct 2026

# ---- fixed by the report ----
SEED = 42
HORIZON = 4                  # predict the return over the next 4 candles
LOOKBACK = 64                # LSTM sees the last 64 candles

N_TRAIN = 1706
N_VAL = 363
N_TEST = 367
GAP = 4                      # candles dropped between train/val and val/test

# ---- LSTM (Table 4.3 and Section 8.3) ----
DROPOUT = 0.2                # not given in the report, my choice
MAX_EPOCHS = 50              # not given in the report, early stopping stops it sooner
BATCH_SIZE = 64
PATIENCE = 5
LEARNING_RATE = 0.001

# ---- indicator details the report does not spell out ----
RSI_METHOD = "simple"        # "simple" = rolling mean of gains/losses, "wilder" = Wilder smoothing
BB_DDOF = 1                  # 1 = pandas default std, 0 = numpy default std
# python check_variants.py tries all combinations against the report's signal counts

# ---- signal / trading settings ----
SIGNAL_THRESHOLD = 0.0       # LSTM must be above +x (BUY) or below -x (SELL). 0 = just the sign
SWING_LEFT = 2
SWING_RIGHT = 2
EQUITY = 50_000              # same as Table 8.2
RISK_FRACTION = 0.005        # 0.5% of equity risked per trade
REWARD_RISK = 2.0            # target = entry +/- 2 x risk

# ---- backtest costs (my assumptions, change to match your broker) ----
COST_PER_SIDE = 0.0010       # 0.10% of traded value on each side (brokerage, STT, charges)
SLIPPAGE_PER_SIDE = 0.0005   # 0.05% worse fill on each side
ALLOW_SHORT = True           # SELL trades in the backtest (needs futures / intraday in real life)

# ---- numbers copied from the report. DO NOT EDIT ----
REPORT_N_CANDLES = 2471
REPORT_FIRST_DATE = "2016-10-28"
REPORT_LAST_DATE = "2026-10-28"
REPORT_FIRST_CLOSE = 230.10
REPORT_LAST_CLOSE = 1158.00
REPORT_SIGNAL_ROWS = 2452
REPORT_SIGNAL_COUNTS = {"BUY": 271, "SELL": 325, "HOLD": 1856}
REPORT_TEST_START = "2025-04-30"
REPORT_TEST_END = "2026-10-22"
REPORT_N_SCORED = 304        # 367 test candles minus 63 used as history

# Table 8.1: last ten candles (date, close, score, signal)
REPORT_TABLE_8_1 = [
    ("2026-10-14", 1174.60, -2, "SELL"),
    ("2026-10-15", 1215.00, 0, "HOLD"),
    ("2026-10-16", 1211.20, 0, "HOLD"),
    ("2026-10-19", 1202.00, 0, "HOLD"),
    ("2026-10-21", 1209.00, 0, "HOLD"),
    ("2026-10-22", 1184.50, 0, "HOLD"),
    ("2026-10-23", 1190.00, 0, "HOLD"),
    ("2026-10-26", 1201.80, 0, "HOLD"),
    ("2026-10-27", 1196.20, 0, "HOLD"),
    ("2026-10-28", 1158.00, 1, "HOLD"),
]

# Table 8.3: (MAE, RMSE, directional accuracy %), None = not defined
REPORT_TABLE_8_3 = {
    "Zero-return baseline": (0.01916, 0.02465, None),
    "Random Forest": (0.02223, 0.02832, 50.0),
    "XGBoost": (0.02123, 0.02728, 51.6),
    "LSTM A: OHLCV only": (0.01962, 0.02500, 47.4),
    "LSTM B: OHLCV + indicators": (0.02179, 0.02869, 51.3),
}
