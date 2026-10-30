"""Download, clean and cache the OHLCV data."""
import pandas as pd

import config

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download(start=config.START_DATE, end=config.END_DATE):
    """Daily candles from Yahoo Finance (adjusted prices)."""
    import yfinance as yf

    raw = yf.download(config.SYMBOL, start=start, end=end, interval="1d",
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data for " + config.SYMBOL)
    if isinstance(raw.columns, pd.MultiIndex):      # newer yfinance versions
        raw.columns = raw.columns.get_level_values(0)
    df = raw[COLUMNS].copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    df.index.name = "Date"
    return df


def clean(df):
    """Section 4.2: sort, drop duplicates, drop bad candles. Prints what it removed."""
    df = df.sort_index()

    dup = df.index.duplicated(keep="first")
    if dup.any():
        print(f"[data] removed {dup.sum()} duplicate timestamps")
        df = df[~dup]

    n_nan = df.isna().any(axis=1).sum()
    if n_nan:
        print(f"[data] removed {n_nan} rows with missing values")
        df = df.dropna()

    top = df[["Open", "Close"]].max(axis=1)
    bottom = df[["Open", "Close"]].min(axis=1)
    bad = (df["High"] < top - 1e-6) | (df["Low"] > bottom + 1e-6)
    bad |= (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    bad |= df["Volume"] < 0
    if bad.any():
        print(f"[data] removed {bad.sum()} invalid candles: {list(df.index[bad].strftime('%Y-%m-%d'))}")
        df = df[~bad]
    return df


def load_ohlcv(refresh=False):
    """The fixed date range used for the experiment. Uses the cached csv if it exists."""
    if config.DATA_FILE.exists() and not refresh:
        df = pd.read_csv(config.DATA_FILE, index_col=0, parse_dates=True)
        df.index.name = "Date"
        return df
    df = clean(download())
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATA_FILE)
    print(f"[data] saved {len(df)} candles to {config.DATA_FILE}")
    return df


def load_latest():
    """Full history up to now (for predict_latest). Drops today's candle if the market is still open."""
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    end = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = clean(download(end=end))
    if df.index[-1].date() == now.date() and now.hour < 16:
        df = df.iloc[:-1]            # today's candle is not finished yet
    return df


def data_summary(df):
    return {
        "candles": len(df),
        "first_date": df.index[0].strftime("%Y-%m-%d"),
        "last_date": df.index[-1].strftime("%Y-%m-%d"),
        "first_close": round(float(df["Close"].iloc[0]), 2),
        "last_close": round(float(df["Close"].iloc[-1]), 2),
        "min_close": round(float(df["Close"].min()), 2),
        "max_close": round(float(df["Close"].max()), 2),
    }
