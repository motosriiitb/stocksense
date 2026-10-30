"""Made-up candles, only for testing the code. NOT real RELIANCE data."""
import numpy as np
import pandas as pd


def make_fake_candles(n=2471, seed=0, end="2026-10-28"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=n)
    ret = rng.normal(0.0004, 0.014, n)
    close = 230 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(5_000_000, 20_000_000, n).astype(float)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
                      index=dates)
    df.index.name = "Date"
    return df
