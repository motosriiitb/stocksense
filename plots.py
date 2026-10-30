"""Charts: Fig. 8.1, swing chart, equity curve and the plotly candlestick used by the app."""
import matplotlib.pyplot as plt
import numpy as np

from features import add_indicators


def fig_8_1(df, n=250, path=None):
    """Technical indicators of RELIANCE.NS, last n candles: EMA, MACD, RSI, Bollinger Bands."""
    d = add_indicators(df).tail(n)
    x = np.arange(len(d))
    step = max(len(d) // 8, 1)
    ticks = x[::step]
    labels = d.index[::step].strftime("%b %y")

    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    ax[0].plot(x, d["Close"], label="Close", color="black", lw=1.2)
    ax[0].plot(x, d["ema12"], label="EMA12", color="tab:blue", lw=1)
    ax[0].plot(x, d["ema26"], label="EMA26", color="tab:orange", lw=1)
    ax[0].set_title("(a) EMA")
    ax[0].legend(loc="upper left")

    ax[1].plot(x, d["macd"], label="MACD", color="tab:blue", lw=1)
    ax[1].plot(x, d["macd_signal"], label="Signal", color="tab:red", lw=1)
    colors = np.where(d["macd_hist"] >= 0, "tab:green", "tab:red")
    ax[1].bar(x, d["macd_hist"], color=colors, alpha=0.5, label="Histogram")
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].set_title("(b) MACD")
    ax[1].legend(loc="upper left")

    ax[2].plot(x, d["rsi"], color="tab:purple", lw=1)
    ax[2].axhline(70, color="tab:red", ls="--", lw=0.8)
    ax[2].axhline(30, color="tab:green", ls="--", lw=0.8)
    ax[2].set_ylim(0, 100)
    ax[2].set_title("(c) RSI (14)")

    ax[3].plot(x, d["Close"], label="Close", color="black", lw=1)
    ax[3].plot(x, d["bb_upper"], label="Upper", color="tab:red", lw=0.8)
    ax[3].plot(x, d["bb_mid"], label="SMA20", color="tab:blue", lw=0.8)
    ax[3].plot(x, d["bb_lower"], label="Lower", color="tab:green", lw=0.8)
    ax[3].fill_between(x, d["bb_lower"], d["bb_upper"], color="grey", alpha=0.12)
    ax[3].set_title("(d) Bollinger Bands (20, 2)")
    ax[3].legend(loc="upper left")

    ax[3].set_xticks(ticks)
    ax[3].set_xticklabels(labels)
    fig.suptitle(f"Fig. 8.1: Technical indicators of RELIANCE.NS (daily, last {len(d)} candles)", y=0.995)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def _draw_candles(ax, d):
    x = np.arange(len(d))
    up = d["Close"] >= d["Open"]
    colors = np.where(up, "tab:green", "tab:red")
    ax.vlines(x, d["Low"], d["High"], color=colors, lw=0.8)
    height = (d["Close"] - d["Open"]).abs().replace(0, 0.01)
    ax.bar(x, height, bottom=d[["Open", "Close"]].min(axis=1), color=colors, width=0.6)
    return x


def swing_chart(df, swings, n=120, path=None):
    """Candles with the confirmed swing highs/lows marked at their pivot candle."""
    d = df.tail(n)
    first = df.index.get_loc(d.index[0])
    s = swings[(swings["pivot_idx"] >= first)]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    _draw_candles(ax, d)
    pad = (d["High"].max() - d["Low"].min()) * 0.02
    hi = s[s["kind"] == "HIGH"]
    lo = s[s["kind"] == "LOW"]
    ax.scatter(hi["pivot_idx"] - first, hi["price"] + pad, marker="v", color="darkred", s=45,
               label="confirmed swing high", zorder=3)
    ax.scatter(lo["pivot_idx"] - first, lo["price"] - pad, marker="^", color="darkgreen", s=45,
               label="confirmed swing low", zorder=3)
    step = max(len(d) // 8, 1)
    ax.set_xticks(np.arange(len(d))[::step])
    ax.set_xticklabels(d.index[::step].strftime("%d %b %y"))
    ax.set_title("RELIANCE.NS daily candles with confirmed swings (2 candles on each side)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def equity_chart(curve, path=None, title="PRELIMINARY backtest equity curve"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(curve.index, curve.values, color="tab:blue")
    ax.set_title(title)
    ax.set_ylabel("Equity (Rs)")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plotly_candles(df, swings, n=150):
    """Plotly candlestick with EMA12/26 and confirmed swings (used by app.py)."""
    import plotly.graph_objects as go

    d = add_indicators(df).tail(n)
    first = df.index.get_loc(d.index[0])
    s = swings[swings["pivot_idx"] >= first]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                                 close=d["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=d.index, y=d["ema12"], name="EMA12", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=d.index, y=d["ema26"], name="EMA26", line=dict(width=1)))
    hi = s[s["kind"] == "HIGH"]
    lo = s[s["kind"] == "LOW"]
    fig.add_trace(go.Scatter(x=hi["pivot_time"], y=hi["price"], mode="markers", name="swing high",
                             marker=dict(symbol="triangle-down", size=10, color="darkred"),
                             customdata=hi["confirm_time"].dt.strftime("%Y-%m-%d"),
                             hovertemplate="swing high %{y:.2f}<br>confirmed %{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=lo["pivot_time"], y=lo["price"], mode="markers", name="swing low",
                             marker=dict(symbol="triangle-up", size=10, color="darkgreen"),
                             customdata=lo["confirm_time"].dt.strftime("%Y-%m-%d"),
                             hovertemplate="swing low %{y:.2f}<br>confirmed %{customdata}<extra></extra>"))
    fig.update_layout(xaxis_rangeslider_visible=False, height=560, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h"))
    # hide weekends / holidays so the candles are not spread out
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig
