"""Small dashboard: chart with confirmed swings + the latest signal.

    streamlit run app.py
"""
import streamlit as st

import config
import data_loader
import models
import plots
import signals
import swings as sw
import trade_planner as tp

st.set_page_config(page_title="StockSense", layout="wide")
st.title("StockSense: RELIANCE.NS")
st.caption("Research tool for study, not financial advice. Daily candles.")

st.sidebar.header("Settings")
n_candles = st.sidebar.slider("Candles on chart", 60, 400, 150, step=10)
model_name = st.sidebar.radio("LSTM model", ["B", "A"], format_func=lambda k: f"LSTM {k}")
live = st.sidebar.checkbox("Download latest data from Yahoo", value=not config.DATA_FILE.exists())


@st.cache_data(show_spinner="Loading data...")
def get_data(live):
    if live:
        try:
            return data_loader.load_latest()
        except Exception as e:
            st.warning(f"Download failed ({e}), using the cached file")
    return data_loader.load_ohlcv()


df = get_data(live)
swing_tbl = sw.find_swings(df)
levels = sw.swing_levels(df, swing_tbl)
sig = signals.signal_table(df)
last = sig.iloc[-1]

# ---- latest signal ----
st.subheader(f"Latest candle: {df.index[-1].date()}  (close {df['Close'].iloc[-1]:,.2f})")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Indicator signal", last["Signal"], f"score {last['Score']:+d}")

pred = None
try:
    pred = models.predict_latest(model_name, df=df, verbose=False)
    c2.metric(f"LSTM {model_name} predicted {config.HORIZON}-day return", f"{pred * 100:+.2f}%")
except FileNotFoundError:
    c2.metric("LSTM prediction", "no model")
    st.info("No saved model yet. Run  python run_experiment.py  once to train and save it.")

ind = last["Signal"]
stop = levels.iloc[-1]["swing_low" if ind == "BUY" else "swing_high"] if ind in ("BUY", "SELL") else None
if pred is not None:
    decision, why = signals.confirm_signal(pred, ind, stop, df["Close"].iloc[-1])
else:
    decision, why = "NO TRADE", "no LSTM prediction"
c3.metric("Final decision", decision)
c4.write(f"**Why:** {why}")

if decision in ("BUY", "SELL"):
    # next open is not known yet, so the last close is used as the entry here
    plan = tp.make_plan(decision, float(df["Close"].iloc[-1]), float(stop))
    if plan["status"] == "TRADE":
        st.success(f"{decision}: entry about {plan['entry']:.2f} (last close, real entry is the next open), "
                   f"stop {plan['stop']:.2f}, target {plan['target']:.2f}, quantity {plan['quantity']}")
    else:
        st.warning(f"NO TRADE: {plan['reason']}")

# ---- chart ----
chart = plots.plotly_candles(df, swing_tbl, n=n_candles)
try:
    st.plotly_chart(chart, width="stretch")
except TypeError:                      # older streamlit versions
    st.plotly_chart(chart, use_container_width=True)
st.caption("Triangles are confirmed swings (2 candles on each side). Hover to see the day each one was confirmed.")

with st.expander("Last ten indicator signals"):
    st.dataframe(signals.last_signals(df, 10))
