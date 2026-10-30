"""Data split, LSTM A/B, baselines, saving/loading and predict_latest()."""
import json
import random
from functools import lru_cache

import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import config
from data_loader import load_latest, load_ohlcv
from features import FEATURE_SETS, build_features, make_target


def set_seed(seed=config.SEED):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ---------------- split ----------------

def prepare_data(df):
    """Feature table + target, and the train / val / test slices.

    Only rows with a full feature set AND a target are used. The split is counted
    back from the last usable row, so the test set ends on the last candle that has a
    target (22 Oct 2026 in the report). Gap of 4 candles before val and before test.
    """
    feats = build_features(df)
    y = make_target(df)
    ok = feats[FEATURE_SETS["B"]].notna().all(axis=1) & y.notna()
    feats, y = feats[ok], y[ok]

    n = len(feats)
    need = config.N_TRAIN + config.N_VAL + config.N_TEST + 2 * config.GAP
    if n < need:
        raise ValueError(f"need at least {need} usable candles, only have {n}")

    test = slice(n - config.N_TEST, n)
    val_end = test.start - config.GAP
    val = slice(val_end - config.N_VAL, val_end)
    train_end = val.start - config.GAP
    train = slice(train_end - config.N_TRAIN, train_end)
    return {"feats": feats, "y": y, "train": train, "val": val, "test": test}


def split_info(data):
    idx = data["feats"].index
    out = {}
    for name in ("train", "val", "test"):
        s = data[name]
        out[name] = {"n": s.stop - s.start,
                     "start": idx[s.start].strftime("%Y-%m-%d"),
                     "end": idx[s.stop - 1].strftime("%Y-%m-%d")}
    return out


def make_windows(X, y, lookback=config.LOOKBACK):
    """Window ending at row i uses rows i-lookback+1 ... i. Windows stay inside one part,
    so the first lookback-1 rows of a part are only used as history."""
    w = sliding_window_view(X, lookback, axis=0)        # (n-lb+1, F, lb)
    w = np.transpose(w, (0, 2, 1))
    pos = np.arange(lookback - 1, len(X))
    return w, y[pos], pos


# ---------------- LSTM ----------------

def build_lstm(n_features):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=(config.LOOKBACK, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(config.DROPOUT),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE), loss="mse")
    return model


def train_lstm(data, set_name):
    """Train LSTM on feature set 'A' or 'B'. Returns a dict with the model, scaler and test predictions."""
    from tensorflow import keras

    cols = FEATURE_SETS[set_name]
    X = data["feats"][cols].values
    y = data["y"].values
    tr, va, te = data["train"], data["val"], data["test"]

    scaler = StandardScaler().fit(X[tr])                 # train period only
    Xs = scaler.transform(X)

    Xtr, ytr, _ = make_windows(Xs[tr], y[tr])
    Xva, yva, _ = make_windows(Xs[va], y[va])
    Xte, yte, pos = make_windows(Xs[te], y[te])

    set_seed()
    model = build_lstm(len(cols))
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE,
                                         restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=config.MAX_EPOCHS,
                     batch_size=config.BATCH_SIZE, callbacks=[stop], verbose=0)
    pred = model.predict(Xte, verbose=0).ravel()

    dates = data["feats"].index[te][pos]
    return {"name": set_name, "model": model, "scaler": scaler, "features": cols,
            "dates": dates, "y_true": yte, "pred": pred,
            "epochs": len(hist.history["loss"]),
            "best_val_loss": float(min(hist.history["val_loss"])),
            "trained_until": data["feats"].index[tr.stop - 1].strftime("%Y-%m-%d")}


# ---------------- baselines ----------------

def run_baselines(data):
    """Zero-return, Random Forest and XGBoost, scored on the same candles as the LSTMs."""
    from xgboost import XGBRegressor

    cols = FEATURE_SETS["B"]
    X = data["feats"][cols].values
    y = data["y"].values
    tr, te = data["train"], data["test"]
    scored = slice(te.start + config.LOOKBACK - 1, te.stop)
    y_true = y[scored]

    rf = RandomForestRegressor(n_estimators=300, random_state=config.SEED, n_jobs=-1)
    rf.fit(X[tr], y[tr])
    xgb = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=config.SEED)
    xgb.fit(X[tr], y[tr])

    return {"Zero-return baseline": np.zeros(len(y_true)),
            "Random Forest": rf.predict(X[scored]),
            "XGBoost": xgb.predict(X[scored])}, y_true


def evaluate(y_true, y_pred, has_direction=True):
    err = y_true - y_pred
    out = {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2)))}
    out["DirAcc"] = float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100) if has_direction else None
    return out


# ---------------- save / load ----------------

def save_lstm(res):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    n = res["name"]
    res["model"].save(config.MODEL_DIR / f"lstm_{n}.keras")
    joblib.dump(res["scaler"], config.MODEL_DIR / f"scaler_{n}.joblib")
    meta = {"features": res["features"], "lookback": config.LOOKBACK, "horizon": config.HORIZON,
            "seed": config.SEED, "symbol": config.SYMBOL, "trained_until": res["trained_until"],
            "epochs_run": res["epochs"]}
    (config.MODEL_DIR / f"meta_{n}.json").write_text(json.dumps(meta, indent=2))


@lru_cache(maxsize=4)
def load_lstm(name="B"):
    """Loads the saved model and scaler. No retraining."""
    from tensorflow import keras

    path = config.MODEL_DIR / f"lstm_{name}.keras"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run  python run_experiment.py  first to train and save it.")
    model = keras.models.load_model(path)
    scaler = joblib.load(config.MODEL_DIR / f"scaler_{name}.joblib")
    meta = json.loads((config.MODEL_DIR / f"meta_{name}.json").read_text())
    return model, scaler, meta


def predict_latest(name="B", df=None, verbose=True):
    """Predicted 4-day return for the newest candle, from the saved model.
    df can be passed in (e.g. to test on an old date); if not, new data is downloaded."""
    model, scaler, meta = load_lstm(name)
    if df is None:
        try:
            df = load_latest()
        except Exception as e:
            print(f"[predict] could not download new data ({e}), using the cached csv instead")
            df = load_ohlcv()

    feats = build_features(df)[meta["features"]]
    window = feats.iloc[-meta["lookback"]:]
    if len(window) < meta["lookback"] or window.isna().any().any():
        raise ValueError("not enough clean candles to build the 64-candle window")

    x = scaler.transform(window.values)[None, :, :]
    pred = float(model.predict(x, verbose=0)[0, 0])

    if verbose:
        date = df.index[-1].strftime("%Y-%m-%d")
        close = float(df["Close"].iloc[-1])
        print(f"{config.SYMBOL}  newest candle {date}  close {close:.2f}")
        print(f"LSTM {name}: predicted {meta['horizon']}-day return = {pred * 100:+.2f}%")
        print("(research output only, not financial advice)")
    return pred


if __name__ == "__main__":
    predict_latest()
