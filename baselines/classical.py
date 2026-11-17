"""Non-deep floor: persistence and XGBoost (same features, same test origins)."""
import argparse
import numpy as np, pandas as pd, torch, xgboost as xgb
from config import *
from preprocess.dataset import load_bundle, split_origins
from models.inference import save_point

LAGS = {"hourly": [1, 3, 6, 12, 24], "daily": [1, 2, 3, 7]}


def tab(X, origins, lags, lat, lon):
    """Tabular rows (one per origin x station): all features now + lagged ground values + coordinates."""
    cur = X[origins]
    lag = [X[origins - l][:, :, :3] for l in lags]
    S, N, _ = cur.shape
    geo = np.broadcast_to(np.stack([lat, lon], -1)[None], (S, N, 2))
    return np.concatenate([cur] + lag + [geo], -1).reshape(S * N, -1)


def main(mode):
    cfg = MODES[mode]; W, H, L = cfg["window"], cfg["horizon"], cfg["leads"]
    B = load_bundle(mode); X, Y = B.X, B.X[:, :, :3]
    o_tr = split_origins(B.times, W, H, "train", stride=cfg["eval_stride"])
    o_va = split_origins(B.times, W, H, "val", stride=cfg["eval_stride"])
    o_te = split_origins(B.times, W, H, "test", stride=cfg["eval_stride"])
    S, N = len(o_te), X.shape[1]

    # ---- persistence: carry the last observed value forward ----
    Yff = pd.DataFrame(Y.reshape(len(Y), -1)).ffill().to_numpy().reshape(Y.shape)
    pers = np.repeat(Yff[o_te][:, None], len(L), 1)                          # (S,L,N,P)
    save_point(PRED / f"{mode}_persistence.npz", o_te, pers, L)

    # ---- XGBoost: one model per (pollutant, lead) ----
    lat, lon = B.st.lat.to_numpy(), B.st.lon.to_numpy()
    Xtr, Xva, Xte = [tab(X, o, LAGS[mode], lat, lon) for o in (o_tr, o_va, o_te)]
    pred = np.full((S, len(L), N, 3), np.nan, np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for p in range(3):
        for li, l in enumerate(L):
            ytr, yva = Y[o_tr + l, :, p].reshape(-1), Y[o_va + l, :, p].reshape(-1)
            kt, kv = ~np.isnan(ytr), ~np.isnan(yva)
            m = xgb.XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", device=dev,
                                 early_stopping_rounds=30, n_jobs=-1)
            m.fit(Xtr[kt], ytr[kt], eval_set=[(Xva[kv], yva[kv])], verbose=False)
            pred[:, li, :, p] = m.predict(Xte).reshape(S, N)
            print(f"xgb pollutant {POLLUTANTS[p]} lead {l} best_iter {m.best_iteration}")
    save_point(PRED / f"{mode}_xgboost.npz", o_te, pred, L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="hourly")
    main(ap.parse_args().mode)