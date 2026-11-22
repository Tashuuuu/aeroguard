"""Tamper-flagging: expected PM2.5 from neighbours + satellite/weather ONLY (station's own readings masked),
compared with the station's actual reading. Sustained, significant residual => flag. Validated by injecting distortions.
Residuals are scaled by the model's own predicted spread; scoring uses only hours with a real reading."""
import argparse, json
import numpy as np, pandas as pd
from config import *
from preprocess.dataset import split_origins
from models.inference import Runtime
from graphs.build_graph import knn_neighbours, neighbour_unit_vectors, nbr_feature

MID = len(QUANTILES) // 2


def expected(rt, origins, X_raw=None):
    """-> (S,N,Q): next-step PM2.5 quantiles for every station, computed with that station's own channels masked."""
    N = len(rt.bundle.st)
    out = np.zeros((len(origins), N, len(QUANTILES)), np.float32)
    for n in range(N):
        out[:, n, :] = rt.run(origins, X_raw=X_raw, mask_node=n, post=lambda y, n=n: y[:, 0, n, 0, :])
    return out


def zbar_of(exp, actual, K):
    """Rolling mean of residuals / predicted spread. spread = (q95 - q05) / 3.29 (90% interval = +-1.645 sd)."""
    spread = np.maximum((exp[:, :, -1] - exp[:, :, 0]) / 3.29, 1.0)
    z = (actual - exp[:, :, MID]) / spread
    return pd.DataFrame(z).rolling(K, min_periods=max(2, K // 2)).mean().to_numpy()


def inject(rt, X, o_te, rng):
    """Inject controlled distortions into PM2.5 of the held-out test period (only where the station really reports)."""
    names, mode = rt.names, rt.mode
    X, N, seg = X.copy(), X.shape[1], ANOM["seg"][mode]
    lab, events = np.zeros(X.shape[:2], bool), []
    lo, hi = o_te[0] + 1, o_te[-1] + 1 - seg
    for kind, mag in ANOM["kinds"]:
        for _ in range(ANOM["per_type"]):
            for _try in range(500):                       # find a segment with >= 50% real readings
                n, s = int(rng.integers(N)), int(rng.integers(lo, hi))
                if np.isfinite(X[s:s + seg, n, 0]).mean() >= 0.5:
                    break
            v = X[s:s + seg, n, 0]
            X[s:s + seg, n, 0] = np.clip(v * mag if kind == "scale" else v + mag, 0, None)
            lab[s:s + seg, n] = True
            events.append(dict(kind=f"{kind}{mag:+g}", station=n, start=s, end=s + seg))
    rm = MODES[mode]["rm"]
    for n in {e["station"] for e in events}:                               # recompute derived own-features
        X[:, n, names.index("pm25_rm")] = pd.Series(X[:, n, 0]).rolling(rm, min_periods=max(1, rm // 4)).mean().to_numpy()
    st = rt.bundle.st
    idx, dist = knn_neighbours(st.lat.to_numpy(), st.lon.to_numpy(), KNN)
    E = neighbour_unit_vectors(st.lat.to_numpy(), st.lon.to_numpy(), idx)
    X[:, :, names.index("nbr_pm25")] = np.nan_to_num(
        nbr_feature(X[:, :, 0], X[:, :, names.index("u10")], X[:, :, names.index("v10")], idx, E, dist))
    return X, lab, events


def main(mode):
    rt, cfg = Runtime(mode), MODES[mode]
    B, K = rt.bundle, ANOM["roll"][mode]
    W, H = cfg["window"], cfg["horizon"]
    Y = B.X[:, :, 0]
    o_va = split_origins(B.times, W, H, "val", stride=1)
    o_te = split_origins(B.times, W, H, "test", stride=1)

    # 1) calibrate on clean validation data: global threshold on |rolling z| at observed hours only
    exp_va = expected(rt, o_va)
    act_va = Y[o_va + 1]
    zb_va = np.abs(zbar_of(exp_va, act_va, K))
    tau = float(np.nanquantile(np.where(~np.isnan(act_va), zb_va, np.nan), 1 - ANOM["fpr"]))
    print("threshold tau =", round(tau, 3))

    # 2) clean test: real flags + false-positive rate (observed hours only)
    exp_te = expected(rt, o_te)
    act_te = Y[o_te + 1]
    ok_te = ~np.isnan(act_te)
    zb = zbar_of(exp_te, act_te, K)
    flag = (np.abs(zb) > tau) & ok_te
    ids = B.st.station_id.to_numpy()
    T1 = B.times[o_te + 1]
    pd.DataFrame({"time": np.repeat(T1.values, len(ids)), "station_id": np.tile(ids, len(T1)),
                  "actual": act_te.reshape(-1), "exp_q05": exp_te[:, :, 0].reshape(-1),
                  "exp_q50": exp_te[:, :, MID].reshape(-1), "exp_q95": exp_te[:, :, -1].reshape(-1),
                  "z_bar": zb.reshape(-1), "flag": flag.reshape(-1)}).to_parquet(ART / f"anomaly_flags_{mode}.parquet")

    # 3) injected distortions on test -> precision / recall (observed hours only)
    rng = np.random.default_rng(0)
    X_inj, lab, events = inject(rt, B.X, o_te, rng)
    exp_inj = expected(rt, o_te, X_raw=X_inj)
    act_i = X_inj[o_te + 1, :, 0]
    obs_i = ~np.isnan(act_i)
    zb_i = np.abs(zbar_of(exp_inj, act_i, K))
    flag_i = (zb_i > tau) & obs_i
    L = lab[o_te + 1] & obs_i
    tp, fp, fn = int((flag_i & L).sum()), int((flag_i & ~L).sum()), int((~flag_i & L).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    per_kind = {}
    for e in events:
        rows = np.where((o_te + 1 >= e["start"]) & (o_te + 1 < e["end"]))[0]
        hit = flag_i[rows, e["station"]]
        zz = zb_i[rows, e["station"]]
        per_kind.setdefault(e["kind"], []).append(
            (bool(hit.any()), int(np.argmax(hit)) if hit.any() else None,
             float(np.nanmax(zz)) if np.isfinite(zz).any() else np.nan))
    res = dict(mode=mode, tau=tau, rolling_window=K, n_events=len(events),
               clean_test_flag_rate=float(flag[ok_te].mean()),
               timestep_precision=prec, timestep_recall=rec, timestep_f1=2 * prec * rec / max(prec + rec, 1e-9),
               event_recall={k: float(np.mean([h for h, _, _ in v])) for k, v in per_kind.items()},
               median_detection_delay_steps={k: (float(np.median([d for h, d, _ in v if h])) if any(h for h, _, _ in v) else None)
                                              for k, v in per_kind.items()},
               median_peak_abs_zbar={k: float(np.nanmedian([z for _, _, z in v])) for k, v in per_kind.items()})
    json.dump(res, open(ART / f"anomaly_metrics_{mode}.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="hourly")
    main(ap.parse_args().mode)