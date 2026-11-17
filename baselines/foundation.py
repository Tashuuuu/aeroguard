"""Pretrained time-series foundation models on PM2.5, evaluated on 100 test origins x all stations.
Chronos-Bolt (zero-shot AND fine-tuned, via AutoGluon) and TimesFM 2.5 (zero-shot)."""
import argparse
import numpy as np, pandas as pd
from config import *
from preprocess.dataset import load_bundle, split_origins
from models.inference import save_preds

N_ORIGINS = 100
CTX = {"hourly": 512, "daily": 192}


def pick(B, cfg):
    o = split_origins(B.times, cfg["window"], cfg["horizon"], "test", stride=cfg["eval_stride"])
    return o[np.linspace(0, len(o) - 1, N_ORIGINS).astype(int)]        # subset of the standard test origins


def make_items(Y, times, origins, ctx):
    rows = []
    for s in origins:
        lo = max(0, s - ctx + 1)
        for n in range(Y.shape[1]):
            v = Y[lo:s + 1, n, 0]
            if np.isfinite(v).sum() >= 24:
                rows.append(pd.DataFrame({"item_id": f"{n}_{s}", "timestamp": times[lo:s + 1], "target": v}))
    return pd.concat(rows, ignore_index=True)


def chronos(mode, finetune):
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    cfg = MODES[mode]; H = cfg["horizon"]; B = load_bundle(mode); Y = B.X[:, :, :3]; N = Y.shape[1]
    o, ctx = pick(B, cfg), CTX[mode]
    tr_end = int((B.times < pd.Timestamp(SPLITS["train"][1]) + pd.Timedelta(days=1)).sum())
    tr = pd.concat([pd.DataFrame({"item_id": str(n), "timestamp": B.times[:tr_end], "target": Y[:tr_end, n, 0]})
                    for n in range(N)], ignore_index=True)
    mk = lambda df: TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")
    hp = {"model_path": "bolt_small", "context_length": ctx}
    if finetune:
        hp.update(fine_tune=True, fine_tune_steps=1000)
    tag = "ft" if finetune else "zs"
    pr = TimeSeriesPredictor(prediction_length=H, target="target", freq=cfg["freq"], quantile_levels=QUANTILES,
                             path=str(ART / f"ag_{mode}_{tag}"), verbosity=1)
    pr.fit(mk(tr), hyperparameters={"Chronos": hp}, time_limit=1800 if finetune else 300, enable_ensemble=False)
    fc = pr.predict(mk(make_items(Y, B.times, o, ctx))).sort_index()
    cols = [str(q) for q in QUANTILES]
    d = {k: g[cols].to_numpy() for k, g in fc.groupby(level="item_id")}
    full = np.full((len(o), H, N, 3, len(QUANTILES)), np.nan, np.float32)
    for si, s in enumerate(o):
        for n in range(N):
            if f"{n}_{s}" in d:
                full[si, :, n, 0, :] = d[f"{n}_{s}"]
    save_preds(PRED / f"{mode}_chronos_{tag}.npz", o, full, QUANTILES, cfg["leads"])


def timesfm_zero_shot(mode):
    import timesfm
    cfg = MODES[mode]; H = cfg["horizon"]; B = load_bundle(mode); Y = B.X[:, :, :3]; N = Y.shape[1]
    o, ctx = pick(B, cfg), CTX[mode]
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=ctx, max_horizon=H, normalize_inputs=True,
                                     use_continuous_quantile_head=True, force_flip_invariance=True,
                                     infer_is_positive=True, fix_quantile_crossing=True))
    keys, inputs = [], []
    for si, s in enumerate(o):
        for n in range(N):
            v = pd.Series(Y[max(0, s - ctx + 1):s + 1, n, 0]).interpolate(limit_direction="both").to_numpy()
            if np.isfinite(v).all():
                keys.append((si, n)); inputs.append(v.astype(np.float32))
    full = np.full((len(o), H, N, 3, 3), np.nan, np.float32)               # quantile levels 0.1, 0.5, 0.9
    for i in range(0, len(inputs), 256):
        _, q = m.forecast(horizon=H, inputs=inputs[i:i + 256])             # q: (b, H, 10) = mean, q10..q90
        for j, (si, n) in enumerate(keys[i:i + 256]):
            full[si, :, n, 0, :] = q[j][:, [1, 5, 9]]
    save_preds(PRED / f"{mode}_timesfm_zs.npz", o, full, [0.1, 0.5, 0.9], cfg["leads"], med_index=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="hourly")
    mode = ap.parse_args().mode
    for fn, args in [(chronos, (mode, False)), (chronos, (mode, True)), (timesfm_zero_shot, (mode,))]:
        try:
            fn(*args)
        except Exception as e:                                             # one failing model must not stop the others
            print("FAILED", fn.__name__, args, "->", repr(e)[:300])