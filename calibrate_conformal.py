"""Conformalised quantile regression (CQR).
Fits one interval-widening constant per lead time on the VALIDATION split, then measures coverage on the
untouched TEST split. Usage: python calibrate_conformal.py --mode hourly   (or daily)"""
import argparse, json
import numpy as np
from config import *
from preprocess.dataset import split_origins
from models.inference import Runtime


def main(mode, target=0.90):
    rt, cfg = Runtime(mode), MODES[mode]
    W, H, stride = cfg["window"], cfg["horizon"], cfg["eval_stride"]
    leads, li = cfg["leads"], [l - 1 for l in cfg["leads"]]
    Y = rt.bundle.X[:, :, 0]

    def get(split):
        o = split_origins(rt.bundle.times, W, H, split, stride=stride)
        q = np.sort(rt.run(o, post=lambda y: y[:, li, :, 0, :]), axis=-1)       # (S,L,N,Q)
        act = np.stack([Y[o + l] for l in leads], 1)                            # (S,L,N)
        return q, act

    qv, av = get("val")
    qt, at = get("test")
    out = {}
    for k, l in enumerate(leads):
        ok = ~np.isnan(av[:, k])
        s = np.maximum(qv[:, k, :, 0] - av[:, k], av[:, k] - qv[:, k, :, -1])[ok]
        qhat = float(np.quantile(s, min(1.0, target * (1 + 1 / len(s)))))
        okt = ~np.isnan(at[:, k])
        lo, hi = qt[:, k, :, 0], qt[:, k, :, -1]
        c0 = float(((at[:, k] >= lo) & (at[:, k] <= hi))[okt].mean())
        c1 = float(((at[:, k] >= lo - qhat) & (at[:, k] <= hi + qhat))[okt].mean())
        w0, w1 = float((hi - lo)[okt].mean()), float((hi - lo + 2 * qhat)[okt].mean())
        out[f"lead_{l}"] = dict(qhat=qhat, coverage_before=c0, coverage_after=c1,
                                width_before=w0, width_after=w1, n_val=int(len(s)), n_test=int(okt.sum()))
        print(f"lead {l:2d}  qhat {qhat:6.1f}  coverage {c0:.3f} -> {c1:.3f} (target {target})  width {w0:.0f} -> {w1:.0f}")
    json.dump(out, open(ART / f"conformal_{mode}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="hourly")
    main(ap.parse_args().mode)