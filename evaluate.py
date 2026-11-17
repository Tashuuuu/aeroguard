"""Accuracy (RMSE/MAE), calibration (interval coverage), benchmark across all methods and both pipelines."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from preprocess.dataset import load_bundle


def load(path):
    d = np.load(path)
    return dict(o=d["origins"], pred=d["pred"], q=d["q"] if "q" in d.files else None,
                ql=d["qlevels"] if "qlevels" in d.files and len(d["qlevels"]) else None, leads=d["leads"])


def score(name, mode, m, Y, common=None):
    sel = np.ones(len(m["o"]), bool) if common is None else np.isin(m["o"], common)
    o, pred = m["o"][sel], m["pred"][sel]
    tr = np.stack([Y[o + l] for l in m["leads"]], 1)                    # (S,L,N,P)
    rows = []
    for pi, p in enumerate(POLLUTANTS):
        for li, l in enumerate(m["leads"]):
            e = pred[:, li, :, pi] - tr[:, li, :, pi]
            ok = ~np.isnan(e)
            if ok.sum() == 0:
                continue
            row = dict(mode=mode, method=name, pollutant=p, lead=int(l), n=int(ok.sum()),
                       rmse=float(np.sqrt(np.mean(e[ok] ** 2))), mae=float(np.mean(np.abs(e[ok]))))
            if m["q"] is not None:
                q = m["q"][sel][:, li, :, pi, :]
                lo, hi, t = q[..., 0], q[..., -1], tr[:, li, :, pi]
                v = ~np.isnan(t) & ~np.isnan(lo) & ~np.isnan(hi)
                row["nominal_cov"] = float(m["ql"][-1] - m["ql"][0])
                row["empirical_cov"] = float(((t >= lo) & (t <= hi))[v].mean()) if v.any() else np.nan
                row["interval_width"] = float((hi - lo)[v].mean()) if v.any() else np.nan
            rows.append(row)
    return rows


def main():
    allrows = []
    for mode in MODES:
        files = sorted(glob.glob(str(PRED / f"{mode}_*.npz")))
        if not files:
            continue
        Y = load_bundle(mode).X[:, :, :3]
        M = {Path(f).stem[len(mode) + 1:]: load(f) for f in files}
        for n, m in M.items():                                           # full test origins for every method
            if not n.startswith(("chronos", "timesfm")):
                allrows += [dict(r, subset="full_test") for r in score(n, mode, m, Y)]
        fm = [m["o"] for n, m in M.items() if n.startswith(("chronos", "timesfm"))]
        if fm:                                                           # common origins incl. foundation models
            common = fm[0]
            for o in fm[1:]:
                common = np.intersect1d(common, o)
            for n, m in M.items():
                allrows += [dict(r, subset="fm_origins") for r in score(n, mode, m, Y, common)]
    df = pd.DataFrame(allrows)
    df.to_csv(ART / "results_all.csv", index=False)
    pm = df[df.pollutant == "pm25"].groupby(["subset", "mode", "method"])[["rmse", "mae"]].mean().round(2)
    print(pm.to_string())
    cal = df[(df.pollutant == "pm25") & df.empirical_cov.notna()].groupby(["subset", "mode", "method"])[
        ["nominal_cov", "empirical_cov", "interval_width"]].mean().round(3)
    print("\nCALIBRATION (PM2.5)\n", cal.to_string())
    fig, ax = plt.subplots(figsize=(9, 4))
    pm.loc["full_test"]["rmse"].unstack(0).plot.bar(ax=ax)
    ax.set_ylabel("PM2.5 RMSE (ug/m3), mean over leads"); fig.savefig(ART / "rmse_by_method.png", dpi=150, bbox_inches="tight")
    for f in ["anomaly_metrics_hourly.json", "counterfactual_hourly.json", "crosscity_results.json"]:
        if (ART / f).exists():
            print("\n==", f, "==\n", json.dumps(json.load(open(ART / f)), indent=1)[:1500])


if __name__ == "__main__":
    main()