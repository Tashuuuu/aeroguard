"""Feature-ablation scenario simulator (NOT formal causal inference) + validation on natural experiments.
All runs use the 'driver-only' mode: ground-reading channels masked, so predictions come from weather,
satellite, fire, calendar and traffic-proxy features -> ablating those features actually moves the output."""
import argparse, json
import numpy as np, pandas as pd
from config import *
from preprocess.dataset import event_origins
from models.inference import Runtime

MID = len(QUANTILES) // 2
SCENARIOS = {
    "no_firecrackers": dict(set={"diwali_flag": 0.0, "festival_flag": 0.0}),
    "no_stubble_fires": dict(set={**{f: 0.0 for f in FIRE}, "stubble_season": 0.0}),
    "no_fires_no_firecrackers": dict(set={**{f: 0.0 for f in FIRE}, "stubble_season": 0.0,
                                          "diwali_flag": 0.0, "festival_flag": 0.0}),
    "reduced_traffic": dict(traffic=True),
}


def apply_scenario(X, names, scen, reduction=0.0):
    X = X.copy()
    for k, v in scen.get("set", {}).items():
        if k in names:
            X[:, :, names.index(k)] = v
    if scen.get("traffic") and "traffic_proxy" in names:
        X[:, :, names.index("traffic_proxy")] *= (1.0 - reduction)
    return X


def driver_only(rt, origins, X_raw=None):
    li = [l - 1 for l in rt.cfg["leads"]]
    return rt.run(origins, X_raw=X_raw, mask_global=True, post=lambda y: y[:, li, :, 0, :])   # (S,L,N,Q) PM2.5


def effect(rt, origins, scen, reduction=0.0):
    bau = driver_only(rt, origins)[..., MID]
    cf = driver_only(rt, origins, apply_scenario(rt.bundle.X, rt.names, scen, reduction))[..., MID]
    return bau, cf


def origins_between(rt, start, end):
    t = rt.bundle.times
    s = np.arange(rt.window - 1, len(t) - 1)
    return s[(t[s + 1] >= pd.Timestamp(start)) & (t[s + 1] < pd.Timestamp(end) + pd.Timedelta(days=1))]


def run_scenario(rt, name, start, end, reduction=0.0):
    o = origins_between(rt, start, end)
    if len(o) == 0:
        raise ValueError("no origins in that period")
    bau, cf = effect(rt, o, SCENARIOS[name], reduction)
    imp = bau - cf
    return dict(scenario=name, start=start, end=end, reduction=reduction,
                baseline_mean=float(bau.mean()), counterfactual_mean=float(cf.mean()),
                abs_impact=float(imp.mean()), pct_impact=float(100 * imp.mean() / max(bau.mean(), 1e-6)),
                per_station={int(i): float(v) for i, v in zip(rt.bundle.st.station_id, imp.mean((0, 1)))})


def validate_event(rt, ev):
    """observed weather-adjusted drop (driver-only BAU vs actual) vs simulated drop for a traffic cut."""
    o = event_origins(rt.bundle.times, rt.window, rt.cfg["horizon"], ev)
    Y = rt.bundle.X[:, :, 0]
    act = np.stack([Y[o + l] for l in rt.cfg["leads"]], 1)
    bau = driver_only(rt, o)[..., MID]
    ok = ~np.isnan(act)
    observed = float(100 * (bau[ok].mean() - act[ok].mean()) / bau[ok].mean())
    grid = np.round(np.arange(0.1, 1.0, 0.1), 1)
    sim = []
    for r in grid:
        b, c = effect(rt, o, SCENARIOS["reduced_traffic"], float(r))
        sim.append(float(100 * (b - c).mean() / b.mean()))
    doc = TRAFFIC_REDUCTION[ev]
    b, c = effect(rt, o, SCENARIOS["reduced_traffic"], doc)
    implied = float(np.interp(observed, sim, grid)) if np.all(np.diff(sim) > 0) else None
    return dict(event=ev, n_origins=int(len(o)), documented_traffic_reduction=doc,
                observed_pm25_drop_pct=observed, simulated_drop_pct_at_documented=float(100 * (b - c).mean() / b.mean()),
                implied_traffic_reduction_to_match_observed=implied,
                sweep=dict(zip(map(float, grid), sim)))


def main(mode):
    rt = Runtime(mode)
    o = origins_between(rt, *SPLITS["test"])[::MODES[mode]["eval_stride"]]
    o = o[o + max(rt.cfg["leads"]) < len(rt.bundle.times)]
    li = [l - 1 for l in rt.cfg["leads"]]
    bau = driver_only(rt, o)[..., MID]
    act = np.stack([rt.bundle.X[o + l, :, 0] for l in rt.cfg["leads"]], 1)
    ok = ~np.isnan(act)
    res = dict(driver_only_test_rmse=float(np.sqrt(np.mean((bau[ok] - act[ok]) ** 2))),
               driver_only_test_mae=float(np.mean(np.abs(bau[ok] - act[ok]))), scenarios=[], validation=[])
    for name, a, b, r in [("no_firecrackers", "2022-10-22", "2022-10-29", 0.0),
                          ("no_stubble_fires", "2022-10-15", "2022-11-30", 0.0),
                          ("no_fires_no_firecrackers", "2022-10-15", "2022-11-30", 0.0),
                          ("reduced_traffic", "2022-11-01", "2022-11-30", 0.25),
                          ("reduced_traffic", "2022-11-01", "2022-11-30", 0.50)]:
        res["scenarios"].append(run_scenario(rt, name, a, b, r))
    for ev in EVENTS:
        res["validation"].append(validate_event(rt, ev))
    json.dump(res, open(ART / f"counterfactual_{mode}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "scenarios"}, indent=2)[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="hourly")
    main(ap.parse_args().mode)