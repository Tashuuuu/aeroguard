"""FastAPI service: forecast, anomalies, counterfactual, cities, benchmark results."""
import os
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config import *
from models.inference import Runtime
from counterfactual import simulator as sim

MODE = os.environ.get("AEROGUARD_MODE", "hourly")
app = FastAPI(title="AeroGuard API", version="1.0")
rt = None


@app.on_event("startup")
def _load():
    global rt
    rt = Runtime(MODE, device="cpu")


class Scenario(BaseModel):
    scenario: str
    start: str
    end: str
    reduction: float = 0.0


@app.get("/health")
def health():
    return {"status": "ok", "mode": MODE, "stations": len(rt.bundle.st)}


@app.get("/stations")
def stations():
    return rt.bundle.st[["station_id", "name", "lat", "lon"]].to_dict("records")


@app.get("/forecast")
def forecast(station_id: int, origin: str | None = None):
    ids = rt.bundle.st.station_id.tolist()
    if station_id not in ids:
        raise HTTPException(404, "unknown station")
    n, t = ids.index(station_id), rt.bundle.times
    s = len(t) - 1 if origin is None else int(t.searchsorted(pd.Timestamp(origin), side="right") - 1)
    s = min(max(s, rt.window - 1), len(t) - 1)
    y = rt.run(np.array([s]))[0, :, n]                                  # (H,P,Q)
    step = pd.Timedelta(hours=1) if MODE == "hourly" else pd.Timedelta(days=1)
    out = {p: [{"time": str(t[s] + step * (h + 1)), **{f"q{int(q * 100):02d}": float(y[h, pi, qi])
                                                      for qi, q in enumerate(QUANTILES)}}
               for h in range(y.shape[0])] for pi, p in enumerate(POLLUTANTS)}
    return {"station_id": station_id, "origin": str(t[s]), "unit": "ug/m3", "forecast": out}


@app.get("/anomalies")
def anomalies(date_from: str | None = None, date_to: str | None = None, only_flagged: bool = True):
    f = ART / f"anomaly_flags_{MODE}.parquet"
    if not f.exists():
        raise HTTPException(404, "run anomaly.detector first")
    d = pd.read_parquet(f)
    if date_from:
        d = d[d.time >= pd.Timestamp(date_from)]
    if date_to:
        d = d[d.time <= pd.Timestamp(date_to)]
    if only_flagged:
        d = d[d.flag]
    d = d.merge(rt.bundle.st[["station_id", "name", "lat", "lon"]], on="station_id")
    d["time"] = d.time.astype(str)
    return d.head(5000).to_dict("records")


@app.post("/counterfactual")
def counterfactual(b: Scenario):
    if b.scenario not in sim.SCENARIOS:
        raise HTTPException(400, f"scenario must be one of {list(sim.SCENARIOS)}")
    return sim.run_scenario(rt, b.scenario, b.start, b.end, b.reduction)


@app.get("/cities")
def cities():
    import json
    f, j = ART / "crosscity_table.csv", ART / "crosscity_results.json"
    if not f.exists():
        raise HTTPException(404, "run crosscity.analysis first")
    return {"table": pd.read_csv(f).to_dict("records"), "stats": json.load(open(j))}


@app.get("/results")
def results():
    f = ART / "results_all.csv"
    if not f.exists():
        raise HTTPException(404, "run evaluate first")
    return pd.read_csv(f).fillna("").to_dict("records")