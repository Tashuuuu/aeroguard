"""Ground PM2.5 / PM10 / NO2 (CPCB via OpenAQ v3). Resumable. Also selects stations -> processed/stations.csv"""
import time
import numpy as np, pandas as pd
from tqdm import tqdm
from config import *
from utils import http_get

BASE = "https://api.openaq.org/v3"
HDR = {"X-API-Key": OPENAQ_API_KEY}


def list_stations():
    bbox = f"{BBOX['west']},{BBOX['south']},{BBOX['east']},{BBOX['north']}"
    rows, page = [], 1
    while True:
        res = http_get(f"{BASE}/locations", HDR, {"bbox": bbox, "limit": 1000, "page": page}).json()["results"]
        for loc in res:
            if not loc.get("isMonitor"):
                continue                                   # reference-grade monitors only (not low-cost sensors)
            sens = {s["parameter"]["name"]: s for s in loc["sensors"] if s["parameter"]["name"] in POLLUTANTS}
            if "pm25" not in sens:
                continue
            row = dict(station_id=loc["id"], name=loc["name"], lat=loc["coordinates"]["latitude"],
                       lon=loc["coordinates"]["longitude"])
            for p in POLLUTANTS:
                row[f"sensor_{p}"] = sens[p]["id"] if p in sens else np.nan
                row[f"units_{p}"] = sens[p]["parameter"].get("units", "") if p in sens else ""
            rows.append(row)
        if len(res) < 1000:
            break
        page += 1
        time.sleep(1)
    return pd.DataFrame(rows).drop_duplicates("station_id")


def fetch_sensor_year(sensor_id, year):
    out = RAW / "openaq" / f"{int(sensor_id)}_{year}.parquet"
    if out.exists():
        return
    out.parent.mkdir(exist_ok=True)
    rows, page = [], 1
    try:
        while True:
            r = http_get(f"{BASE}/sensors/{int(sensor_id)}/hours", HDR,
                         {"datetime_from": f"{year}-01-01T00:00:00Z", "datetime_to": f"{year + 1}-01-01T00:00:00Z",
                          "limit": 1000, "page": page}).json()["results"]
            rows += [(d["period"]["datetimeFrom"]["utc"], d["value"]) for d in r]
            if len(r) < 1000:
                break
            page += 1
            time.sleep(1.05)
    except Exception as e:                                  # 4xx = no data for that year -> store empty marker
        print("  no data", sensor_id, year, str(e)[:60])
    pd.DataFrame(rows, columns=["time_utc", "value"]).to_parquet(out)
    time.sleep(1.05)                                        # stay under 60 requests/min


def assemble_and_select(stations):
    frames = []
    for s in stations.itertuples():
        for p in POLLUTANTS:
            sid = getattr(s, f"sensor_{p}")
            if pd.isna(sid):
                continue
            for y in YEARS:
                f = RAW / "openaq" / f"{int(sid)}_{y}.parquet"
                if f.exists():
                    d = pd.read_parquet(f)
                    if len(d):
                        d["col"], d["v"] = f"{p}|{s.station_id}", d["value"].astype(float)
                        if p == "no2" and str(getattr(s, "units_no2")).lower() == "ppm":
                            d["v"] *= 1880.0                # ppm -> ug/m3 for NO2
                        frames.append(d[["time_utc", "col", "v"]])
    long = pd.concat(frames)
    long["time_utc"] = pd.to_datetime(long["time_utc"], utc=True).dt.tz_convert(None)
    long["time_utc"] = long["time_utc"].dt.floor("h")
    lim = long["col"].str.split("|").str[0].map({"pm25": 1500, "pm10": 2500, "no2": 1000})
    long = long[(long.v > 0) & (long.v < lim)]                          # drop impossible values
    wide = long.pivot_table(index="time_utc", columns="col", values="v", aggfunc="mean")
    wide = wide.reindex(pd.date_range(START, f"{END} 23:00", freq="h"))
    cov = {s: wide[f"pm25|{s}"].notna().mean() if f"pm25|{s}" in wide else 0 for s in stations.station_id}
    keep = stations[stations.station_id.map(cov) >= MIN_COVERAGE].sort_values("station_id").reset_index(drop=True)
    keep["coverage_pm25"] = keep.station_id.map(cov)
    cols = [c for c in wide.columns if int(c.split("|")[1]) in set(keep.station_id)]
    wide[cols].to_parquet(PROC / "ground_hourly.parquet")
    keep.to_csv(PROC / "stations.csv", index=False)
    print(f"kept {len(keep)} / {len(stations)} stations (coverage >= {MIN_COVERAGE})")


if __name__ == "__main__":
    st = list_stations()
    st.to_csv(PROC / "stations_all.csv", index=False)
    print(len(st), "candidate monitors")
    jobs = [(getattr(r, f"sensor_{p}"), y) for r in st.itertuples() for p in POLLUTANTS
            if not pd.isna(getattr(r, f"sensor_{p}")) for y in YEARS]
    for sid, y in tqdm(jobs):
        fetch_sensor_year(sid, y)
    assemble_and_select(st)