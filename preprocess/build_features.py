"""Align ground + ERA5 + AOD + Sentinel-5P + fires onto station x time grid, engineer features, save compact Parquet."""
import argparse
import numpy as np, pandas as pd, xarray as xr
from sklearn.neighbors import BallTree
from config import *
from utils import open_era5
from graphs.build_graph import knn_neighbours, neighbour_unit_vectors, nbr_feature

DATES = pd.date_range(START, END, freq="D")


# ---------------- loaders ----------------
def load_ground(st, idx):
    g = pd.read_parquet(PROC / "ground_hourly.parquet").reindex(idx)
    T = len(idx)
    return {p: np.stack([g[f"{p}|{s}"].to_numpy() if f"{p}|{s}" in g.columns else np.full(T, np.nan)
                         for s in st.station_id], 1) for p in POLLUTANTS}


def load_era5(st, idx):
    lat = xr.DataArray(st.lat.values, dims="station")
    lon = xr.DataArray(st.lon.values, dims="station")
    parts = [open_era5(f).sortby("latitude").sortby("longitude").interp(latitude=lat, longitude=lon).load()
             for f in sorted((RAW / "era5").glob("era5_*.nc"))]
    ds = xr.concat(parts, "time").sortby("time").reindex(time=idx)
    g = lambda v: pd.DataFrame(ds[v].transpose("time", "station").values).ffill().bfill().to_numpy()
    t, td = g("t2m") - 273.15, g("d2m") - 273.15
    es = lambda x: 6.112 * np.exp(17.62 * x / (243.12 + x))
    return dict(u10=g("u10"), v10=g("v10"), t2m=t, rh=np.clip(100 * es(td) / es(t), 0, 100),
                sp=g("sp") / 100.0, blh=g("blh"), tp=g("tp") * 1000.0)


def load_aod(st):
    f = next((RAW / "aod").glob("*results*.csv"))
    df = pd.read_csv(f)
    cols = [c for c in df.columns if "Optical_Depth_055" in c and "QA" not in c and "Uncertainty" not in c]
    df["aod"] = df[cols].where(lambda x: (x >= 0) & (x <= 5)).mean(axis=1)    # mean over Terra/Aqua passes
    pv = df.pivot_table(index=pd.to_datetime(df["Date"]), columns=df["ID"].astype(str), values="aod")
    return pd.DataFrame({s: pv[str(s)].reindex(DATES) if str(s) in pv else np.nan for s in st.station_id}, index=DATES)


def load_s5p(st, var):
    out = pd.DataFrame(np.nan, index=DATES, columns=st.station_id)
    for s in st.station_id:
        fs = sorted((RAW / "s5p").glob(f"{s}_{var}_*.parquet"))
        d = pd.concat([pd.read_parquet(x) for x in fs]) if fs else pd.DataFrame(columns=["date", "mean"])
        if len(d):
            d["date"] = pd.to_datetime(d["date"], utc=True).dt.tz_convert(None).dt.normalize()
            out[s] = d.drop_duplicates("date").set_index("date")["mean"].reindex(DATES)
    return out


def load_fire(st):
    f = pd.read_parquet(RAW / "firms.parquet")
    f["date"] = pd.to_datetime(f["acq_date"])
    out = {r: pd.DataFrame(0.0, index=DATES, columns=st.station_id) for r in FIRE_RADII_KM}
    sp = np.radians(st[["lat", "lon"]].to_numpy())
    for d, g in f.groupby("date"):
        if d not in DATES:
            continue
        tree = BallTree(np.radians(g[["latitude", "longitude"]].to_numpy()), metric="haversine")
        for r in FIRE_RADII_KM:
            out[r].loc[d] = tree.query_radius(sp, r=r / 6371.0, count_only=True)
    return out


# ---------------- helpers ----------------
def to_hourly(daily, idx):
    """Daily satellite value -> placed at ~noon IST, linearly interpolated to hourly (max gap 48 h)."""
    h = pd.DataFrame(np.nan, index=idx, columns=daily.columns)
    stamp = daily.index + pd.Timedelta(hours=6)
    ok = stamp.isin(idx)
    h.loc[stamp[ok]] = daily.loc[daily.index[ok]].to_numpy()
    return h.interpolate(method="time", limit=48, limit_area="inside").to_numpy()


def daily_agg(arr, idx, how="mean", min_count=16):
    df = pd.DataFrame(arr, index=idx + IST)
    g = df.resample("D")
    out = g.sum() if how == "sum" else g.mean()
    out[g.count() < min_count] = np.nan
    return out.reindex(DATES).to_numpy()


def calendar(ci, daily):
    hr, dow, mon, day = ci.hour.values, ci.dayofweek.values, ci.month.values, ci.day.values
    dates = ci.normalize().values
    dw = pd.to_datetime(DIWALI)
    diw = np.isin(dates, dw.values) | np.isin(dates, (dw + pd.Timedelta(days=1)).values)
    fest = np.isin(dates, pd.to_datetime(OTHER_FESTIVALS).values)
    stub = ((mon == 10) & (day >= 15)) | (mon == 11)
    traffic = (np.mean(HOUR_TRAFFIC) if daily else np.array(HOUR_TRAFFIC)[hr]) * np.array(WEEK_TRAFFIC)[dow]
    two = 2 * np.pi
    return dict(diwali_flag=diw, festival_flag=fest, stubble_season=stub, traffic_proxy=traffic,
                hr_sin=np.sin(two * hr / 24), hr_cos=np.cos(two * hr / 24), dow_sin=np.sin(two * dow / 7),
                dow_cos=np.cos(two * dow / 7), mon_sin=np.sin(two * mon / 12), mon_cos=np.cos(two * mon / 12),
                is_weekend=dow >= 5)


def main(mode):
    cfg = MODES[mode]
    st = pd.read_csv(PROC / "stations.csv").sort_values("station_id").reset_index(drop=True)
    N = len(st)
    idx = pd.date_range(START, f"{END} 23:00", freq="h")                  # UTC hourly
    gr, wx = load_ground(st, idx), load_era5(st, idx)
    sat = {"aod": load_aod(st), "uvai": load_s5p(st, "uvai"), "no2_trop": load_s5p(st, "no2_trop")}
    fire = load_fire(st)
    lat, lon = st.lat.to_numpy(), st.lon.to_numpy()
    nb_idx, nb_dist = knn_neighbours(lat, lon, KNN)
    E = neighbour_unit_vectors(lat, lon, nb_idx)

    if mode == "hourly":
        F = {**gr, **wx}
        for k in sat:
            F[k] = to_hourly(sat[k], idx)                                 # hourly-interpolated variant
        for r in FIRE_RADII_KM:
            F[f"fire_{r}"] = fire[r].reindex((idx + IST).normalize()).to_numpy()
        index, cal_idx = idx, idx + IST
    else:                                                                 # daily-aggregated variant
        F = {p: daily_agg(gr[p], idx) for p in POLLUTANTS}
        F.update({k: daily_agg(wx[k], idx, "sum" if k == "tp" else "mean") for k in wx})
        for k in sat:
            F[k] = sat[k].to_numpy()
        for r in FIRE_RADII_KM:
            F[f"fire_{r}"] = fire[r].to_numpy()
        index, cal_idx = DATES, DATES

    T, rm = len(index), cfg["rm"]
    F["wspd"] = np.hypot(F["u10"], F["v10"])
    F["wdir_sin"], F["wdir_cos"] = F["v10"] / (F["wspd"] + 1e-6), F["u10"] / (F["wspd"] + 1e-6)
    for p in POLLUTANTS:
        F[f"miss_{p}"] = np.isnan(F[p]).astype(np.float32)
        F[f"{p}_rm"] = pd.DataFrame(F[p]).rolling(rm, min_periods=max(1, rm // 4)).mean().to_numpy()
    for k in ("aod", "uvai", "no2_trop"):
        F[f"{k}_miss"] = np.isnan(F[k]).astype(np.float32)
    F["nbr_pm25"] = nbr_feature(F["pm25"], F["u10"], F["v10"], nb_idx, E, nb_dist)
    for k, v in calendar(cal_idx, mode == "daily").items():
        F[k] = np.repeat(np.asarray(v, dtype=np.float32)[:, None], N, 1)

    arr = np.stack([F[n] for n in FEATURES], -1).astype(np.float32)       # (T,N,F)
    df = pd.DataFrame(arr.reshape(T * N, -1), columns=FEATURES)
    df.insert(0, "station_id", np.tile(st.station_id.to_numpy(), T))
    df.insert(0, "time", np.repeat(index.values, N))
    df.to_parquet(PROC / f"features_{mode}.parquet", index=False)
    print(mode, "->", arr.shape, "| NaN share pm25:", float(np.isnan(arr[:, :, 0]).mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="hourly")
    main(ap.parse_args().mode)