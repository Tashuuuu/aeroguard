"""Station graph: kNN edges weighted by inverse-distance x alignment with prevailing wind. Also the wind-aware neighbour feature."""
import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from utils import haversine_km


def knn_neighbours(lat, lon, k):
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    return idx, np.take_along_axis(D, idx, axis=1)                     # (N,k) neighbour ids, km


def neighbour_unit_vectors(lat, lon, idx):
    """Unit vector (east, north) pointing FROM neighbour j TO station i. Shape (N,k,2)."""
    dx = (lon[:, None] - lon[idx]) * np.cos(np.radians(lat[:, None])) * 111.32
    dy = (lat[:, None] - lat[idx]) * 110.57
    n = np.hypot(dx, dy) + 1e-9
    return np.stack([dx / n, dy / n], -1)


def nbr_feature(pm, u, v, idx, E, D):
    """Wind-adjusted neighbour influence: for each station/time, weighted mean of neighbours' PM2.5,
    weight = max(0, cos(angle between wind and j->i)) / distance (upwind neighbours count more).
    NEVER uses the station's own reading. pm,u,v: (T,N) -> (T,N)."""
    ws = np.hypot(u, v) + 1e-6
    cos = (u[:, :, None] * E[None, :, :, 0] + v[:, :, None] * E[None, :, :, 1]) / ws[:, :, None]
    pm_nb = pm[:, idx]                                                 # (T,N,k)
    valid = ~np.isnan(pm_nb)
    pm0 = np.where(valid, pm_nb, 0.0)
    w = np.clip(cos, 0, None) / D[None]
    den = (w * valid).sum(2)
    up = (w * pm0).sum(2) / np.maximum(den, 1e-9)
    w2 = valid / D[None]
    fb = (w2 * pm0).sum(2) / np.maximum(w2.sum(2), 1e-9)               # fallback: plain inverse-distance mean
    out = np.where(den > 1e-9, up, fb)
    return np.where(valid.any(2), out, np.nan)


def build_edges(lat, lon, wind_uv, k=KNN):
    idx, dist = knn_neighbours(lat, lon, k)
    E = neighbour_unit_vectors(lat, lon, idx)
    wn = wind_uv / (np.linalg.norm(wind_uv) + 1e-9)
    sigma = np.median(dist)
    src, dst, w = [], [], []
    for i in range(len(lat)):
        for kk in range(idx.shape[1]):
            align = 0.5 * (1 + float(E[i, kk] @ wn))                   # 1 = neighbour is directly upwind of i
            src.append(idx[i, kk]); dst.append(i)
            w.append(np.exp(-dist[i, kk] / sigma) * (0.1 + 0.9 * align))
    return np.array([src, dst], dtype=np.int64), np.array(w, dtype=np.float32), idx, dist


def main(mode):
    st = pd.read_csv(PROC / "stations.csv").sort_values("station_id").reset_index(drop=True)
    f = pd.read_parquet(PROC / f"features_{mode}.parquet", columns=["time", "u10", "v10"])
    m = pd.to_datetime(f["time"]).dt.month.isin([10, 11, 12, 1, 2])   # pollution season prevailing wind
    wind = np.array([f.loc[m, "u10"].mean(), f.loc[m, "v10"].mean()])
    lat, lon = st.lat.to_numpy(), st.lon.to_numpy()
    ei, ew, idx, dist = build_edges(lat, lon, wind)
    np.savez(PROC / f"graph_{mode}.npz", edge_index=ei, edge_weight=ew, station_ids=st.station_id.to_numpy(),
             lat=lat, lon=lon, wind=wind)
    fig, ax = plt.subplots(figsize=(6, 6))
    for (s, d), w in zip(ei.T, ew):
        ax.plot([lon[s], lon[d]], [lat[s], lat[d]], lw=0.5 + 3 * w, alpha=0.4, c="tab:blue")
    ax.scatter(lon, lat, c="k", s=12)
    ax.set_title(f"Station graph (prevailing winter wind u,v = {wind.round(2)})")
    fig.savefig(ART / "graph.png", dpi=150, bbox_inches="tight")
    print("nodes", len(st), "edges", ei.shape[1], "wind", wind)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="hourly")
    main(ap.parse_args().mode)