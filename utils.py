"""utils.py — HTTP retry, distance, ERA5 reader."""
import time, zipfile
from pathlib import Path
import numpy as np
import requests
import xarray as xr


def http_get(url, headers=None, params=None, tries=6):
    """GET with retry/backoff; sleeps on HTTP 429 (rate limit)."""
    for k in range(tries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=120)
            if r.status_code in (408, 429):
                time.sleep(float(r.headers.get("x-ratelimit-reset", 30 if r.status_code == 429 else 5)) + 1)
                continue
            if r.status_code >= 500:
                time.sleep(2.0 * (k + 1))
                continue
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout):
            if k == tries - 1:
                raise
            time.sleep(2.0 * (k + 1))
    raise RuntimeError(f"GET failed: {url}")


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def open_era5(path):
    """Open one ERA5 monthly file (plain .nc or zip of .nc) -> xarray Dataset with dim 'time'."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        d = path.with_suffix("")
        d.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(d)
        files = sorted(d.glob("*.nc"))
    else:
        files = [path]
    ds = xr.merge([xr.open_dataset(f) for f in files], compat="override")
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds.drop_vars([v for v in ("expver", "number") if v in ds.variables])