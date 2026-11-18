"""ERA5 hourly single-levels for Delhi-NCR. Requests go ONE AT A TIME.
Tries 3-month chunks; if CDS says 'too large', falls back to single months."""
import requests, cdsapi
from config import *

VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "2m_dewpoint_temperature",
        "surface_pressure", "boundary_layer_height", "total_precipitation"]
OUT = RAW / "era5"


def fetch(y, months, tgt):
    tgt.parent.mkdir(exist_ok=True)
    tmp = tgt.with_suffix(".part")                       # never leave a half-downloaded .nc behind
    cdsapi.Client(quiet=True).retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"], "variable": VARS, "year": [str(y)],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)], "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf", "download_format": "unarchived", "area": ERA5_AREA}, str(tmp))
    tmp.rename(tgt)
    print("done", tgt.name, flush=True)


def monthly(y, months):
    for m in months:
        t = OUT / f"era5_{y}_{m:02d}.nc"
        if not t.exists():
            fetch(y, [m], t)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for y in YEARS:
        for q in ([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]):
            chunk = OUT / f"era5_{y}_q{q[0]:02d}.nc"
            month_files = [OUT / f"era5_{y}_{m:02d}.nc" for m in q]
            if chunk.exists() or all(f.exists() for f in month_files):
                continue
            if any(f.exists() for f in month_files):     # monthly already started: finish monthly (no overlap)
                monthly(y, q)
                continue
            try:
                fetch(y, q, chunk)
            except requests.HTTPError as e:
                if "too large" in str(e) or "cost limits" in str(e):
                    print("3-month request too large -> using single months", flush=True)
                    monthly(y, q)
                else:
                    raise