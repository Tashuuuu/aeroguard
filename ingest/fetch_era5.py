"""ERA5 hourly single-levels for Delhi-NCR, one file per half-year, requested ONE AT A TIME (CDS queue limit)."""
import cdsapi
from config import *

VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "2m_dewpoint_temperature",
        "surface_pressure", "boundary_layer_height", "total_precipitation"]


def job(y, half):
    months = list(range(1, 7)) if half == 1 else list(range(7, 13))
    tgt = RAW / "era5" / f"era5_{y}_h{half}.nc"
    if tgt.exists():
        return
    tgt.parent.mkdir(exist_ok=True)
    tmp = tgt.with_suffix(".part")                      # never leave a half-downloaded .nc behind
    cdsapi.Client(quiet=True).retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"], "variable": VARS, "year": [str(y)],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)], "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf", "download_format": "unarchived", "area": ERA5_AREA}, str(tmp))
    tmp.rename(tgt)
    print("done", y, "half", half, flush=True)


if __name__ == "__main__":
    for y in YEARS:
        for half in (1, 2):
            job(y, half)