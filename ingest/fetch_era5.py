"""ERA5 hourly single-levels for Delhi-NCR, one file per month, 4 downloads in parallel."""
from concurrent.futures import ThreadPoolExecutor
import cdsapi
from config import *

VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "2m_dewpoint_temperature",
        "surface_pressure", "boundary_layer_height", "total_precipitation"]


def job(ym):
    y, m = ym
    tgt = RAW / "era5" / f"era5_{y}_{m:02d}.nc"
    if tgt.exists():
        return
    tgt.parent.mkdir(exist_ok=True)
    cdsapi.Client(quiet=True).retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"], "variable": VARS, "year": [str(y)], "month": [f"{m:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)], "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf", "download_format": "unarchived", "area": ERA5_AREA}, str(tgt))
    print("done", y, m)


if __name__ == "__main__":
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(job, [(y, m) for y in YEARS for m in range(1, 13)]))