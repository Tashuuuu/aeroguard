"""Satellite AOD (MODIS MAIAC MCD19A2, 1 km, Optical_Depth_055) sampled at every station via NASA AppEEARS."""
import time, requests
import pandas as pd
from config import *

API = "https://appeears.earthdatacloud.nasa.gov/api"


def main():
    st = pd.read_csv(PROC / "stations.csv")
    tok = requests.post(f"{API}/login", auth=(EARTHDATA_USER, EARTHDATA_PASS)).json()["token"]
    h = {"Authorization": f"Bearer {tok}"}
    fmt = lambda d: pd.Timestamp(d).strftime("%m-%d-%Y")
    task = {"task_type": "point", "task_name": "aeroguard_aod",
            "params": {"dates": [{"startDate": fmt(START), "endDate": fmt(END)}],
                       "layers": [{"product": "MCD19A2.061", "layer": "Optical_Depth_055"}],
                       "coordinates": [{"id": str(r.station_id), "category": "station",
                                        "latitude": float(r.lat), "longitude": float(r.lon)} for r in st.itertuples()]}}
    out = RAW / "aod"
    out.mkdir(exist_ok=True)
    if not list(out.glob("*results*.csv")):
        r = requests.post(f"{API}/task", json=task, headers=h)
        r.raise_for_status()
        tid = r.json()["task_id"]
        while True:
            s = requests.get(f"{API}/task/{tid}", headers=h).json().get("status")
            print("AppEEARS status:", s)
            if s == "done":
                break
            if s == "error":
                raise RuntimeError("AppEEARS task failed")
            time.sleep(30)
        for f in requests.get(f"{API}/bundle/{tid}", headers=h).json()["files"]:
            if f["file_name"].endswith(".csv") and "results" in f["file_name"].lower():
                data = requests.get(f"{API}/bundle/{tid}/{f['file_id']}", headers=h, stream=True)
                (out / f["file_name"].split("/")[-1]).write_bytes(data.content)
                print("saved", f["file_name"])


if __name__ == "__main__":
    main()