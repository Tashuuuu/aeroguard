"""Satellite AOD (MODIS MAIAC MCD19A2, 1 km, Optical_Depth_055)
sampled at every station via NASA AppEEARS."""

import time
import requests
import pandas as pd
from requests.auth import HTTPBasicAuth
from config import *

API = "https://appeears.earthdatacloud.nasa.gov/api"


def get_token():
    """Authenticate with NASA AppEEARS and return a bearer token."""

    if not EARTHDATA_USER or not EARTHDATA_PASS:
        raise ValueError("Earthdata username or password is missing.")

    response = requests.post(
        f"{API}/login",
        auth=HTTPBasicAuth(EARTHDATA_USER, EARTHDATA_PASS),
        data={"grant_type": "client_credentials"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60
    )

    print("Login HTTP status:", response.status_code)

    if not response.ok:
        raise RuntimeError(
            f"AppEEARS login failed.\n"
            f"HTTP status: {response.status_code}\n"
            f"Response: {response.text[:1000]}"
        )

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        raise RuntimeError(
            f"NASA returned a non-JSON login response:\n"
            f"{response.text[:1000]}"
        )

    if "token" not in result:
        raise RuntimeError(f"Token missing from NASA response: {result}")

    print("AppEEARS authentication successful.")
    return result["token"]


def main():

    st = pd.read_csv(PROC / "stations.csv")

    tok = get_token()

    h = {"Authorization": f"Bearer {tok}"}

    fmt = lambda d: pd.Timestamp(d).strftime("%m-%d-%Y")

    task = {
        "task_type": "point",
        "task_name": "aeroguard_aod",
        "params": {
            "dates": [{
                "startDate": fmt(START),
                "endDate": fmt(END)
            }],
            "layers": [{
                "product": "MCD19A2.061",
                "layer": "Optical_Depth_055"
            }],
            "coordinates": [
                {
                    "id": str(r.station_id),
                    "category": "station",
                    "latitude": float(r.lat),
                    "longitude": float(r.lon)
                }
                for r in st.itertuples()
            ]
        }
    }

    out = RAW / "aod"
    out.mkdir(parents=True, exist_ok=True)

    if not list(out.glob("*results*.csv")):

        r = requests.post(
            f"{API}/task",
            json=task,
            headers=h,
            timeout=120
        )
        r.raise_for_status()

        tid = r.json()["task_id"]

        print("AppEEARS task submitted:", tid)

        while True:

            response = requests.get(
                f"{API}/task/{tid}",
                headers=h,
                timeout=60
            )
            response.raise_for_status()

            s = response.json().get("status")

            print("AppEEARS status:", s)

            if s == "done":
                break

            if s in ("error", "failed"):
                raise RuntimeError(
                    f"AppEEARS task failed: {response.text}"
                )

            time.sleep(30)

        bundle = requests.get(
            f"{API}/bundle/{tid}",
            headers=h,
            timeout=60
        )
        bundle.raise_for_status()

        for f in bundle.json()["files"]:

            if (
                f["file_name"].endswith(".csv")
                and "results" in f["file_name"].lower()
            ):

                data = requests.get(
                    f"{API}/bundle/{tid}/{f['file_id']}",
                    headers=h,
                    timeout=120
                )
                data.raise_for_status()

                filename = f["file_name"].split("/")[-1]

                (out / filename).write_bytes(data.content)

                print("Saved:", filename)

    else:
        print("AOD results already exist. Skipping download.")


if __name__ == "__main__":
    main()
