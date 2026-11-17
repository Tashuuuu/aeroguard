"""Sentinel-5P TROPOMI daily UV Aerosol Index + tropospheric NO2 column around every station
via Copernicus Data Space Sentinel Hub *Statistical API*. Needs OAuth client (CDSE dashboard -> OAuth clients)."""
import time, requests
import pandas as pd
from tqdm import tqdm
from config import *

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STAT_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"
BANDS = {"uvai": "AER_AI_340_380", "no2_trop": "NO2"}


def evalscript(band):
    return f"""//VERSION=3
function setup() {{ return {{ input: [{{bands: ["{band}", "dataMask"]}}],
  output: [{{id: "v", bands: 1}}, {{id: "dataMask", bands: 1}}] }}; }}
function evaluatePixel(s) {{ return {{ v: [s.{band}], dataMask: [s.dataMask] }}; }}"""


def token():
    r = requests.post(TOKEN_URL, data={"grant_type": "client_credentials", "client_id": CDSE_CLIENT_ID,
                                       "client_secret": CDSE_CLIENT_SECRET})
    r.raise_for_status()
    return r.json()["access_token"]


def stats(tok, lat, lon, band, t0, t1):
    body = {"input": {"bounds": {"bbox": [lon - .1, lat - .1, lon + .1, lat + .1],
                                 "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                      "data": [{"type": "sentinel-5p-l2", "dataFilter": {}, "processing": {"minQa": 50}}]},
            "aggregation": {"timeRange": {"from": t0, "to": t1}, "aggregationInterval": {"of": "P1D"},
                            "evalscript": evalscript(band), "resx": 0.05, "resy": 0.05}}
    for k in range(5):
        r = requests.post(STAT_URL, json=body, headers={"Authorization": f"Bearer {tok}"}, timeout=180)
        if r.status_code in (429, 500, 502, 503):
            time.sleep(5 * (k + 1)); continue
        r.raise_for_status()
        return r.json()["data"]
    raise RuntimeError("Statistical API kept failing")


def main():
    st = pd.read_csv(PROC / "stations.csv")
    (RAW / "s5p").mkdir(exist_ok=True)
    for r in tqdm(list(st.itertuples())):
        tok = token()
        for var, band in BANDS.items():
            for y in YEARS:
                f = RAW / "s5p" / f"{r.station_id}_{var}_{y}.parquet"
                if f.exists():
                    continue
                try:
                    data = stats(tok, r.lat, r.lon, band, f"{y}-01-01T00:00:00Z", f"{y}-12-31T23:59:59Z")
                    rows = [(d["interval"]["from"], float(d["outputs"]["v"]["bands"]["B0"]["stats"]["mean"]))
                            for d in data]
                except Exception as e:
                    print("skip", r.station_id, var, y, str(e)[:80]); rows = []
                pd.DataFrame(rows, columns=["date", "mean"]).to_parquet(f)
                time.sleep(0.5)


if __name__ == "__main__":
    main()