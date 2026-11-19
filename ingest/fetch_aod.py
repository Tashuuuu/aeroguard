"""Satellite AOD (MODIS MAIAC MCD19A2, 1 km, Optical_Depth_055) sampled at every station via Google Earth Engine."""
import os, sys
import pandas as pd
import ee
from config import *   # needs PROC, RAW, START, END

PROJECT = os.environ.get("EE_PROJECT")
CHUNK_DAYS = 10


def main():
    if not PROJECT:
        sys.exit("Set EE_PROJECT first (your Earth Engine / Cloud project ID).")
    try:
        ee.Initialize(project=PROJECT)
    except Exception as e:
        sys.exit(f"Earth Engine init failed: {e}\nRun ee.Authenticate() in a notebook cell first.")

    st = pd.read_csv(PROC / "stations.csv")
    need = {"station_id", "lat", "lon"}
    if not need.issubset(st.columns):
        sys.exit(f"stations.csv needs columns {need}; it has {list(st.columns)}")

    out = RAW / "aod"
    out.mkdir(parents=True, exist_ok=True)
    final = out / "aod_gee_results.csv"
    part = out / "aod_gee_partial.csv"
    if final.exists():
        print("Already done:", final); return

    done_until = None
    rows = []
    if part.exists():                                   # resume after a disconnect
        prev = pd.read_csv(part)
        rows = list(prev.itertuples(index=False, name=None))
        done_until = pd.read_csv(out / "aod_progress.txt", header=None).iloc[0, 0]
        print("Resuming after", done_until)

    pts = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]).buffer(2500),
                   {"station_id": str(r.station_id)}) for r in st.itertuples()])
    col = (ee.ImageCollection("MODIS/061/MCD19A2_GRANULES").select("Optical_Depth_055")
           .filterBounds(pts.geometry()))

    start, end = pd.Timestamp(START), pd.Timestamp(END)
    for a in pd.date_range(start, end, freq=f"{CHUNK_DAYS}D"):
        if done_until and a.strftime("%Y-%m-%d") <= done_until:
            continue
        b = min(a + pd.Timedelta(days=CHUNK_DAYS), end + pd.Timedelta(days=1))

        def per_img(img):
            return img.reduceRegions(pts, ee.Reducer.mean(), 1000).map(
                lambda f: f.set("date", img.date().format("YYYY-MM-dd")))

        fc = col.filterDate(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")).map(per_img).flatten()
        for f in fc.getInfo()["features"]:
            p = f["properties"]
            if p.get("mean") is not None:
                rows.append((p["station_id"], p["date"], p["mean"] * 0.001))   # MODIS scale factor
        pd.DataFrame(rows, columns=["station_id", "date", "aod_055"]).to_csv(part, index=False)
        (out / "aod_progress.txt").write_text(a.strftime("%Y-%m-%d"))
        print(a.date(), "->", len(rows), "rows")

    df = pd.DataFrame(rows, columns=["station_id", "date", "aod_055"])
    df = df.groupby(["station_id", "date"], as_index=False).aod_055.mean()   # several granules per day
    df.to_csv(final, index=False)
    print("saved", final, df.shape)


if __name__ == "__main__":
    main()