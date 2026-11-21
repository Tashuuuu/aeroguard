"""Satellite AOD (MODIS MAIAC MCD19A2, 1 km, Optical_Depth_055) sampled at every station via Google Earth Engine.
Granules are averaged into one image per day server-side, so a chunk returns (days x stations) rows."""
import os, sys
import pandas as pd
import ee
from config import *   # needs PROC, RAW, START, END

PROJECT = os.environ.get("EE_PROJECT")
CHUNK_DAYS = 30


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
    print(len(st), "stations")

    out = RAW / "aod"
    out.mkdir(parents=True, exist_ok=True)
    final = out / "aod_gee_results.csv"
    part = out / "aod_gee_partial.csv"
    prog = out / "aod_progress.txt"
    if final.exists():
        print("Already done:", final); return

    done_until, rows = None, []
    if part.exists() and prog.exists():                 # resume after a disconnect
        rows = list(pd.read_csv(part).itertuples(index=False, name=None))
        done_until = prog.read_text().strip()
        print("Resuming after", done_until)

    pts = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]).buffer(2500),
                   {"station_id": str(r.station_id)}) for r in st.itertuples()])
    col = (ee.ImageCollection("MODIS/061/MCD19A2_GRANULES").select("Optical_Depth_055")
           .filterBounds(pts.geometry()))

    def sample_chunk(a, nd):
        start = ee.Date(a.strftime("%Y-%m-%d"))

        def day_img(i):
            d = start.advance(ee.Number(i), "day")
            sub = col.filterDate(d, d.advance(1, "day"))
            img = ee.Image(ee.Algorithms.If(
                sub.size().gt(0), sub.mean(),
                ee.Image.constant(0).rename("Optical_Depth_055").selfMask()))
            return img.set("date", d.format("YYYY-MM-dd"))

        imgs = ee.ImageCollection.fromImages(ee.List.sequence(0, nd - 1).map(day_img))
        fc = imgs.map(lambda img: img.reduceRegions(pts, ee.Reducer.mean(), 1000).map(
            lambda f: f.set("date", img.get("date")))).flatten()
        return fc.getInfo()["features"]

    start, end = pd.Timestamp(START), pd.Timestamp(END)
    for a in pd.date_range(start, end, freq=f"{CHUNK_DAYS}D"):
        if done_until and a.strftime("%Y-%m-%d") <= done_until:
            continue
        nd = min(CHUNK_DAYS, (end - a).days + 1)
        for f in sample_chunk(a, nd):
            p = f["properties"]
            if p.get("mean") is not None:
                rows.append((p["station_id"], p["date"], p["mean"] * 0.001))   # MODIS scale factor
        pd.DataFrame(rows, columns=["station_id", "date", "aod_055"]).to_csv(part, index=False)
        prog.write_text(a.strftime("%Y-%m-%d"))
        print(a.date(), "->", len(rows), "rows")

    df = pd.DataFrame(rows, columns=["station_id", "date", "aod_055"])
    df = df.groupby(["station_id", "date"], as_index=False).aod_055.mean()
    df.to_csv(final, index=False)
    print("saved", final, df.shape)


if __name__ == "__main__":
    main()