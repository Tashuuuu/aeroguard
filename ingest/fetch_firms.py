"""NASA FIRMS active-fire detections (VIIRS S-NPP standard product) over Punjab-Haryana-Delhi,5-day chunks."""
import io
import pandas as pd
from config import *
from utils import http_get

COLS = ["latitude", "longitude", "acq_date", "frp", "confidence"]


def main():
    d, end, b = pd.Timestamp(START), pd.Timestamp(END), FIRE_BBOX
    (RAW / "firms").mkdir(exist_ok=True)
    while d <= end:
        f = RAW / "firms" / f"{d:%Y%m%d}.parquet"
        if not f.exists():
            url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/VIIRS_SNPP_SP/"
                   f"{b['west']},{b['south']},{b['east']},{b['north']}/5/{d:%Y-%m-%d}")
            txt = http_get(url).text
            if not txt.startswith("latitude"):
                if "nvalid" in txt:
                    raise RuntimeError(txt[:200])
                df = pd.DataFrame(columns=COLS)
            else:
                df = pd.read_csv(io.StringIO(txt))
                df = df[df["confidence"].astype(str).str.lower().isin(["n", "h", "nominal", "high"])]
            df[COLS].astype({"confidence": str}).to_parquet(f)
        d += pd.Timedelta(days=5)
    allf = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "firms").glob("*.parquet"))])
    allf.to_parquet(RAW / "firms.parquet")
    print(len(allf), "fire detections")


if __name__ == "__main__":
    main()