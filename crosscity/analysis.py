"""Cross-city comparative analysis: why are Indian cities so polluted? Correlation + ridge + random-forest importance.
PM2.5 per city is fetched from OpenAQ (CPCB); population / vehicles / factories come from YOUR filled crosscity/city_stats.csv
(Census of India, MoRTH VAHAN, MoSPI ASI). Optional crosscity/iqair_top.csv (city,country,pm25) from the IQAir report."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from config import *
from utils import http_get

HERE = Path(__file__).parent
BASE, HDR = "https://api.openaq.org/v3", {"X-API-Key": OPENAQ_API_KEY}
CITY_YEAR = 2022
CITIES = {"Delhi": (28.61, 77.21), "Mumbai": (19.08, 72.88), "Kolkata": (22.57, 88.36), "Chennai": (13.08, 80.27),
          "Bengaluru": (12.97, 77.59), "Hyderabad": (17.39, 78.49), "Ahmedabad": (23.02, 72.57), "Pune": (18.52, 73.86),
          "Lucknow": (26.85, 80.95), "Kanpur": (26.45, 80.33), "Patna": (25.59, 85.14), "Jaipur": (26.91, 75.79),
          "Visakhapatnam": (17.69, 83.22), "Nagpur": (21.15, 79.09)}
FEATS = ["pop_density", "vehicles_per_1000", "factories_per_100k"]


def fetch_city_pm25():
    cache = RAW / "city_pm25.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for city, (la, lo) in CITIES.items():
        locs = http_get(f"{BASE}/locations", HDR, {"bbox": f"{lo - .25},{la - .25},{lo + .25},{la + .25}", "limit": 1000}).json()["results"]
        vals = []
        for loc in locs:
            if not loc.get("isMonitor"):
                continue
            for s in loc["sensors"]:
                if s["parameter"]["name"] == "pm25":
                    r = http_get(f"{BASE}/sensors/{s['id']}/days", HDR,
                                 {"datetime_from": f"{CITY_YEAR}-01-01T00:00:00Z",
                                  "datetime_to": f"{CITY_YEAR + 1}-01-01T00:00:00Z", "limit": 1000}).json()["results"]
                    v = [d["value"] for d in r if d["value"] is not None and d["value"] > 0]
                    if len(v) >= 200:
                        vals.append(float(np.mean(v)))
                    time.sleep(1.05)
        rows.append(dict(city=city, pm25=float(np.mean(vals)) if vals else np.nan, n_stations=len(vals)))
        print(city, rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def main():
    stats_f = HERE / "city_stats.csv"
    if not stats_f.exists():
        pd.DataFrame({"city": list(CITIES), "population": "", "area_km2": "", "vehicles_registered": "", "factories": ""}
                     ).to_csv(stats_f, index=False)
        raise SystemExit(f"Fill {stats_f} from Census / VAHAN / ASI, then re-run.")
    df = fetch_city_pm25().merge(pd.read_csv(stats_f), on="city")
    df["pop_density"] = df.population / df.area_km2
    df["vehicles_per_1000"] = df.vehicles_registered / df.population * 1000
    df["factories_per_100k"] = df.factories / df.population * 1e5
    df = df.dropna(subset=FEATS + ["pm25"]).reset_index(drop=True)
    X, y = df[FEATS], df["pm25"]
    out = dict(n_cities=len(df), spearman=df[FEATS + ["pm25"]].corr(method="spearman")["pm25"].drop("pm25").to_dict())
    ridge = Ridge(alpha=1.0).fit(StandardScaler().fit_transform(X), y)
    out["ridge_std_coef"] = dict(zip(FEATS, ridge.coef_.tolist()))
    rf = RandomForestRegressor(500, random_state=0).fit(X, y)
    out["rf_importance"] = dict(zip(FEATS, rf.feature_importances_.tolist()))
    out["rf_permutation_importance"] = dict(zip(FEATS, permutation_importance(rf, X, y, n_repeats=30, random_state=0).importances_mean.tolist()))
    iq = HERE / "iqair_top.csv"
    if iq.exists():
        t = pd.read_csv(iq).sort_values("pm25", ascending=False)
        out["india_share_top50"] = float((t.head(50).country == "India").mean())
        out["india_share_top100"] = float((t.head(100).country == "India").mean())
    df.to_csv(ART / "crosscity_table.csv", index=False)
    json.dump(out, open(ART / "crosscity_results.json", "w"), indent=2)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    for a, f in zip(ax, FEATS):
        a.scatter(df[f], df.pm25)
        for _, r in df.iterrows():
            a.annotate(r.city, (r[f], r.pm25), fontsize=6)
        a.set_xlabel(f); a.set_ylabel("PM2.5 (ug/m3)")
    fig.savefig(ART / "crosscity.png", dpi=150, bbox_inches="tight")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()