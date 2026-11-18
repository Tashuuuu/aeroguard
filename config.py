"""config.py — single source of truth for paths, dates, features, hyper-parameters."""
import os
from pathlib import Path
import pandas as pd

# ---------- paths ----------
ROOT = Path(os.environ.get("AEROGUARD_ROOT", "./workspace"))
RAW, PROC, ART = ROOT / "raw", ROOT / "processed", ROOT / "artifacts"
PRED = ART / "preds"
for _p in (RAW, PROC, ART, PRED):
    _p.mkdir(parents=True, exist_ok=True)

# ---------- secrets (from environment) ----------
OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "")
FIRMS_MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")
EARTHDATA_USER = os.environ.get("EARTHDATA_USER", "")
EARTHDATA_PASS = os.environ.get("EARTHDATA_PASS", "")
CDSE_CLIENT_ID = os.environ.get("CDSE_CLIENT_ID", "")
CDSE_CLIENT_SECRET = os.environ.get("CDSE_CLIENT_SECRET", "")

# ---------- study window & region ----------
START, END = "2019-01-01", "2022-12-31"
YEARS = list(range(int(START[:4]), int(END[:4]) + 1))
BBOX = dict(west=76.6, south=27.9, east=77.9, north=29.2)        # Delhi-NCR stations
ERA5_AREA = [29.5, 76.3, 27.6, 78.2]                              # N, W, S, E (CDS order)
FIRE_BBOX = dict(west=72.5, south=26.5, east=80.0, north=33.5)    # incl. Punjab/Haryana stubble belt
FIRE_RADII_KM = [50, 300]                                         # configurable fire-count radii
IST = pd.Timedelta(hours=5, minutes=30)
MIN_COVERAGE = 0.5                                                # keep stations with >=50% hourly PM2.5
KNN = 6                                                           # graph neighbours per station

# ---------- pollutants / quantiles ----------
POLLUTANTS = ["pm25", "pm10", "no2"]
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]                         # 90% and 50% intervals

# ---------- modes: daily-aggregated vs hourly-interpolated pipeline ----------
MODES = {
    "hourly": dict(freq="h", window=72, horizon=24, dilations=[1, 2, 4, 8, 16], rm=24,
                   leads=[1, 6, 12, 24], eval_stride=3, batch=32),
    "daily":  dict(freq="D", window=14, horizon=3, dilations=[1, 2, 4], rm=7,
                   leads=[1, 2, 3], eval_stride=1, batch=64),
}

# ---------- chronological splits (by target time) ----------
SPLITS = {"train": ("2019-01-01", "2021-06-30"),
          "val":   ("2021-07-01", "2021-12-31"),
          "test":  ("2022-01-01", "2022-12-31")}

# ---------- natural experiments (held out of training; used for counterfactual validation) ----------
EVENTS = {"odd_even_2019": ("2019-11-04", "2019-11-15"),
          "covid_lockdown_2020": ("2020-03-25", "2020-05-31")}
# !! EDIT: documented fractional drop in road traffic during each event. Replace with the figure
# !! from the study you cite in your report. These two numbers are placeholders, not facts.
TRAFFIC_REDUCTION = {"odd_even_2019": 0.02, "covid_lockdown_2020": 0.65}

# ---------- calendar ----------
DIWALI = ["2019-10-27", "2020-11-14", "2021-11-04", "2022-10-24"]
OTHER_FESTIVALS = ["2019-03-21", "2020-03-10", "2021-03-29", "2022-03-18",     # Holi
                   "2019-10-08", "2020-10-25", "2021-10-15", "2022-10-05"]     # Dussehra
HOUR_TRAFFIC = [.15, .10, .08, .08, .12, .25, .50, .80, 1.0, .90, .70, .65,
                .65, .65, .70, .80, .95, 1.0, .95, .80, .60, .45, .30, .20]    # assumed diurnal shape
WEEK_TRAFFIC = [1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.65]                            # Mon..Sun

# ---------- feature schema (ORDER MATTERS; first 3 = targets) ----------
OWN_VAL = ["pm25", "pm10", "no2", "pm25_rm", "pm10_rm", "no2_rm"]   # station's OWN ground readings
OWN_MISS = ["miss_pm25", "miss_pm10", "miss_no2"]                  # 1 where own reading missing/masked
NBR = ["nbr_pm25"]                                                 # wind-adjusted NEIGHBOUR influence
SAT = ["aod", "aod_miss", "uvai", "uvai_miss", "no2_trop", "no2_trop_miss"]
FIRE = [f"fire_{r}" for r in FIRE_RADII_KM]
WEATHER = ["u10", "v10", "wspd", "wdir_sin", "wdir_cos", "t2m", "rh", "sp", "blh", "tp"]
CAL = ["diwali_flag", "festival_flag", "stubble_season", "traffic_proxy",
       "hr_sin", "hr_cos", "dow_sin", "dow_cos", "mon_sin", "mon_cos", "is_weekend"]
FEATURES = OWN_VAL + OWN_MISS + NBR + SAT + FIRE + WEATHER + CAL
SATFIRE = SAT + FIRE                                               # dropped in the "no satellite/fire" ablation
NO_SCALE = set(OWN_MISS + ["aod_miss", "uvai_miss", "no2_trop_miss", "diwali_flag", "festival_flag",
               "stubble_season", "is_weekend", "hr_sin", "hr_cos", "dow_sin", "dow_cos",
               "mon_sin", "mon_cos", "wdir_sin", "wdir_cos"])

# ---------- training hyper-parameters ----------
HP = dict(hidden=64, heads=4, dropout=0.1, kernel=3, lr=1e-3, wd=1e-4, epochs=40, patience=6,
          p_node_mask=0.25, p_global_mask=0.10, node_mask_frac=0.25, seed=42)

# ---------- anomaly layer ----------
ANOM = dict(roll={"hourly": 12, "daily": 3}, fpr=0.01, seg={"hourly": 48, "daily": 7},
            per_type=8, kinds=[("scale", 0.6), ("scale", 1.5), ("offset", -40.0), ("offset", 40.0)])