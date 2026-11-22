# AeroGuard: air-quality analysis and forecasting for Delhi-NCR

A deep-learning project that models the factors affecting air quality (PM2.5, PM10, NO2) across Delhi-NCR and lets you explore them through an API and a Streamlit dashboard.

## What it does
- Forecasts the next 24 hours of PM2.5, PM10 and NO2 for 32 monitoring stations, with prediction intervals (5%-95% quantile outputs).
- Models the factors that affect air quality: the station's own readings, wind-adjusted influence of neighbouring stations, weather (wind, temperature, humidity, pressure, boundary-layer height, rain), satellite aerosol and gas measurements, crop-residue fires, Diwali and other festivals, a traffic proxy, and time of day, week and season.
- Runs what-if scenarios (no firecrackers, no stubble fires, reduced traffic) by changing these inputs and re-running the trained model.
- Flags stations whose readings differ strongly from what neighbours, satellite and weather predict.

## Model
- Spatio-temporal graph neural network (ST-GNN):
  - Temporal causal convolutional network (TCN) encoder for each station
  - Graph attention layer (GAT) across neighbouring stations
  - MLP head that outputs quantiles for 3 pollutants over the forecast horizon
- Trained with quantile (pinball) loss, with random masking of stations' own readings.
- Chronological split: train 2019 to mid-2021, validation late 2021, test 2022.

## Data (2019-2022)
- OpenAQ (CPCB) ground monitors
- ERA5 weather (Copernicus)
- MODIS AOD (NASA)
- Sentinel-5P UV aerosol index and tropospheric NO2 (Copernicus)
- VIIRS fire detections (NASA FIRMS)

## Run locally
1. `pip install -r requirements-serve.txt`
2. Place the trained workspace (processed data and model files) in `./workspace`
3. `python -m uvicorn api.main:app --port 8000`
4. `python -m streamlit run app/dashboard.py` (opens at http://localhost:8501)
5. Tests: `python -m pytest -q`

Dockerfile and docker-compose.yml are
