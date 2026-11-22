# AeroGuard: air-quality forecasting and analysis for Delhi-NCR
 
AeroGuard is an end-to-end deep-learning system that models the factors affecting air quality (PM2.5, PM10 and NO2) across Delhi-NCR. It forecasts pollution at 32 monitoring stations with prediction intervals, simulates what-if scenarios, flags unusual station readings, and serves everything through an API and a Streamlit dashboard.
 
## What was built
 
**1. Multi-source data pipeline (2019-2022)**
- Ground air-quality readings from CPCB monitors via OpenAQ (PM2.5, PM10, NO2)
- ERA5 reanalysis weather from Copernicus (wind, temperature, dew point, pressure, boundary-layer height, precipitation)
- MODIS aerosol optical depth (NASA)
- Sentinel-5P UV aerosol index and tropospheric NO2 (Copernicus)
- VIIRS fire detections from NASA FIRMS, used for crop-residue burning
- Resumable downloaders, hourly alignment and quality filtering for all sources
**2. Feature engineering for the factors that affect air quality**
- Each station's own readings and rolling averages, with missing-data indicators
- Wind-adjusted influence of neighbouring stations
- Weather: wind speed and direction, temperature, humidity, pressure, boundary-layer height, rain
- Satellite aerosol and gas measurements
- Fire counts within 50 km and 300 km radii, and stubble-burning season
- Diwali and other festival flags, a traffic proxy, and hour, day-of-week and month cycles
**3. Station graph**
- Stations connected to their 6 nearest neighbours using geographic distance, with edge weights
**4. Deep-learning model: spatio-temporal graph neural network (ST-GNN)**
- Temporal convolutional network (causal, dilated, residual) that encodes each station's recent history
- Graph attention layer (GAT) that shares information between neighbouring stations
- MLP output head predicting 5 quantiles (5%, 25%, 50%, 75%, 95%) for 3 pollutants over the forecast horizon
- Trained with quantile (pinball) loss, AdamW, learning-rate scheduling, gradient clipping and early stopping
- Masked-input training that hides stations' own readings so the model learns from neighbours, satellite and weather
- Hourly model (72-hour window, 24-hour horizon) and daily model (14-day window, 3-day horizon)
- Ablation variants without graph attention and without satellite and fire features
- Chronological split: train 2019 to mid-2021, validation late 2021, test 2022
- Conformal calibration script for the prediction intervals
**5. Anomaly detection**
- Predicts each station's expected PM2.5 from neighbours, satellite and weather only, then flags sustained, significant deviations
- Detection threshold set on validation data and tested with injected distortions
**6. What-if scenario simulator**
- Re-runs the trained model with factors changed: no firecrackers, no stubble fires, no fires and no firecrackers, and reduced traffic
- Reports the change in PM2.5 overall and per station
**7. REST API (FastAPI)**
- `/health`, `/stations`, `/forecast`, `/anomalies`, `/counterfactual`, `/results`
- Interactive API documentation at `/docs`
**8. Streamlit dashboard**
- Forecast: station and pollutant selection, median forecast with 50% and 90% interval bands
- Anomaly flags: map and table of flagged stations for any date range
- What-if scenarios: run a scenario and see the PM2.5 change per station
- Benchmarks: evaluation metrics (RMSE, MAE and interval coverage)
**9. Engineering**
- Unit tests (pytest) for the model, the loss and the graph construction
- GitHub Actions workflow
- Colab-based training and evaluation workflow
## Project structure
```
ingest/          data downloaders (OpenAQ, ERA5, AOD, Sentinel-5P, FIRMS)
preprocess/      feature building and datasets
graphs/          station graph construction
models/          ST-GNN, loss function and inference
anomaly/         anomaly detector
counterfactual/  what-if simulator
api/             FastAPI service
app/             Streamlit dashboard
tests/           unit tests
train.py         model training
evaluate.py      evaluation
```
 
## Run locally
1. `pip install -r requirements-serve.txt`
2. Place the trained workspace (processed data and model files) in `./workspace`
3. Start the API: `python -m uvicorn api.main:app --port 8000`
4. Start the dashboard: `python -m streamlit run app/dashboard.py` (opens at http://localhost:8501)
5. Run the tests: `python -m pytest -q`
