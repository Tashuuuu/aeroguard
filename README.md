# AeroGuard: air-quality forecasting for Delhi-NCR

Spatio-temporal graph neural network (TCN + graph attention) that forecasts PM2.5, PM10 and NO2 with uncertainty bands for 32 monitoring stations, plus an anomaly detector, a what-if simulator, an API and a Streamlit dashboard.

**Data (2019-2022):** OpenAQ (CPCB) ground monitors, ERA5 weather, NASA MODIS AOD, Copernicus Sentinel-5P (UV aerosol index, tropospheric NO2), NASA FIRMS fires

## Results (hourly, 2022 test set, PM2.5 RMSE, lower is better)
| Method | RMSE |
|---|---|
| Persistence | 83.50 |
| ST-GNN | 67.84 |
| XGBoost | 62.51 |

## Limitations
- The ST-GNN beats persistence but not XGBoost.
- Removing graph attention or satellite/fire features did not hurt, so those components are not shown to help.
- Prediction intervals are somewhat overconfident (90% bands cover about 80%).
- The anomaly detector mainly catches large upward changes.
- What-if results show model sensitivity, not proven causal effects.
- Cross-city analysis was planned but not completed.

## Run locally
1. `pip install -r requirements-serve.txt`
2. Place the trained workspace (data and model files) in `./workspace`.
3. `python -m uvicorn api.main:app --port 8000`
4. `python -m streamlit run app/dashboard.py`
