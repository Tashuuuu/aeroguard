"""Streamlit dashboard on top of the FastAPI service."""
import os
import pandas as pd, plotly.graph_objects as go, requests, streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")
st.set_page_config(page_title="AeroGuard", layout="wide")
st.title("AeroGuard — tamper-resilient air-quality forecasting, Delhi-NCR")


@st.cache_data(ttl=300)
def get(path, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=120)
    r.raise_for_status()
    return r.json()


tabs = st.tabs(["Forecast", "Anomaly flags", "What-if scenarios", "Cross-city", "Benchmarks"])
stations = pd.DataFrame(get("/stations"))

with tabs[0]:
    name = st.selectbox("Station", stations.name)
    sid = int(stations.loc[stations.name == name, "station_id"].iloc[0])
    origin = st.text_input("Forecast origin (UTC/IST date-time, blank = latest)", "")
    f = get("/forecast", station_id=sid, **({"origin": origin} if origin else {}))
    pol = st.radio("Pollutant", list(f["forecast"].keys()), horizontal=True)
    d = pd.DataFrame(f["forecast"][pol])
    fig = go.Figure()
    fig.add_scatter(x=d.time, y=d.q95, line=dict(width=0), showlegend=False)
    fig.add_scatter(x=d.time, y=d.q05, fill="tonexty", name="90% interval", line=dict(width=0))
    fig.add_scatter(x=d.time, y=d.q75, line=dict(width=0), showlegend=False)
    fig.add_scatter(x=d.time, y=d.q25, fill="tonexty", name="50% interval", line=dict(width=0))
    fig.add_scatter(x=d.time, y=d.q50, name="median", line=dict(color="black"))
    fig.update_layout(yaxis_title=f"{pol} (µg/m³)", height=420)
    st.caption(f"Forecast issued from {f['origin']}")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    c1, c2 = st.columns(2)
    a, b = c1.text_input("From", "2022-11-01"), c2.text_input("To", "2022-11-30")
    an = pd.DataFrame(get("/anomalies", date_from=a, date_to=b))
    if an.empty:
        st.success("No flagged stations in this period.")
    else:
        st.warning("Negative z_bar = station reads LOWER than independent evidence predicts (possible under-reporting).")
        st.map(an[["lat", "lon"]])
        st.dataframe(an.groupby("name").agg(flagged_steps=("flag", "sum"), mean_z=("z_bar", "mean")).sort_values("flagged_steps", ascending=False))

with tabs[2]:
    sc = st.selectbox("Scenario", ["no_firecrackers", "no_stubble_fires", "no_fires_no_firecrackers", "reduced_traffic"])
    c1, c2, c3 = st.columns(3)
    s0, s1 = c1.text_input("Start", "2022-10-22"), c2.text_input("End", "2022-10-29")
    red = c3.slider("Traffic reduction", 0.0, 0.9, 0.25) if sc == "reduced_traffic" else 0.0
    if st.button("Run scenario"):
        r = requests.post(f"{API}/counterfactual", json=dict(scenario=sc, start=s0, end=s1, reduction=red), timeout=600).json()
        st.metric("Estimated PM2.5 change", f"-{r['abs_impact']:.1f} µg/m³", f"{-r['pct_impact']:.1f}%")
        st.caption("Feature-ablation scenario simulation with a trained model — not formal causal inference.")
        st.bar_chart(pd.Series(r["per_station"]))

with tabs[3]:
    c = get("/cities")
    st.dataframe(pd.DataFrame(c["table"]))
    st.json(c["stats"])

with tabs[4]:
    st.dataframe(pd.DataFrame(get("/results")))