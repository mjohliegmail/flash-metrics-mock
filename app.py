import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

st.set_page_config(
    page_title="Relativity • Daily Flash Metrics (Prototype)",
    page_icon="📈",
    layout="wide",
)

@st.cache_data
def make_dummy_timeseries(days: int = 365, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(date.today())
    idx = pd.date_range(end=end, periods=days, freq="D")

    def series(base, drift=0.0, vol=1.0, weekly_season=0.15):
        t = np.arange(len(idx))
        season = 1.0 + weekly_season * np.sin(2 * np.pi * t / 7.0)
        noise = rng.normal(0, vol, size=len(idx))
        trend = drift * (t / len(idx))
        s = (base * season) + noise + (base * trend)
        return np.maximum(0, s)

    df = pd.DataFrame(
        {
            "logged_in_users": series(12000, drift=0.18, vol=350, weekly_season=0.10).round(),
            "document_views": series(480000, drift=0.12, vol=22000, weekly_season=0.12).round(),
            "search_queries": series(220000, drift=0.14, vol=16000, weekly_season=0.10).round(),
            "data_imported_gb": series(7800, drift=0.22, vol=450, weekly_season=0.08).round(1),
            "data_exported_gb": series(2400, drift=0.10, vol=220, weekly_season=0.06).round(1),
            "coding_decisions_manual": series(320000, drift=-0.03, vol=18000, weekly_season=0.10).round(),
            "coding_decisions_air": series(140000, drift=0.35, vol=12000, weekly_season=0.10).round(),
            "legacy_invariant_dbs": series(820, drift=-0.10, vol=6, weekly_season=0.00).round(),
            "processed_stage_ingest_gb": series(6200, drift=0.20, vol=380, weekly_season=0.08).round(1),
            "processed_stage_index_gb": series(5400, drift=0.18, vol=360, weekly_season=0.08).round(1),
            "processed_stage_analytics_gb": series(3100, drift=0.16, vol=280, weekly_season=0.08).round(1),
        },
        index=idx,
    )
    df.index.name = "date"
    return df


DATA = make_dummy_timeseries()

PRODUCT_AREAS = {
    "Adoption & Engagement": {
        "description": "Top-of-funnel usage + recurring engagement signals.",
        "dashboards": {
            "Grafana — Engagement Overview": "https://grafana.example.com/d/engagement",
            "Databricks BI Genie — Adoption": "https://databricks.example.com/bi-genie/adoption",
        },
        "metrics": {
            "Logged-in users": {"col": "logged_in_users", "format": "{:,.0f}", "unit": ""},
        },
    },
    "Data Movement": {
        "description": "Imports/exports volume and related throughput signals.",
        "dashboards": {
            "Grafana — Ingest": "https://grafana.example.com/d/ingest",
            "Databricks BI Genie — Imports": "https://databricks.example.com/bi-genie/imports",
            "Grafana — Exports": "https://grafana.example.com/d/exports",
        },
        "metrics": {
            "Data imported": {"col": "data_imported_gb", "format": "{:,.1f}", "unit": " GB"},
            "Data exported": {"col": "data_exported_gb", "format": "{:,.1f}", "unit": " GB"},
        },
    },
    "Review Experience": {
        "description": "Review activity and core workflow intensity.",
        "dashboards": {
            "Databricks BI Genie — Review UX": "https://databricks.example.com/bi-genie/review-ux",
        },
        "metrics": {
            "Document views": {"col": "document_views", "format": "{:,.0f}", "unit": ""},
        },
    },
    "Search": {
        "description": "Search volume and usage health indicators.",
        "dashboards": {
            "Grafana — Search": "https://grafana.example.com/d/search",
        },
        "metrics": {
            "Search queries": {"col": "search_queries", "format": "{:,.0f}", "unit": ""},
        },
    },
    "Coding": {
        "description": "Manual vs aiR-assisted coding activity and mix shifts.",
        "dashboards": {
            "Databricks BI Genie — Coding": "https://databricks.example.com/bi-genie/coding",
            "Databricks BI Genie — aiR": "https://databricks.example.com/bi-genie/air",
        },
        "metrics": {
            "Coding decisions (Manual)": {"col": "coding_decisions_manual", "format": "{:,.0f}", "unit": ""},
            "Coding decisions (aiR)": {"col": "coding_decisions_air", "format": "{:,.0f}", "unit": ""},
        },
    },
    "Platform Inventory": {
        "description": "Legacy / footprint signals and platform inventory.",
        "dashboards": {
            "Grafana — Inventory": "https://grafana.example.com/d/inventory",
        },
        "metrics": {
            "Legacy invariant DBs": {"col": "legacy_invariant_dbs", "format": "{:,.0f}", "unit": ""},
        },
    },
    "Processing Pipeline": {
        "description": "Throughput by pipeline stage; watch for bottlenecks.",
        "dashboards": {
            "Databricks BI Genie — Pipeline": "https://databricks.example.com/bi-genie/pipeline",
            "Databricks BI Genie — Indexing": "https://databricks.example.com/bi-genie/indexing",
            "Databricks BI Genie — Analytics": "https://databricks.example.com/bi-genie/analytics",
        },
        "metrics": {
            "Processed (Ingest stage)": {"col": "processed_stage_ingest_gb", "format": "{:,.1f}", "unit": " GB"},
            "Processed (Index stage)": {"col": "processed_stage_index_gb", "format": "{:,.1f}", "unit": " GB"},
            "Processed (Analytics stage)": {"col": "processed_stage_analytics_gb", "format": "{:,.1f}", "unit": " GB"},
        },
    },
}


def resample_for_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "Daily":
        return df
    if period == "Weekly":
        return df.resample("W").sum(numeric_only=True)
    if period == "Monthly":
        return df.resample("M").sum(numeric_only=True)
    if period == "Quarterly":
        return df.resample("Q").sum(numeric_only=True)
    return df


def current_and_spark(series: pd.Series, period: str):
    agg = resample_for_period(series.to_frame(), period)[series.name]
    current = agg.iloc[-2] if len(agg) >= 2 else agg.iloc[-1]  # last complete bucket if possible
    spark = agg.tail(12)
    return current, spark


def wow_delta(series: pd.Series):
    w = series.resample("W").sum()
    if len(w) < 3:
        return None, None
    last = w.iloc[-2]
    prev = w.iloc[-3]
    delta = float(last - prev)
    pct = float((delta / prev) * 100.0) if prev != 0 else np.nan
    return delta, pct


def metric_tile(metric_name: str, m_cfg: dict, period: str):
    col = m_cfg["col"]
    s = DATA[col].copy()
    s.name = col

    cur, spark = current_and_spark(s, period)
    d, p = wow_delta(s)

    delta_str = None
    if d is not None:
        if np.isnan(p):
            delta_str = f"{d:+,.0f} WoW"
        else:
            delta_str = f"{d:+,.0f} WoW ({p:+.1f}%)"

    with st.container(border=True):
        st.caption(metric_name)
        st.metric("Current", m_cfg["format"].format(cur) + m_cfg.get("unit", ""), delta_str)
        st.line_chart(spark, height=120)


# ----------------------------
# Sidebar: dummy filters + navigation
# ----------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

st.sidebar.title("Daily Flash")

st.sidebar.subheader("Filters (dummy)")
segment = st.sidebar.selectbox("Segment", ["All", "Enterprise", "Mid-market", "SMB"], index=0)
region = st.sidebar.selectbox("Region", ["All", "NA", "EMEA", "APAC", "LATAM"], index=0)
account = st.sidebar.selectbox("Account", ["All", "Acme Co.", "Globex", "Initech", "Umbrella"], index=0)
st.sidebar.caption(f"Applied: {segment} • {region} • {account}")

st.sidebar.divider()

period = st.sidebar.selectbox("Trend granularity", ["Weekly", "Monthly", "Quarterly"], index=0)

nav = st.sidebar.radio("Navigate", ["Home"] + list(PRODUCT_AREAS.keys()), index=0)
st.session_state["page"] = nav


# ----------------------------
# Pages
# ----------------------------
def home_page():
    st.title("📈 Daily Flash Metrics")
    st.caption("Landing view: high-level trends + key metrics grouped by product area (dummy data).")

    # High-level trend focus (single chart to avoid clutter)
    st.subheader("High-level trend (focus)")
    focus_choices = []
    for area_name, area_cfg in PRODUCT_AREAS.items():
        for m_name, m_cfg in area_cfg["metrics"].items():
            focus_choices.append((f"{area_name} • {m_name}", m_cfg["col"]))

    focus_label = st.selectbox("Select a metric", [x[0] for x in focus_choices], index=0)
    focus_col = dict(focus_choices)[focus_label]
    df = resample_for_period(DATA[[focus_col]], period)
    st.line_chart(df[focus_col], height=320)

    st.divider()

    # Product areas (the ONLY place key metrics appear on landing)
    st.subheader("Product areas")
    for area, cfg in PRODUCT_AREAS.items():
        with st.container(border=True):
            header = st.columns([3, 1], vertical_alignment="center")
            with header[0]:
                st.markdown(f"**{area}**")
                st.caption(cfg["description"])
            with header[1]:
                if st.button("Explore more data", key=f"explore_{area}"):
                    st.session_state["page"] = area
                    st.rerun()

            metric_items = list(cfg["metrics"].items())
            grid = st.columns(3)
            for j, (m_name, m_cfg) in enumerate(metric_items):
                with grid[j % 3]:
                    metric_tile(m_name, m_cfg, period)


def area_page(area: str):
    cfg = PRODUCT_AREAS[area]
    st.title(area)
    st.caption(cfg["description"])

    tabs = st.tabs(["Overview", "Dashboards & links"])

    with tabs[0]:
        metric_names = list(cfg["metrics"].keys())
        focus_metric = st.selectbox("Focus metric", metric_names, index=0)
        focus_col = cfg["metrics"][focus_metric]["col"]
        df = resample_for_period(DATA[[focus_col]], period)
        st.line_chart(df[focus_col], height=320)

        st.subheader("Key metrics")
        grid = st.columns(3)
        for i, (m_name, m_cfg) in enumerate(cfg["metrics"].items()):
            with grid[i % 3]:
                metric_tile(m_name, m_cfg, period)

    with tabs[1]:
        st.subheader("Drilldown dashboards")
        st.caption("Replace placeholders with real links (Grafana / Databricks BI Genie / other).")
        for label, url in cfg.get("dashboards", {}).items():
            st.link_button(label, url)

        st.divider()
        st.subheader("Applied filters (dummy)")
        st.write({"Segment": segment, "Region": region, "Account": account})

    st.divider()
    if st.button("← Back to Home"):
        st.session_state["page"] = "Home"
        st.rerun()


if st.session_state["page"] == "Home":
    home_page()
else:
    area_page(st.session_state["page"])
