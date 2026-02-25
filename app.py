import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

st.set_page_config(
    page_title="RelativityOne Daily Flash Metric Report",
    page_icon="📈",
    layout="wide",
)

# ----------------------------
# Branding / styling
# ----------------------------
BRAND_ORANGE = "#EF5F11"
SIDEBAR_BG = "#FCDFCF"

st.markdown(
    f"""
    <style>
      /* Sidebar background */
      [data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG};
      }}

      /* Brand headers */
      h1, h2, h3 {{
        color: {BRAND_ORANGE} !important;
      }}

      /* Avoid header clipping across browsers/layouts */
      h1 {{
        line-height: 1.15 !important;
        margin-top: 0 !important;
        padding-top: 0.25rem !important;
      }}

      /* Give the page enough breathing room so the header doesn't clip */
      .block-container {{
        padding-top: 2.2rem;
        padding-bottom: 2rem;
      }}

      /* Slightly punch up metric name text */
      .metric-name {{
        font-weight: 800;
        font-size: 1.05rem;
        margin-bottom: 0.25rem;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

def render_app_header():
    """Top header with logo + branded title."""
    logo_path = Path(__file__).parent / "assets" / "relativity-logo.png"

    with st.container():
        c1, c2 = st.columns([1, 14], vertical_alignment="center")
        with c1:
            if logo_path.exists():
                st.image(str(logo_path), width=44)
            else:
                st.write("")

        with c2:
            st.markdown(
                f"""
                <h1 style="
                    margin:0;
                    padding:0.25rem 0 0 0;
                    color:{BRAND_ORANGE};
                    line-height:1.15;">
                  RelativityOne Daily Flash Metric Report
                </h1>
                """,
                unsafe_allow_html=True,
            )

    # small spacer to prevent overlap/clipping in some layouts
    st.write("")


# ----------------------------
# Dummy data generation
# ----------------------------
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

# ----------------------------
# Product areas & metrics model
# ----------------------------
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

# ----------------------------
# Helpers
# ----------------------------
def resample_for_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "Weekly":
        return df.resample("W").sum(numeric_only=True)
    if period == "Monthly":
        return df.resample("M").sum(numeric_only=True)
    if period == "Quarterly":
        return df.resample("Q").sum(numeric_only=True)
    return df


def yesterday_value(series: pd.Series):
    if len(series) < 2:
        return float(series.iloc[-1])
    return float(series.iloc[-2])


def wow_delta(series: pd.Series):
    w = series.resample("W").sum()
    if len(w) < 3:
        return None, None
    last = w.iloc[-2]
    prev = w.iloc[-3]
    delta = float(last - prev)
    pct = float((delta / prev) * 100.0) if prev != 0 else np.nan
    return delta, pct


def spark_for_period(series: pd.Series, period: str):
    df = resample_for_period(series.to_frame(name="v"), period)
    return df["v"].tail(12)


def metric_tile(metric_name: str, m_cfg: dict, period: str):
    col = m_cfg["col"]
    s = DATA[col].copy()

    y = yesterday_value(s)
    d, p = wow_delta(s)
    spark = spark_for_period(s, period)

    delta_str = None
    if d is not None:
        if np.isnan(p):
            delta_str = f"{d:+,.0f} WoW"
        else:
            delta_str = f"{d:+,.0f} WoW ({p:+.1f}%)"

    with st.container(border=True):
        st.markdown(f"<div class='metric-name'>{metric_name}</div>", unsafe_allow_html=True)
        st.metric("Yesterday", m_cfg["format"].format(y) + m_cfg.get("unit", ""), delta_str)
        st.line_chart(spark, height=120)


# ----------------------------
# Mock “local dashboards” per product area
# ----------------------------
@st.cache_data
def make_mock_breakdowns(seed: int = 11):
    rng = np.random.default_rng(seed)
    segments = ["Enterprise", "Mid-market", "SMB"]
    regions = ["NA", "EMEA", "APAC", "LATAM"]
    accounts = ["Acme Co.", "Globex", "Initech", "Umbrella", "Soylent"]

    seg = pd.Series(rng.uniform(0.8, 1.3, len(segments)), index=segments)
    reg = pd.Series(rng.uniform(0.7, 1.4, len(regions)), index=regions)
    acc = pd.Series(rng.uniform(0.5, 1.6, len(accounts)), index=accounts).sort_values(ascending=False)

    top_items = pd.DataFrame(
        {"Item": [f"Item {i}" for i in range(1, 8)],
         "Count": np.maximum(0, rng.normal(1000, 260, 7)).round().astype(int)}
    ).sort_values("Count", ascending=False)

    return seg, reg, acc, top_items


SEG_W, REG_W, ACC_W, TOP_ITEMS = make_mock_breakdowns()


def local_dashboards(area: str, period: str):
    left, right = st.columns([2, 1], vertical_alignment="top")

    with left:
        st.subheader("In-app dashboards (mock)")

        if area == "Adoption & Engagement":
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("**Active users by region (index)**")
                    st.bar_chart((REG_W * 100).round(0))
                    st.caption("Illustrative regional mix; not tied to real filters yet.")
            with c2:
                with st.container(border=True):
                    st.markdown("**Engagement depth (7-day rolling)**")
                    s = DATA["logged_in_users"].rolling(7).mean()
                    st.line_chart(resample_for_period(s.to_frame("v"), period)["v"], height=220)

        elif area == "Data Movement":
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("**Import volume vs export volume**")
                    df = DATA[["data_imported_gb", "data_exported_gb"]].copy()
                    df = resample_for_period(df, period)
                    st.line_chart(df, height=220)
            with c2:
                with st.container(border=True):
                    st.markdown("**Top “data movers” (mock accounts)**")
                    st.bar_chart((ACC_W * 100).round(0).head(5))

        elif area == "Review Experience":
            with st.container(border=True):
                st.markdown("**Document views trend**")
                df = resample_for_period(DATA[["document_views"]], period)
                st.line_chart(df["document_views"], height=260)
            with st.container(border=True):
                st.markdown("**Top workspaces (mock)**")
                tbl = TOP_ITEMS.rename(columns={"Item": "Workspace"}).copy()
                st.dataframe(tbl, use_container_width=True, hide_index=True)

        elif area == "Search":
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("**Search query volume**")
                    df = resample_for_period(DATA[["search_queries"]], period)
                    st.line_chart(df["search_queries"], height=220)
            with c2:
                with st.container(border=True):
                    st.markdown("**Top query categories (mock)**")
                    cats = pd.Series({"People": 1.15, "Concept": 0.95, "Keyword": 1.05, "Filters": 0.88}) * 100
                    st.bar_chart(cats.round(0))

        elif area == "Coding":
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("**Manual vs aiR decisions**")
                    df = DATA[["coding_decisions_manual", "coding_decisions_air"]].copy()
                    df = resample_for_period(df, period)
                    st.line_chart(df, height=220)
            with c2:
                with st.container(border=True):
                    st.markdown("**aiR share (mock KPI)**")
                    air_y = yesterday_value(DATA["coding_decisions_air"])
                    man_y = yesterday_value(DATA["coding_decisions_manual"])
                    share = air_y / max(1.0, (air_y + man_y))
                    st.metric("Yesterday aiR share", f"{share*100:,.1f}%")
                    st.progress(min(1.0, max(0.0, share)))

        elif area == "Platform Inventory":
            with st.container(border=True):
                st.markdown("**Legacy invariant DBs (trend)**")
                df = resample_for_period(DATA[["legacy_invariant_dbs"]], period)
                st.line_chart(df["legacy_invariant_dbs"], height=260)
            with st.container(border=True):
                st.markdown("**Inventory composition (mock)**")
                comp = pd.Series({"Invariant": 0.62, "Transient": 0.26, "Other": 0.12}) * 100
                st.bar_chart(comp.round(0))

        else:  # Processing Pipeline
            with st.container(border=True):
                st.markdown("**Throughput by stage**")
                df = DATA[["processed_stage_ingest_gb", "processed_stage_index_gb", "processed_stage_analytics_gb"]].copy()
                df = resample_for_period(df, period)
                st.line_chart(df, height=260)
            with st.container(border=True):
                st.markdown("**Bottleneck watch (mock)**")
                score = min(100, max(0, 55 + np.random.default_rng(3).normal(0, 12)))
                st.metric("Index stage health score", f"{score:,.0f}/100")
                st.progress(score / 100)

    with right:
        st.subheader("External dashboards")
        st.caption("These are placeholders — swap in real Grafana / Databricks / etc.")
        for label, url in PRODUCT_AREAS[area].get("dashboards", {}).items():
            st.link_button(label, url)

        st.divider()
        st.subheader("Applied filters (dummy)")
        st.write({"Segment": segment, "Region": region, "Account": account})


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
    render_app_header()
    st.caption("Landing view: key metrics grouped by product area (dummy data).")

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
    render_app_header()
    st.markdown(f"## {area}")
    st.caption(PRODUCT_AREAS[area]["description"])

    tabs = st.tabs(["Overview", "Dashboards"])

    with tabs[0]:
        metric_names = list(PRODUCT_AREAS[area]["metrics"].keys())
        focus_metric = st.selectbox("Focus metric", metric_names, index=0)
        focus_col = PRODUCT_AREAS[area]["metrics"][focus_metric]["col"]
        df = resample_for_period(DATA[[focus_col]], period)
        st.line_chart(df[focus_col], height=320)

        st.subheader("Key metrics")
        grid = st.columns(3)
        for i, (m_name, m_cfg) in enumerate(PRODUCT_AREAS[area]["metrics"].items()):
            with grid[i % 3]:
                metric_tile(m_name, m_cfg, period)

    with tabs[1]:
        local_dashboards(area, period)

    st.divider()
    if st.button("← Back to Home"):
        st.session_state["page"] = "Home"
        st.rerun()


if st.session_state["page"] == "Home":
    home_page()
else:
    area_page(st.session_state["page"])
