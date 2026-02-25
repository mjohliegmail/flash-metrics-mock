import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Relativity • Daily Flash Metrics (Prototype)",
    page_icon="📈",
    layout="wide",
)

# ----------------------------
# Theme / style helpers
# ----------------------------
def apply_theme(dark: bool):
    """
    Streamlit doesn't fully support runtime theme switching like a SPA,
    but we can get a very solid "dark vs light" feel by injecting CSS.
    """
    if dark:
        bg = "#0b0f17"
        card = "#111827"
        card2 = "#0f172a"
        text = "#e5e7eb"
        subtle = "#9ca3af"
        border = "rgba(255,255,255,0.08)"
        accent = "#60a5fa"
        chip = "rgba(96,165,250,0.18)"
    else:
        bg = "#f6f7fb"
        card = "#ffffff"
        card2 = "#fbfcff"
        text = "#0f172a"
        subtle = "#64748b"
        border = "rgba(15,23,42,0.10)"
        accent = "#2563eb"
        chip = "rgba(37,99,235,0.12)"

    css = f"""
    <style>
      .stApp {{
        background: {bg};
        color: {text};
      }}

      /* reduce top padding a touch */
      .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 2.0rem;
      }}

      /* Headings */
      h1, h2, h3, h4, h5, h6 {{
        color: {text};
        letter-spacing: -0.02em;
      }}

      /* “Card” container */
      .dfm-card {{
        background: linear-gradient(180deg, {card}, {card2});
        border: 1px solid {border};
        border-radius: 16px;
        padding: 14px 14px 10px 14px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.18);
      }}

      .dfm-card-title {{
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 2px;
      }}

      .dfm-card-subtle {{
        color: {subtle};
        font-size: 0.85rem;
        margin-bottom: 10px;
      }}

      .dfm-chip {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: {chip};
        border: 1px solid {border};
        color: {text};
        font-size: 0.8rem;
        margin-left: 8px;
      }}

      /* Buttons */
      .stButton > button {{
        border-radius: 12px;
        border: 1px solid {border};
        padding: 0.5rem 0.75rem;
      }}

      /* Dataframe */
      div[data-testid="stDataFrame"] {{
        border: 1px solid {border};
        border-radius: 12px;
        overflow: hidden;
      }}

      /* Links */
      a {{
        color: {accent};
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


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
            # Engagement / usage
            "logged_in_users": series(12000, drift=0.18, vol=350, weekly_season=0.10).round(),
            "document_views": series(480000, drift=0.12, vol=22000, weekly_season=0.12).round(),
            "search_queries": series(220000, drift=0.14, vol=16000, weekly_season=0.10).round(),

            # Data movement
            "data_imported_gb": series(7800, drift=0.22, vol=450, weekly_season=0.08).round(1),
            "data_exported_gb": series(2400, drift=0.10, vol=220, weekly_season=0.06).round(1),

            # AI / coding
            "coding_decisions_manual": series(320000, drift=-0.03, vol=18000, weekly_season=0.10).round(),
            "coding_decisions_air": series(140000, drift=0.35, vol=12000, weekly_season=0.10).round(),

            # Legacy / platform inventory
            "legacy_invariant_dbs": series(820, drift=-0.10, vol=6, weekly_season=0.00).round(),

            # Processing pipeline by stage (toy example)
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
# Aggregation helpers
# ----------------------------
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


def get_current_value(series: pd.Series, period: str):
    agg = resample_for_period(series.to_frame(), period)[series.name]
    if len(agg) >= 2:
        return agg.iloc[-2], agg.tail(12)  # last complete bucket + spark
    return agg.iloc[-1], agg.tail(12)


def compute_wow_delta(series: pd.Series):
    w = series.resample("W").sum()
    if len(w) < 3:
        return None
    last = w.iloc[-2]  # last complete week
    prev = w.iloc[-3]
    delta = last - prev
    pct = (delta / prev * 100.0) if prev != 0 else np.nan
    return delta, pct


# ----------------------------
# UI components
# ----------------------------
def metric_card(title: str, col: str, fmt: str, unit: str, period: str):
    s = DATA[col]
    s.name = col
    current, spark = get_current_value(s, period)
    wow = compute_wow_delta(s)

    delta_txt = ""
    if wow is not None:
        d, p = wow
        if np.isnan(p):
            delta_txt = f"{d:+,.0f} WoW"
        else:
            delta_txt = f"{d:+,.0f} WoW ({p:+.1f}%)"

    st.markdown(
        f"""
        <div class="dfm-card">
          <div class="dfm-card-title">{title}</div>
          <div class="dfm-card-subtle">{period} trend <span class="dfm-chip">Dummy</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top = st.columns([2, 1], vertical_alignment="center")
    with top[0]:
        st.metric("Current", fmt.format(current) + unit, delta_txt if delta_txt else None)
    with top[1]:
        st.caption("")

    st.line_chart(spark, height=120)


def product_area_header(name: str, description: str):
    st.markdown(f"## {name}")
    st.caption(description)


# ----------------------------
# Sidebar: dummy filters + nav
# ----------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
if "selected_area" not in st.session_state:
    st.session_state["selected_area"] = list(PRODUCT_AREAS.keys())[0]
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True  # default dark

st.sidebar.title("Daily Flash")

# Theme toggle (default dark)
st.session_state["dark_mode"] = st.sidebar.toggle("Dark mode", value=st.session_state["dark_mode"])
apply_theme(st.session_state["dark_mode"])

st.sidebar.divider()

# Dummy filters
st.sidebar.subheader("Filters (dummy)")
segment = st.sidebar.selectbox("Segment", ["All", "Enterprise", "Mid-market", "SMB"], index=0)
region = st.sidebar.selectbox("Region", ["All", "NA", "EMEA", "APAC", "LATAM"], index=0)
account = st.sidebar.selectbox("Account", ["All", "Acme Co.", "Globex", "Initech", "Umbrella"], index=0)

st.sidebar.caption(f"Applied: {segment} • {region} • {account}")

st.sidebar.divider()

# Navigation: Home + product areas
nav = st.sidebar.radio(
    "Navigate",
    ["Home"] + list(PRODUCT_AREAS.keys()),
    index=(["Home"] + list(PRODUCT_AREAS.keys())).index(
        st.session_state["page"] if st.session_state["page"] in (["Home"] + list(PRODUCT_AREAS.keys())) else "Home"
    ),
)
st.session_state["page"] = nav

# Global trend granularity
period = st.sidebar.selectbox("Trend granularity", ["Weekly", "Monthly", "Quarterly"], index=0)

# ----------------------------
# Pages
# ----------------------------
def home_page():
    st.title("📈 Daily Flash Metrics")
    st.caption("Landing view grouped by Product Area • Dummy data • Clean trends")

    # Quick overview: product area tiles
    st.markdown("### Product areas")
    cols = st.columns(3)
    areas = list(PRODUCT_AREAS.items())

    for i, (area, cfg) in enumerate(areas):
        with cols[i % 3]:
            with st.container():
                st.markdown(
                    f"""
                    <div class="dfm-card">
                      <div class="dfm-card-title">{area}</div>
                      <div class="dfm-card-subtle">{cfg["description"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Open {area}", key=f"open_{area}"):
                    st.session_state["page"] = area
                    st.rerun()

    st.divider()

    # Grouped metrics by product area (compact + uncluttered)
    st.markdown("### Key metrics by area")
    for area, cfg in PRODUCT_AREAS.items():
        with st.expander(area, expanded=False):
            st.caption(cfg["description"])
            metric_items = list(cfg["metrics"].items())

            grid = st.columns(3)
            for j, (m_name, m_cfg) in enumerate(metric_items):
                with grid[j % 3]:
                    metric_card(m_name, m_cfg["col"], m_cfg["format"], m_cfg.get("unit", ""), period)

            st.divider()
            left, right = st.columns([1, 2], vertical_alignment="center")
            with left:
                if st.button(f"Go to {area} page →", key=f"go_{area}"):
                    st.session_state["page"] = area
                    st.rerun()
            with right:
                st.caption("Use the area page for drilldown dashboards and a cleaner focused trend view.")


def product_area_page(area: str):
    cfg = PRODUCT_AREAS[area]
    product_area_header(area, cfg["description"])

    # Highlight: choose one focus metric in this area (keeps it uncluttered)
    metric_names = list(cfg["metrics"].keys())
    focus_metric = st.selectbox("Focus metric", metric_names, index=0)
    focus_col = cfg["metrics"][focus_metric]["col"]

    left, right = st.columns([2, 1], vertical_alignment="top")

    with left:
        st.markdown("### Focus trend")
        df = resample_for_period(DATA[[focus_col]], period)
        st.line_chart(df[focus_col], height=320)

        st.markdown("### KPI cards")
        grid = st.columns(3)
        for i, (m_name, m_cfg) in enumerate(cfg["metrics"].items()):
            with grid[i % 3]:
                metric_card(m_name, m_cfg["col"], m_cfg["format"], m_cfg.get("unit", ""), period)

    with right:
        st.markdown("### Drilldown dashboards")
        st.caption("Links below are placeholders—swap in real Grafana / Databricks BI Genie URLs.")
        for label, url in cfg.get("dashboards", {}).items():
            st.link_button(label, url)

        st.divider()
        st.markdown("### Filters (dummy)")
        st.caption("These don’t change data yet; they’re here to validate UX.")
        st.write(
            {
                "Segment": segment,
                "Region": region,
                "Account": account,
            }
        )

        st.divider()
        if st.button("← Back to Home"):
            st.session_state["page"] = "Home"
            st.rerun()


# Router
if st.session_state["page"] == "Home":
    home_page()
else:
    product_area_page(st.session_state["page"])
