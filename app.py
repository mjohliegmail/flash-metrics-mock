import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ----------------------------
# App config
# ----------------------------
st.set_page_config(
    page_title="Relativity • Daily Flash Metrics (Prototype)",
    page_icon="📈",
    layout="wide",
)

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

METRICS = {
    "Logged-in users": {
        "col": "logged_in_users",
        "format": "{:,.0f}",
        "unit": "",
        "domain": "Adoption & Engagement",
        "description": "Unique logged-in users (daily).",
        "external_links": {
            "Grafana (Engagement)": "https://grafana.example.com/d/engagement",
            "Databricks BI Genie (Adoption)": "https://databricks.example.com/bi-genie/adoption",
        },
    },
    "Data imported": {
        "col": "data_imported_gb",
        "format": "{:,.1f}",
        "unit": " GB",
        "domain": "Data Movement",
        "description": "Total data imported per day (GB).",
        "external_links": {
            "Grafana (Ingest)": "https://grafana.example.com/d/ingest",
            "Databricks BI Genie (Imports)": "https://databricks.example.com/bi-genie/imports",
        },
    },
    "Data exported": {
        "col": "data_exported_gb",
        "format": "{:,.1f}",
        "unit": " GB",
        "domain": "Data Movement",
        "description": "Total data exported per day (GB).",
        "external_links": {
            "Grafana (Exports)": "https://grafana.example.com/d/exports",
        },
    },
    "Document views": {
        "col": "document_views",
        "format": "{:,.0f}",
        "unit": "",
        "domain": "Review Experience",
        "description": "Total document views (daily).",
        "external_links": {
            "Databricks BI Genie (Review UX)": "https://databricks.example.com/bi-genie/review-ux",
        },
    },
    "Search queries": {
        "col": "search_queries",
        "format": "{:,.0f}",
        "unit": "",
        "domain": "Search",
        "description": "Total search queries executed (daily).",
        "external_links": {
            "Grafana (Search)": "https://grafana.example.com/d/search",
        },
    },
    "Coding decisions (Manual)": {
        "col": "coding_decisions_manual",
        "format": "{:,.0f}",
        "unit": "",
        "domain": "Coding",
        "description": "Manual coding decisions (daily).",
        "external_links": {
            "Databricks BI Genie (Coding)": "https://databricks.example.com/bi-genie/coding",
        },
    },
    "Coding decisions (aiR)": {
        "col": "coding_decisions_air",
        "format": "{:,.0f}",
        "unit": "",
        "domain": "Coding",
        "description": "aiR-assisted coding decisions (daily).",
        "external_links": {
            "Databricks BI Genie (aiR)": "https://databricks.example.com/bi-genie/air",
        },
    },
    "Legacy invariant DBs": {
        "col": "legacy_invariant_dbs",
        "format": "{:,.0f}",
        "unit": "",
        "domain": "Platform Inventory",
        "description": "Total legacy invariant databases (daily snapshot).",
        "external_links": {
            "Grafana (Inventory)": "https://grafana.example.com/d/inventory",
        },
    },
    "Processed (Ingest stage)": {
        "col": "processed_stage_ingest_gb",
        "format": "{:,.1f}",
        "unit": " GB",
        "domain": "Processing Pipeline",
        "description": "Data processed at ingest stage (GB/day).",
        "external_links": {
            "Databricks BI Genie (Pipeline)": "https://databricks.example.com/bi-genie/pipeline",
        },
    },
    "Processed (Index stage)": {
        "col": "processed_stage_index_gb",
        "format": "{:,.1f}",
        "unit": " GB",
        "domain": "Processing Pipeline",
        "description": "Data processed at indexing stage (GB/day).",
        "external_links": {
            "Databricks BI Genie (Index)": "https://databricks.example.com/bi-genie/indexing",
        },
    },
    "Processed (Analytics stage)": {
        "col": "processed_stage_analytics_gb",
        "format": "{:,.1f}",
        "unit": " GB",
        "domain": "Processing Pipeline",
        "description": "Data processed at analytics stage (GB/day).",
        "external_links": {
            "Databricks BI Genie (Analytics)": "https://databricks.example.com/bi-genie/analytics",
        },
    },
}


def period_slice(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "Weekly":
        return df.last("28D")  # show 4 weeks for context
    if period == "Monthly":
        return df.last("180D")  # ~6 months
    if period == "Quarterly":
        return df.last("365D")  # 12 months
    return df


def resample_for_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "Weekly":
        return df.resample("W").sum(numeric_only=True)
    if period == "Monthly":
        return df.resample("M").sum(numeric_only=True)
    if period == "Quarterly":
        return df.resample("Q").sum(numeric_only=True)
    return df


def compute_deltas(s: pd.Series):
    # Compare last complete window vs prior window based on weekly buckets
    w = s.resample("W").sum()
    if len(w) < 3:
        return None, None
    last = w.iloc[-2]          # last completed week
    prev = w.iloc[-3]
    delta = last - prev
    pct = (delta / prev * 100.0) if prev != 0 else np.nan
    return float(delta), float(pct)


# ----------------------------
# UI components
# ----------------------------
def kpi_tile(title: str, cfg: dict, period: str):
    col = cfg["col"]
    window = period_slice(DATA[[col]], period)[col]
    roll = resample_for_period(window.to_frame(), period)[col]

    # Current value: last completed bucket (week/month/quarter)
    if len(roll) >= 2:
        current = roll.iloc[-2]
    else:
        current = roll.iloc[-1]

    delta, pct = compute_deltas(DATA[col])

    # Tiny sparkline input: last N buckets
    spark = roll.tail(12)

    # Tile layout
    with st.container(border=True):
        top = st.columns([3, 2], vertical_alignment="center")
        with top[0]:
            st.markdown(f"**{title}**")
            st.caption(cfg["domain"])
        with top[1]:
            # Keep delta simple: show WoW-ish delta even if the period toggle is monthly/quarterly
            if delta is None:
                st.metric("Current", cfg["format"].format(current) + cfg.get("unit", ""))
            else:
                d_txt = f"{delta:,.0f}" if abs(delta) >= 10 else f"{delta:,.2f}"
                if not np.isnan(pct):
                    d_txt = f"{d_txt} ({pct:+.1f}%)"
                st.metric("Current", cfg["format"].format(current) + cfg.get("unit", ""), d_txt)

        st.line_chart(spark, height=110)

        cols = st.columns([1, 1, 1])
        with cols[0]:
            if st.button("View details", key=f"view_{title}"):
                st.session_state["page"] = "Metric details"
                st.session_state["metric_selected"] = title
                st.rerun()
        with cols[1]:
            st.link_button("Open external dashboards", "https://example.com")  # placeholder
        with cols[2]:
            st.caption(cfg["description"])


def landing_page():
    st.title("📈 Daily Flash Metrics (Prototype)")
    st.caption("Dummy data • Landing view across product areas with clean trends + clickthroughs.")

    c = st.columns([2, 1, 1, 1], vertical_alignment="center")
    with c[0]:
        st.markdown("### Snapshot")
    with c[1]:
        period = st.selectbox("Trend granularity", ["Weekly", "Monthly", "Quarterly"], index=0)
    with c[2]:
        domain_filter = st.selectbox(
            "Domain",
            ["All"] + sorted({m["domain"] for m in METRICS.values()}),
            index=0,
        )
    with c[3]:
        q = st.text_input("Search metrics", placeholder="e.g. search, aiR, ingest")

    # Filter metrics list
    def matches(title, cfg):
        if domain_filter != "All" and cfg["domain"] != domain_filter:
            return False
        if q and q.lower() not in (title + " " + cfg["domain"] + " " + cfg["description"]).lower():
            return False
        return True

    selected = [(t, cfg) for t, cfg in METRICS.items() if matches(t, cfg)]

    st.divider()

    # KPI grid
    if not selected:
        st.info("No metrics match your filters.")
        return

    # 3-column grid of tiles
    cols = st.columns(3)
    for i, (title, cfg) in enumerate(selected):
        with cols[i % 3]:
            kpi_tile(title, cfg, period)

    st.divider()

    # One uncluttered “focus” chart (pick one key metric)
    st.markdown("### Focus trend")
    focus = st.selectbox("Choose a metric", list(METRICS.keys()), index=0)
    focus_col = METRICS[focus]["col"]
    focus_unit = METRICS[focus].get("unit", "")
    df = period_slice(DATA[[focus_col]], period)
    df_roll = resample_for_period(df, period)

    st.line_chart(df_roll[focus_col], height=260)
    st.caption(f"{focus} ({period} aggregation){focus_unit}")


def metric_details_page():
    metric = st.session_state.get("metric_selected", list(METRICS.keys())[0])
    cfg = METRICS[metric]
    col = cfg["col"]

    st.title(f"🔎 {metric}")
    st.caption(cfg["description"])

    # Period controls
    c = st.columns([1, 1, 2], vertical_alignment="center")
    with c[0]:
        period = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly", "Quarterly"], index=1)
    with c[1]:
        lookback = st.selectbox("Lookback", ["30D", "90D", "180D", "365D"], index=1)
    with c[2]:
        st.write("")

    df = DATA[[col]].copy()
    if lookback == "30D":
        df = df.last("30D")
    elif lookback == "90D":
        df = df.last("90D")
    elif lookback == "180D":
        df = df.last("180D")
    else:
        df = df.last("365D")

    if period == "Weekly":
        df = df.resample("W").sum(numeric_only=True)
    elif period == "Monthly":
        df = df.resample("M").sum(numeric_only=True)
    elif period == "Quarterly":
        df = df.resample("Q").sum(numeric_only=True)

    # Page layout
    a, b = st.columns([2, 1], vertical_alignment="top")
    with a:
        st.subheader("Trend")
        st.line_chart(df[col], height=320)

        st.subheader("Recent values")
        st.dataframe(df.tail(15), use_container_width=True)

    with b:
        st.subheader("Dashboards")
        links = cfg.get("external_links", {})
        if not links:
            st.info("No external links configured.")
        else:
            for label, url in links.items():
                st.link_button(label, url)

        st.subheader("Navigation")
        if st.button("← Back to landing"):
            st.session_state["page"] = "Landing"
            st.rerun()

        st.divider()

        # Example: related internal clickthroughs (stub)
        st.caption("Example internal drilldowns (stubs):")
        st.button("Open domain dashboard (stub)", disabled=True)
        st.button("Open cohort view (stub)", disabled=True)


def domain_dashboards_page():
    st.title("🧭 Domain dashboards")
    st.caption("Stub list of domain pages and/or external dashboard links.")

    domains = {}
    for title, cfg in METRICS.items():
        domains.setdefault(cfg["domain"], []).append((title, cfg))

    for domain, items in sorted(domains.items()):
        with st.container(border=True):
            st.markdown(f"### {domain}")
            st.caption(f"{len(items)} metrics")

            # internal clickthrough options
            cc = st.columns([1, 2])
            with cc[0]:
                if st.button(f"Open {domain} (internal)", key=f"open_{domain}"):
                    st.session_state["page"] = "Landing"
                    st.rerun()
            with cc[1]:
                st.link_button(f"Open {domain} in Grafana (placeholder)", "https://grafana.example.com")

            # compact list
            for title, cfg in items[:6]:
                st.markdown(f"- **{title}** — {cfg['description']}")
            if len(items) > 6:
                st.caption(f"+ {len(items) - 6} more…")


# ----------------------------
# Lightweight page router
# ----------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Landing"
if "metric_selected" not in st.session_state:
    st.session_state["metric_selected"] = list(METRICS.keys())[0]

st.sidebar.title("Daily Flash")
page = st.sidebar.radio(
    "Navigate",
    ["Landing", "Metric details", "Domain dashboards"],
    index=["Landing", "Metric details", "Domain dashboards"].index(st.session_state["page"]),
)

st.session_state["page"] = page

if page == "Landing":
    landing_page()
elif page == "Metric details":
    metric_details_page()
else:
    domain_dashboards_page()