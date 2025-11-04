# app.py — Kansas Nutrition First — Leadership Dashboard (Streamlit)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Kansas Nutrition First — Leadership Dashboard", layout="wide")

# === Brand CSS & plot theme (colors + fonts) ===
BRAND = {
    "new_orleans": "#436DB3",   # primary
    "los_angeles": "#BFD0EE",   # light brand blue
    "honolulu":    "#F4454E",   # alert/red
    "palm":        "#F7F3EF",   # panel bg
    "detroit":     "#EDEDED",   # gridlines
    "link_blue":   "#0071BC",   # links
    "ink":         "#0B1221"    # text
}

# ---------------------------
# CSS Styling Fixes (title wrapping, padding)
# ---------------------------
st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700&display=swap");

html, body, [class*="css"]  {
  font-family: "Roboto", system-ui, -apple-system, Segoe UI, Arial, sans-serif !important;
  color: #0B1221;
}

/* ✨ Title & subhead: allow wrapping + avoid clipping */
.hero-title {
  font-weight: 300;
  font-size: 36px;
  line-height: 40px;
  margin: 0 0 .25rem 0;
  white-space: normal;
  word-break: break-word;
  overflow: visible;
  max-width: 100%;
}
.subhead {
  font-weight: 600;
  font-size: 18px;
  line-height: 22px;
  margin: 0 0 .5rem 0;
}

/* ✨ More breathing room at the top */
.block-container {
  padding-top: 2rem;
}

/* KPI cards */
[data-testid="stMetric"] > div {
  background: white;
  border: 1px solid #EDEDED;
  border-radius: 10px;
  padding: 12px 14px;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-weight: 500;
  font-size: 26px;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  color: #465264;
  font-size: 12px;
}

/* Links */
a, .markdown-text-container a {
  color: #0071BC !important;
  text-decoration: none;
}
a:hover { text-decoration: underline; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: white !important;
  border-right: 1px solid #EDEDED;
}

/* ✨ Responsive tweak for smaller widths */
@media (max-width: 1200px) {
  .hero-title { font-size: 30px; line-height: 34px; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Plotly defaults
# ---------------------------
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    BRAND["new_orleans"], BRAND["los_angeles"], BRAND["honolulu"],
    BRAND["detroit"], BRAND["link_blue"]
]

def brandify(fig, title=None):
    fig.update_layout(
        title=title,
        title_font=dict(size=20, family="Roboto", color=BRAND["ink"]),
        font=dict(family="Roboto", size=12, color=BRAND["ink"]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_xaxes(showgrid=True, gridcolor=BRAND["detroit"])
    fig.update_yaxes(showgrid=True, gridcolor=BRAND["detroit"])
    return fig

# ---------------------------
# Data loading
# ---------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["event_timestamp"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    for c in ["event_type","traffic_source","utm_campaign","device_type","browser","city","state"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

# ---------------------------
# Header Section (Fixed title)
# ---------------------------
st.markdown("""
<div id="brand-header" style="padding: 6px 0 10px 0;">
  <div class="hero-title">Kansas Nutrition First &mdash; Leadership Dashboard</div>
  <div class="subhead">Clinically-validated Nutrition First program — {}</div>
</div>
<br/>
""".format(date.today().strftime("%b %Y")), unsafe_allow_html=True)

# ---------------------------
# Load data
# ---------------------------
default_path = "data/kansas_nutrition_first_web_analytics_events.csv"
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if uploaded:
    data = load_data(uploaded)
else:
    try:
        data = load_data(default_path)
    except Exception as e:
        st.error(f"Could not load default CSV at {default_path}. Upload a CSV using the sidebar.\n\n{e}")
        st.stop()

# Kansas-only filter
data = data[data["state"].astype(str).str.lower() == "kansas"].copy()

# ---------------------------
# Helper functions
# ---------------------------
def count_events(df, etype):
    return int((df["event_type"] == etype).sum())

def rate(n, d):
    return 0.0 if d == 0 else round(100.0 * n / d, 1)

def same_length_prev_range(start_d: pd.Timestamp, end_d: pd.Timestamp):
    length = (end_d.normalize() - start_d.normalize()).days + 1
    prev_end = start_d - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=length-1)
    return prev_start, prev_end

def kpi_delta(curr, prev):
    if prev in (None, 0):
        return "0.0% vs prev", BRAND["detroit"]
    pct = round(100.0 * (curr - prev) / prev, 1)
    arrow = "▲" if pct >= 0 else "▼"
    color = BRAND["new_orleans"] if pct >= 0 else BRAND["honolulu"]
    return f"{arrow} {pct}%", color

# ---------------------------
# Sidebar + filters
# ---------------------------
min_d, max_d = data["event_date"].min(), data["event_date"].max()
date_range = st.sidebar.date_input("Date Range", (min_d, max_d), min_value=min_d, max_value=max_d)
if isinstance(date_range, tuple):
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_d, end_d = min_d, max_d

freq = st.sidebar.selectbox("Granularity", ["D","W","M"], index=1,
                            format_func=lambda x: {"D":"Daily","W":"Weekly","M":"Monthly"}[x])

all_cities = ["All"] + sorted(data["city"].dropna().astype(str).unique().tolist())
all_chans  = ["All"] + sorted(data["traffic_source"].dropna().astype(str).unique().tolist())
all_device = ["All"] + sorted(data["device_type"].dropna().astype(str).unique().tolist())

c1, c2, c3 = st.columns([1,1,1])
sel_city = c1.selectbox("Geography", options=all_cities, index=0)
sel_chan = c2.selectbox("Channel",   options=all_chans,  index=0)
sel_dev  = c3.selectbox("Device",    options=all_device, index=0)

mask_time = (data["event_date"] >= start_d) & (data["event_date"] <= end_d)
fdf = data.loc[mask_time].copy()
if sel_city != "All": fdf = fdf[fdf["city"] == sel_city]
if sel_chan != "All": fdf = fdf[fdf["traffic_source"] == sel_chan]
if sel_dev  != "All": fdf = fdf[fdf["device_type"] == sel_dev]

ps, pe = same_length_prev_range(start_d, end_d)
pmask = (data["event_date"] >= ps) & (data["event_date"] <= pe)
fprev = data.loc[pmask].copy()
if sel_city != "All": fprev = fprev[fprev["city"] == sel_city]
if sel_chan != "All": fprev = fprev[fprev["traffic_source"] == sel_chan]
if sel_dev  != "All": fprev = fprev[fprev["device_type"] == sel_dev]

# KPIs current & previous
crossovers   = count_events(fdf, "crossover")
clicks       = count_events(fdf, "link_click")
signups      = count_events(fdf, "signup")
improvements = count_events(fdf, "improvement")

pcross = count_events(fprev, "crossover")
pclick = count_events(fprev, "link_click")
psign  = count_events(fprev, "signup")
pimpr  = count_events(fprev, "improvement")

d_cross, _ = kpi_delta(crossovers, pcross)
d_click, _ = kpi_delta(clicks, pclick)
d_sign,  _ = kpi_delta(signups, psign)
d_impr,  _ = kpi_delta(improvements, pimpr)

# Unique users
uu_cross = fdf.loc[fdf["event_type"]=="crossover","user_id"].nunique()
uu_click = fdf.loc[fdf["event_type"]=="link_click","user_id"].nunique()
uu_sign  = fdf.loc[fdf["event_type"]=="signup","user_id"].nunique()
uu_impr  = fdf.loc[fdf["event_type"]=="improvement","user_id"].nunique()

# ---------------------------
# Tabs (Wireframe)
# ---------------------------
tab_exec, tab_acq, tab_conv, tab_out = st.tabs(
    ["Executive Overview", "Acquisition & Engagement", "Conversion", "Health Outcomes"]
)

# ===== Executive Overview =====
with tab_exec:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Website Crossovers", f"{crossovers:,}", d_cross)
    k2.metric("Virta Link Clicks",  f"{clicks:,}",     d_click)
    k3.metric("Virta Sign-ups",     f"{signups:,}",    d_sign)
    k4.metric("Health Improvements",f"{improvements:,}", d_impr)

    st.markdown("<br/>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.35])

    # LEFT: Funnel (event counts)
    with left:
        st.subheader("The \"Kansas Initiative\" Conversion Funnel")
        funnel_df = pd.DataFrame({
            "stage": ["Crossover", "Link Click", "Signup", "Improvement (events)"],
            "count": [crossovers, clicks, signups, improvements]
        })
        fig_funnel = px.funnel(funnel_df, x="count", y="stage")
        st.plotly_chart(brandify(fig_funnel, "Event Count Funnel"), use_container_width=True)

    # RIGHT: Key trends (Crossovers, Clicks, Sign-ups)
    with right:
        st.subheader("Key Metric Trends — Crossovers, Clicks, Sign-ups")
        byp = (fdf.assign(period=fdf["event_date"].dt.to_period(freq).dt.to_timestamp())
                 .groupby(["period","event_type"]).size().reset_index(name="count"))
        pivot = (byp.pivot(index="period", columns="event_type", values="count")
                   .fillna(0).rename(columns={"crossover":"Crossovers","link_click":"Clicks","signup":"Sign-ups"}))
        for col in ["Crossovers","Clicks","Sign-ups"]:
            if col not in pivot.columns:
                pivot[col] = 0
        fig_tr = px.line(pivot.reset_index(), x="period", y=["Crossovers","Clicks","Sign-ups"])
        st.plotly_chart(brandify(fig_tr, "Trends ({})".format({"D":"Daily","W":"Weekly","M":"Monthly"}[freq])),
                        use_container_width=True)

# ===== Acquisition & Engagement =====
with tab_acq:
    st.subheader("Acquisition & Engagement")
    c1, c2 = st.columns(2)
    by_src = (fdf.groupby("traffic_source").size().reset_index(name="count").sort_values("count", ascending=False))
    by_cmp = (fdf.groupby("utm_campaign").size().reset_index(name="count").sort_values("count", ascending=False))
    c1.plotly_chart(brandify(px.bar(by_src, x="traffic_source", y="count"), "Events by Channel"),
                    use_container_width=True)
    c2.plotly_chart(brandify(px.bar(by_cmp, x="utm_campaign", y="count"), "Events by Campaign"),
                    use_container_width=True)

# ===== Conversion =====
with tab_conv:
    st.subheader("Conversion")
    uu_df = pd.DataFrame({
        "stage": ["Crossover Users","Link Click Users","Signup Users","Improved Users"],
        "users": [uu_cross, uu_click, uu_sign, uu_impr]
    })
    fig_uu = px.funnel(uu_df, x="users", y="stage")
    st.plotly_chart(brandify(fig_uu, "Funnel — Unique Users"), use_container_width=True)

    byp = (fdf.assign(period=fdf["event_date"].dt.to_period(freq).dt.to_timestamp())
              .groupby(["period","event_type"]).size().reset_index(name="count"))
    base = (byp.pivot(index="period", columns="event_type", values="count")
              .fillna(0).rename(columns={"crossover":"crossovers","link_click":"clicks",
                                         "signup":"signups","improvement":"improvements"}))
    for c in ["crossovers","clicks","signups","improvements"]:
        if c not in base:
            base[c] = 0
    base["CTR %"]      = (100*base["clicks"]/base["crossovers"]).replace([np.inf,np.nan],0).round(1)
    base["Signup %"]   = (100*base["signups"]/base["clicks"]).replace([np.inf,np.nan],0).round(1)
    base["Improve %*"] = (100*base["improvements"]/base["signups"]).replace([np.inf,np.nan],0).round(1)

    st.plotly_chart(brandify(px.line(base.reset_index(), x="period", y=["CTR %","Signup %","Improve %*"]),
                             "Conversion by Period"),
                    use_container_width=True)

# ===== Health Outcomes =====
with tab_out:
    st.subheader("Health Outcomes")
    imp = fdf[fdf["event_type"]=="improvement"].copy()
    if not imp.empty:
        imp["a1c_delta"] = imp["a1c_baseline"] - imp["a1c_current"]
        imp["weight_delta_kg"] = imp["weight_baseline_kg"] - imp["weight_current_kg"]

        # Time to first improvement
        first_signup = (fdf[fdf["event_type"]=="signup"]
                        .sort_values("event_timestamp")
                        .drop_duplicates("user_id")[["user_id","event_timestamp"]]
                        .rename(columns={"event_timestamp":"signup_ts"}))
        first_impr = (imp.sort_values("event_timestamp")
                        .drop_duplicates("user_id")[["user_id","event_timestamp"]]
                        .rename(columns={"event_timestamp":"impr_ts"}))
        tt = first_signup.merge(first_impr, on="user_id", how="inner")
        tt["days_to_improve"] = (pd.to_datetime(tt["impr_ts"]) - pd.to_datetime(tt["signup_ts"])).dt.days

        k1, k2, k3 = st.columns(3)
        k1.metric("Avg A1c Reduction", f"{imp['a1c_delta'].mean():.2f}")
        k2.metric("Avg Weight Loss (kg)", f"{imp['weight_delta_kg'].mean():.1f}")
        k3.metric("Median Days to First Improvement", f"{tt['days_to_improve'].median():.0f}")

        c1, c2 = st.columns(2)
        c1.plotly_chart(brandify(px.histogram(imp, x="a1c_delta", nbins=20), "A1c Reduction"),
                        use_container_width=True)
        c2.plotly_chart(brandify(px.histogram(imp, x="weight_delta_kg", nbins=20), "Weight Loss (kg)"),
                        use_container_width=True)

        c3, c4 = st.columns(2)
        if "retention_status" in imp.columns:
            ret = imp.groupby("retention_status").size().reset_index(name="count")
            c3.plotly_chart(brandify(px.bar(ret, x="retention_status", y="count"),
                                     "Retention (Improvement Events)"),
                            use_container_width=True)
        meds = imp.groupby("medications_reduced").size().reset_index(name="count")
        c4.plotly_chart(brandify(px.bar(meds, x="medications_reduced", y="count"),
                                 "Medications Reduced"),
                        use_container_width=True)
    else:
        st.info("No improvement events in the current selection.")
