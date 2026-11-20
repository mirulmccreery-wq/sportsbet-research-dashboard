import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(
    page_title="SportsBet Research Dashboard",
    page_icon="⚽",
    layout="wide"
)

def load_from_bytes(data: bytes) -> pd.DataFrame:
    records = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def settled_only(df):
    if df.empty:
        return df
    mask = df["event_id"].astype(str).str.contains("_SETTLED", na=False)
    if "hit_1x2" in df.columns:
        mask = mask | df["hit_1x2"].notna()
    return df[mask].copy()

def upcoming_only(df):
    if df.empty:
        return df
    settled_ids = set(settled_only(df)["event_id"].astype(str))
    return df[~df["event_id"].astype(str).isin(settled_ids)].copy()

st.title("⚽ SportsBet Research Dashboard")
st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

uploaded = st.file_uploader("Upload SportsBet_Research_Log.jsonl", type=["jsonl", "json", "txt"])

if uploaded is None:
    st.info("Upload your research log JSONL file to view the dashboard.")
    st.stop()

df = load_from_bytes(uploaded.getvalue())
if df.empty:
    st.warning("No valid data found in the uploaded file.")
    st.stop()

settled = settled_only(df)
upcoming = upcoming_only(df)

# ==================== FILTERS ====================
st.sidebar.header("Filters")
competitions = ["All"] + sorted(df["competition"].dropna().unique().tolist()) if "competition" in df.columns else ["All"]
selected_comp = st.sidebar.selectbox("Competition", competitions)

date_min = df["date"].min() if "date" in df.columns and not df["date"].isna().all() else None
date_max = df["date"].max() if "date" in df.columns and not df["date"].isna().all() else None
if date_min and date_max:
    date_range = st.sidebar.date_input(
        "Date range",
        value=(date_min.date(), date_max.date()),
        min_value=date_min.date(),
        max_value=date_max.date()
    )
else:
    date_range = None

# Apply filters
filtered = settled.copy()
if selected_comp != "All" and "competition" in filtered.columns:
    filtered = filtered[filtered["competition"] == selected_comp]
if date_range and len(date_range) == 2 and "date" in filtered.columns:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]

# ==================== KPIs ====================
col1, col2, col3, col4, col5 = st.columns(5)
total_settled = len(filtered)
hit_1x2 = filtered["hit_1x2"].sum() if "hit_1x2" in filtered.columns and total_settled else 0
hit_ou = filtered["hit_over_2_5"].sum() if "hit_over_2_5" in filtered.columns and total_settled else 0
hit_btts = filtered["hit_btts"].sum() if "hit_btts" in filtered.columns and total_settled else 0

with col1:
    st.metric("Settled matches", total_settled)
with col2:
    rate = (hit_1x2 / total_settled * 100) if total_settled else 0
    st.metric("1X2 hit rate", f"{rate:.1f}%")
with col3:
    rate_ou = (hit_ou / total_settled * 100) if total_settled else 0
    st.metric("Over 2.5 hit rate", f"{rate_ou:.1f}%")
with col4:
    rate_btts = (hit_btts / total_settled * 100) if total_settled else 0
    st.metric("BTTS hit rate", f"{rate_btts:.1f}%")
with col5:
    st.metric("Total log entries", len(df))

st.divider()

# ==================== CHARTS ====================
tab1, tab2, tab3 = st.tabs(["Rolling Hit Rate", "Calibration", "Recent Matches"])

with tab1:
    st.subheader("Rolling 1X2 Hit Rate (last 20 matches)")
    if not filtered.empty and "hit_1x2" in filtered.columns and "date" in filtered.columns:
        roll = filtered.sort_values("date").copy()
        roll["rolling_hit"] = roll["hit_1x2"].rolling(window=20, min_periods=5).mean() * 100
        fig = px.line(
            roll.dropna(subset=["rolling_hit"]),
            x="date",
            y="rolling_hit",
            title="Rolling 20-match 1X2 Hit Rate",
            labels={"rolling_hit": "Hit Rate %", "date": "Date"}
        )
        fig.add_hline(y=33.3, line_dash="dash", line_color="gray", annotation_text="Random (~33%)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for rolling chart.")

with tab2:
    st.subheader("Calibration (Model Probability vs Actual)")
    if not filtered.empty and "model_probs" in filtered.columns and "hit_1x2" in filtered.columns:
        def get_home_prob(probs):
            if isinstance(probs, dict) and probs:
                return list(probs.values())[0]
            return None
        cal = filtered.copy()
        cal["home_prob"] = cal["model_probs"].apply(get_home_prob)
        cal = cal.dropna(subset=["home_prob", "hit_1x2"])
        if len(cal) >= 10:
            cal["prob_bin"] = pd.cut(cal["home_prob"], bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0], labels=["<40%", "40-50%", "50-60%", "60-70%", ">70%"])
            calib = cal.groupby("prob_bin", observed=True).agg(
                predicted=("home_prob", "mean"),
                actual=("hit_1x2", "mean"),
                n=("hit_1x2", "count")
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=calib["predicted"], y=calib["actual"], mode="markers+text",
                                     text=calib["n"].astype(str), textposition="top center",
                                     name="Observed", marker=dict(size=12)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash", color="gray")))
            fig.update_layout(title="Calibration: Predicted vs Actual (Home Win)",
                              xaxis_title="Avg Predicted Probability",
                              yaxis_title="Actual Hit Rate",
                              height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need more settled matches for calibration.")
    else:
        st.info("Model probabilities not available for calibration.")

with tab3:
    st.subheader("Recent Settled Matches")
    if not filtered.empty:
        display_cols = [c for c in [
            "date", "home", "away", "competition",
            "ft_home", "ft_away", "hit_1x2", "hit_over_2_5", "hit_btts"
        ] if c in filtered.columns]
        recent = filtered.sort_values("date", ascending=False).head(20)
        st.dataframe(recent[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No matches match the current filters.")

# ==================== UPCOMING ====================
st.divider()
st.subheader("Upcoming / Unsettled Matches")
if not upcoming.empty:
    up_cols = [c for c in ["date", "home", "away", "competition", "model_probs", "projected_goals"] if c in upcoming.columns]
    up_show = upcoming.sort_values("date", ascending=False).head(15)
    st.dataframe(up_show[up_cols], use_container_width=True, hide_index=True)
else:
    st.info("No upcoming matches in the log.")

# ==================== BY COMPETITION ====================
if not filtered.empty and "competition" in filtered.columns:
    st.subheader("Hit Rate by Competition")
    by_comp = (
        filtered.groupby("competition")
        .agg(
            matches=("event_id", "count"),
            hit_1x2=("hit_1x2", "sum"),
        )
        .reset_index()
    )
    by_comp["1X2 %"] = (by_comp["hit_1x2"] / by_comp["matches"] * 100).round(1)
    st.dataframe(by_comp.sort_values("matches", ascending=False), use_container_width=True, hide_index=True)
