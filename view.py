
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time

# Config
SERVER = "http://localhost:8000"
REFRESH_INTERVAL = 2 

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
)


def fetch_stats() -> dict:
    try:
        r = requests.get(f"{SERVER}/stats", timeout=2)
        return r.json()
    except Exception:
        return None


def fetch_logs(limit: int = 100) -> list:
    try:
        r = requests.get(f"{SERVER}/logs?limit={limit}", timeout=2)
        return r.json().get("transactions", [])
    except Exception:
        return []


st.title("Fraud Detection - Realtime POS Monitor")
st.caption(f"Streaming Engine: `{SERVER}` - Refresh every {REFRESH_INTERVAL}s")

server_status = st.empty()
col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
kpi1_ph = col_kpi1.empty()
kpi2_ph = col_kpi2.empty()
kpi3_ph = col_kpi3.empty()
kpi4_ph = col_kpi4.empty()
kpi5_ph = col_kpi5.empty()

st.divider()
col_chart1, col_chart2 = st.columns([2, 1])

with col_chart1:
    st.subheader("Fraud Probability over Time")
    chart_proba = st.empty()

with col_chart2:
    st.subheader("ALLOW / BLOCK Ratio")
    chart_pie = st.empty()

st.divider()
st.subheader("Model Performance (Accuracy & Confusion Matrix)")
col_perf1, col_perf2 = st.columns(2)
perf_sgd_ph = col_perf1.empty()
perf_adam_ph = col_perf2.empty()

st.divider()
st.subheader("Recent Transaction Logs")
table_placeholder = st.empty()


while True:
    stats = fetch_stats()

    if stats is None:
        server_status.error("Server offline")
        time.sleep(REFRESH_INTERVAL)
        continue
    else:
        server_status.success("Server online")

    kpi1_ph.metric("Total Transactions",   stats.get("total", 0))
    kpi2_ph.metric("Fraud Detected", stats.get("fraud_detected", 0),
                    delta=None, delta_color="inverse")
    kpi3_ph.metric("Blocked",  stats.get("blocked", 0))
    kpi4_ph.metric("ROC-AUC (SGD)",
                    f"{stats.get('roc_auc_sgd', 0.0):.4f}",
                    help="Online model evaluation metric (SGD)")
    kpi5_ph.metric("ROC-AUC (Adam)",
                    f"{stats.get('roc_auc_adam', 0.0):.4f}",
                    help="Online model evaluation metric (Adam)")
                    
    perf_sgd_ph.markdown(f"""
**SGD Accuracy**: `{stats.get('acc_sgd', 0.0)*100:.4f}%`
```text
True Positives : {stats.get('cm_sgd', {}).get('tp', 0):<6} | False Positives: {stats.get('cm_sgd', {}).get('fp', 0):<6}
False Negatives: {stats.get('cm_sgd', {}).get('fn', 0):<6} | True Negatives : {stats.get('cm_sgd', {}).get('tn', 0):<6}
```
""")

    perf_adam_ph.markdown(f"""
**Adam Accuracy**: `{stats.get('acc_adam', 0.0)*100:.4f}%`
```text
True Positives : {stats.get('cm_adam', {}).get('tp', 0):<6} | False Positives: {stats.get('cm_adam', {}).get('fp', 0):<6}
False Negatives: {stats.get('cm_adam', {}).get('fn', 0):<6} | True Negatives : {stats.get('cm_adam', {}).get('tn', 0):<6}
```
""")

    logs = fetch_logs(limit=100)
    if logs:
        df = pd.DataFrame(logs)
        df["index"] = range(len(df))
        df["color"] = df["action"].map({"BLOCK": "#ef4444", "ALLOW": "#22c55e"})

        fig_line = px.line(
            df,
            x="index",
            y=["prob_sgd", "prob_adam"],
            labels={"index": "Transaction #", "value": "P(fraud)", "variable": "Model"},
            template="plotly_dark",
            height=320,
        )
        fig_line.add_hline(y=0.7, line_dash="dash", line_color="#facc15",
                           annotation_text="Block Threshold 0.7")
        fig_line.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        chart_proba.plotly_chart(fig_line, use_container_width=True, key=f"line_{stats.get('total')}")

        action_counts = df["action"].value_counts().reset_index()
        action_counts.columns = ["action", "count"]
        fig_pie = px.pie(
            action_counts,
            names="action",
            values="count",
            color="action",
            color_discrete_map={"BLOCK": "#ef4444", "ALLOW": "#22c55e"},
            template="plotly_dark",
            height=320,
            hole=0.5,
        )
        fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        chart_pie.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{stats.get('total')}")

        # Recent transaction log table
        display_df = df[["transaction_id", "prob_sgd", "prob_adam", "action", "timestamp"]].copy()
        display_df["timestamp"] = pd.to_datetime(display_df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
        display_df = display_df.iloc[::-1].reset_index(drop=True)
        table_placeholder.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "prob_sgd": st.column_config.ProgressColumn(
                    "P(fraud) SGD", min_value=0, max_value=1, format="%.4f"
                ),
                "prob_adam": st.column_config.ProgressColumn(
                    "P(fraud) Adam", min_value=0, max_value=1, format="%.4f"
                ),
                "action": st.column_config.TextColumn("Decision"),
            }
        )
    else:
        chart_proba.info("No data yet. Run `pos_producer.py` to start streaming.")
        chart_pie.empty()
        perf_sgd_ph.empty()
        perf_adam_ph.empty()
        table_placeholder.empty()

    time.sleep(REFRESH_INTERVAL)
