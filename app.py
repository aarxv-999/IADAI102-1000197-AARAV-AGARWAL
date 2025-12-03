import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import KBinsDiscretizer

# ---------- CONFIG ----------
st.set_page_config(layout="wide", page_title="Hospital Inpatient Discharges Dashboard")
DATA_PATH = "data/hospital_data.csv"

# UPDATED: Column mapping for your actual data structure
COLUMN_MAP = {
    "age": ["Age Group", "age"],
    "length_of_stay": ["Length of Stay", "length_of_stay"],
    "charges": ["Total Charges", "charges"],
    "diagnosis": ["CCSR Diagnosis Description", "Diagnosis_Code", "Diagnosis"],
    "facility": ["Facility Name", "facility"],
    "county": ["Hospital County", "county"],
    "payment": ["Payment Typology 1", "payment"],
    "severity": ["APR Severity of Illness Description", "severity"]
}

# ---------- HELPERS ----------
def load_data(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_excel(path)
    df.columns = df.columns.astype(str)
    return df

def map_columns(df):
    found = {}
    cols = list(df.columns)
    for key, options in COLUMN_MAP.items():
        for opt in options:
            if opt in cols:
                found[key] = opt
                break
        if key not in found:
            for c in cols:
                if c.lower() in [o.lower() for o in options]:
                    found[key] = c
                    break
    return found

def safe_cast_numeric(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def preprocess(df):
    mapped = map_columns(df)
    
    # Add any missing expected columns as NaN
    for k in COLUMN_MAP.keys():
        if k not in mapped:
            mapped[k] = None
    
    # Standardize column names
    std = {}
    for k, v in mapped.items():
        if v:
            std[v] = k
    df = df.rename(columns=std)
    
    # Cast numeric columns
    df = safe_cast_numeric(df, "length_of_stay")
    df = safe_cast_numeric(df, "charges")
    
    # Clean categorical columns
    for c in ["diagnosis", "facility", "county", "payment", "severity"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("Unknown")
    
    # Remove rows with no length_of_stay
    if "length_of_stay" in df.columns:
        df = df.dropna(subset=["length_of_stay"])
    
    return df

# ---------- METRICS ----------
def compute_metrics(df):
    metrics = {}
    if "length_of_stay" in df.columns:
        metrics["avg_los"] = df["length_of_stay"].mean()
        thresh = df["length_of_stay"].mean() + df["length_of_stay"].std()
        metrics["pct_long_stay"] = (df["length_of_stay"] > thresh).mean() * 100
    else:
        metrics["avg_los"] = None
        metrics["pct_long_stay"] = None
    
    if "charges" in df.columns:
        metrics["avg_charges"] = df["charges"].mean()
    else:
        metrics["avg_charges"] = None
    
    return metrics

# ---------- VISUALIZATIONS ----------
def vis_avg_stay_by_diagnosis(df, top_n=15):
    if "diagnosis" not in df.columns or "length_of_stay" not in df.columns:
        return None
    agg = df.groupby("diagnosis")["length_of_stay"].mean().reset_index().sort_values("length_of_stay", ascending=False).head(top_n)
    fig = px.bar(agg, x="length_of_stay", y="diagnosis", orientation='h', labels={"length_of_stay":"Avg Length of Stay (days)","diagnosis":"Diagnosis"})
    return fig

def vis_box_charges_by_severity(df):
    if "charges" not in df.columns or "severity" not in df.columns:
        return None
    fig = px.box(df, x="severity", y="charges", labels={"charges":"Total Charges","severity":"Severity"})
    return fig

def vis_heatmap_facility_county(df):
    if "facility" not in df.columns or "county" not in df.columns or "length_of_stay" not in df.columns:
        return None
    pivot = df.groupby(["facility","county"])["length_of_stay"].mean().reset_index()
    pivot_table = pivot.pivot(index="facility", columns="county", values="length_of_stay").fillna(0)
    fig = px.imshow(pivot_table, labels=dict(x="County", y="Facility", color="Avg LOS (days)"), aspect="auto")
    return fig

def vis_payment_pie(df):
    if "payment" not in df.columns:
        return None
    counts = df["payment"].value_counts().reset_index()
    counts.columns = ["payment","count"]
    fig = px.pie(counts, values="count", names="payment", title="Patient Distribution by Payment Type")
    return fig

def vis_los_histogram(df):
    if "length_of_stay" not in df.columns:
        return None
    fig = px.histogram(df, x="length_of_stay", nbins=30, labels={"length_of_stay":"Length of Stay (days)"})
    return fig

# ---------- STREAMLIT LAYOUT ----------
def main():
    st.title("Hospital Inpatient Discharges — Interactive Dashboard")
    st.markdown("**Purpose:** Explore length of stay, charges, and patterns across diagnosis, facilities, and payment types.")
    
    with st.spinner("Loading data..."):
        df_raw = load_data(DATA_PATH)
    
    st.sidebar.header("Filters & Settings")
    st.sidebar.markdown("Data source: `" + DATA_PATH + "`")
    
    df = preprocess(df_raw.copy())
    
    if st.sidebar.checkbox("Show raw sample / column mapping", value=False):
        st.subheader("Raw data sample")
        st.dataframe(df_raw.head(10))
        st.write("Detected standardized columns:", list(df.columns))
        st.write("Column mapping details:", map_columns(df_raw))
    
    filters = {}
    if "facility" in df.columns:
        facs = sorted(df["facility"].unique().tolist())
        selection = st.sidebar.multiselect("Facility", options=facs, default=facs[:5])
        filters["facility"] = selection
    
    if "county" in df.columns:
        cnts = sorted(df["county"].unique().tolist())
        selection = st.sidebar.multiselect("County", options=cnts, default=cnts)
        filters["county"] = selection
    
    if "diagnosis" in df.columns:
        diags = sorted(df["diagnosis"].unique().tolist())
        selection = st.sidebar.multiselect("Diagnosis", options=diags[:100], default=diags[:10])
        filters["diagnosis"] = selection
    
    if "severity" in df.columns:
        sevs = sorted(df["severity"].unique().tolist())
        selection = st.sidebar.multiselect("Severity", options=sevs, default=sevs)
        filters["severity"] = selection
    
    for k, v in filters.items():
        if v:
            df = df[df[k].isin(v)]
    
    metrics = compute_metrics(df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average LOS (days)", f"{metrics['avg_los']:.2f}" if metrics['avg_los'] is not None else "N/A")
    col2.metric("Pct Long Stay (%)", f"{metrics['pct_long_stay']:.2f}%" if metrics['pct_long_stay'] is not None else "N/A")
    col3.metric("Average Charges", f"${metrics['avg_charges']:.0f}" if metrics.get('avg_charges') else "N/A")
    col4.metric("Total Records", f"{len(df):,}")
    
    st.markdown("---")
    left, right = st.columns((2,1))
    with left:
        st.subheader("Avg Length of Stay by Diagnosis")
        fig1 = vis_avg_stay_by_diagnosis(df)
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Diagnosis / LOS columns not found.")
        
        st.subheader("Length of Stay Distribution")
        fig_hist = vis_los_histogram(df)
        if fig_hist:
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with right:
        st.subheader("Charges by Severity (Boxplot)")
        fig2 = vis_box_charges_by_severity(df)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Charges or Severity column missing.")
        
        st.subheader("Payment Type Distribution")
        fig3 = vis_payment_pie(df)
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Payment column missing.")
    
    st.markdown("---")
    st.subheader("Facility × County — Avg LOS Heatmap")
    fig4 = vis_heatmap_facility_county(df)
    if fig4:
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Facility / County / LOS columns missing for heatmap.")
    
    st.markdown("---")
    st.subheader("Data Table (filtered)")
    st.dataframe(df.head(100))
    
    st.markdown("### Notes")
    st.markdown("- Long stay threshold = mean + 1 * std")
    st.markdown("- Column mapping automatically detects your data structure.")

if __name__ == "__main__":
    main()
