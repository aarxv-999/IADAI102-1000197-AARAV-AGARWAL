# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import KBinsDiscretizer

# ---------- CONFIG ----------
st.set_page_config(layout="wide", page_title="Hospital Inpatient Discharges Dashboard")

DATA_PATH = "data/hospital_data.csv"  # <-- update this if your file is elsewhere

# Common expected column names (change mapping if your CSV uses different names)
COLUMN_MAP = {
    "age": ["Age", "Patient_Age", "age"],
    "length_of_stay": ["Length_of_stay", "LOS", "length_of_stay", "LengthOfStay"],
    "charges": ["Total_Charges", "Charges", "TotalCharges", "Charge"],
    "diagnosis": ["Diagnosis_Code", "Diagnosis", "Diag", "DiagnosisCode"],
    "facility": ["Facility", "Hospital", "Hospital_Name"],
    "county": ["County", "Region", "State"],
    "payment": ["Payment_Type", "Payment", "Payer"],
    "severity": ["Severity", "Severity_Level", "DRG_Severity"]
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
    # find best matches for required columns
    found = {}
    cols = list(df.columns)
    for key, options in COLUMN_MAP.items():
        for opt in options:
            if opt in cols:
                found[key] = opt
                break
        # fallback: case-insensitive search
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
    # map columns
    mapped = map_columns(df)
    # add any missing expected columns as NaN
    for k in COLUMN_MAP.keys():
        if k not in mapped:
            mapped[k] = None
    # standardize column names in df for easy coding
    std = {}
    for k, v in mapped.items():
        if v:
            std[v] = k
    df = df.rename(columns=std)
    # cast numeric columns
    df = safe_cast_numeric(df, "length_of_stay")
    df = safe_cast_numeric(df, "charges")
    # create age group if age exists
    if "age" in df.columns:
        df["age_group"] = pd.cut(df["age"], bins=[0,17,35,50,65,200],
                                 labels=["0-17","18-35","36-50","51-65","65+"], include_lowest=True)
    # clean payment, diagnosis, facility, severity columns
    for c in ["diagnosis", "facility", "county", "payment", "severity"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("Unknown")
    # remove rows with no length_of_stay
    if "length_of_stay" in df.columns:
        df = df.dropna(subset=["length_of_stay"])
    return df

# ---------- METRICS ----------
def compute_metrics(df):
    metrics = {}
    if "length_of_stay" in df.columns:
        metrics["avg_los"] = df["length_of_stay"].mean()
        # define long stay as > mean + 1 std (teacher suggested)
        thresh = df["length_of_stay"].mean() + df["length_of_stay"].std()
        metrics["pct_long_stay"] = (df["length_of_stay"] > thresh).mean() * 100
    else:
        metrics["avg_los"] = None
        metrics["pct_long_stay"] = None
    if "charges" in df.columns:
        metrics["avg_charges"] = df["charges"].mean()
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
    st.markdown("**Purpose:** Explore length of stay, charges, and patterns across diagnosis, facilities, and payment types. (Summative Assessment)")

    # Load data
    with st.spinner("Loading data..."):
        df_raw = load_data(DATA_PATH)
    st.sidebar.header("Filters & Settings")
    st.sidebar.markdown("Data source: `" + DATA_PATH + "`")
    # Preprocess & map
    df = preprocess(df_raw.copy())

    # Show sample and column mapping
    if st.sidebar.checkbox("Show raw sample / column mapping", value=False):
        st.subheader("Raw data sample")
        st.dataframe(df_raw.head(10))
        st.write("Detected standardized columns (available):", list(df.columns))

    # Sidebar filters (only show if columns exist)
    filters = {}
    if "facility" in df.columns:
        facs = df["facility"].unique().tolist()
        selection = st.sidebar.multiselect("Facility", options=sorted(facs), default=facs)
        filters["facility"] = selection
    if "county" in df.columns:
        cnts = df["county"].unique().tolist()
        selection = st.sidebar.multiselect("County", options=sorted(cnts), default=cnts)
        filters["county"] = selection
    if "diagnosis" in df.columns:
        diags = df["diagnosis"].unique().tolist()
        selection = st.sidebar.multiselect("Diagnosis", options=sorted(diags)[:200], default=sorted(diags)[:20])
        filters["diagnosis"] = selection
    if "severity" in df.columns:
        sevs = df["severity"].unique().tolist()
        selection = st.sidebar.multiselect("Severity", options=sorted(sevs), default=sorted(sevs))
        filters["severity"] = selection
    if "age_group" in df.columns:
        ags = df["age_group"].unique().tolist()
        selection = st.sidebar.multiselect("Age Group", options=sorted(ags.astype(str)), default=sorted(ags.astype(str)))
        filters["age_group"] = selection

    # Apply filters
    for k, v in filters.items():
        if v:
            df = df[df[k].isin(v)]

    # Compute metrics
    metrics = compute_metrics(df)
    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average LOS (days)", f"{metrics['avg_los']:.2f}" if metrics['avg_los'] is not None else "N/A")
    col2.metric("Pct Long Stay (%)", f"{metrics['pct_long_stay']:.2f}%" if metrics['pct_long_stay'] is not None else "N/A")
    col3.metric("Average Charges", f"₹{metrics['avg_charges']:.2f}" if metrics.get('avg_charges') else "N/A")
    col4.metric("Total Records", f"{len(df):,}")

    # Main visuals in two columns
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
    st.markdown("- Long stay threshold = mean + 1 * std (configurable in code).")
    st.markdown("- If your dataset uses different column names, update `COLUMN_MAP` at the top of app.py.")

if __name__ == "__main__":
    main()
