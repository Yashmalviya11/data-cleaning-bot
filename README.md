# Automated-data-cleaning-bot
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# ---- Data Cleaning Functions ----
def fix_dtypes(df):
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
    return df

def handle_missing(df):
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna("Unknown", inplace=True)
    return df

def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

def detect_outliers(df, contamination=0.05):
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(df[num_cols])
        df = df[preds != -1]
    return df

def clean_text(df):
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].str.strip().str.title()
    return df

def data_summary(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

# ---- Streamlit App ----
st.set_page_config(page_title="AutoClean Bot", layout="wide")
st.title("🤖 Automated Data Cleaning Bot")
st.write("Upload your CSV/Excel file and get a cleaned dataset instantly!")

uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📊 Original Data Preview")
    st.dataframe(df.head())

    st.subheader("📋 Original Data Summary")
    st.dataframe(data_summary(df))

    # Cleaning Steps
    df = fix_dtypes(df)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = detect_outliers(df)
    df = clean_text(df)

    st.subheader("✅ Cleaned Data Preview")
    st.dataframe(df.head())

    st.subheader("📋 Cleaned Data Summary")
    st.dataframe(data_summary(df))

    # Download cleaned data
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")
