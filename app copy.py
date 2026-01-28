import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import altair as alt

# --- 1. CONFIG & DATA LOADING ---
st.set_page_config(page_title="SmokeSignal AI", layout="wide")

@st.cache_data
def load_data():
    try:
        with zipfile.ZipFile("data.zip", "r") as z:
            all_files = z.namelist()
            # Filter out Mac hidden files
            csv_files = [f for f in all_files if f.endswith('.csv') and not f.startswith('__MACOSX')]
            
            if not csv_files:
                st.error("Error: No CSV file found inside data.zip")
                st.stop()
            
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f)
                
    except FileNotFoundError:
        st.error("CRITICAL ERROR: Could not find 'data.zip'. Check GitHub file list.")
        st.stop()

    # --- PREPROCESSING ---
    # Create Location Label
    if 'Lat' in df.columns and 'Lon' in df.columns:
        df['Location_Label'] = df['Lat'].astype(str) + ", " + df['Lon'].astype(str)
    else:
        st.error("Missing 'Lat' or 'Lon' columns in CSV.")
        st.stop()
    
    # Process Dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate Error if columns exist (Soft check)
    if 'Actual_PM25' in df.columns and 'Predicted_PM25' in df.columns:
        df['Prediction_Error'] = df['Actual_PM25'] - df['Predicted_PM25']
        
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Data Load Error: {e}")
    st.stop()

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.title("🌲 SmokeSignal AI")
st.sidebar.header("Configuration")

# Debugging Helper: Show columns if needed
with st.sidebar.expander("⚠️ Debug: View Column Names"):
    st.write(df.columns.tolist())

# Location Selector
unique_locations = df['Location_Label'].unique()
selected_loc = st.sidebar.selectbox("📍 Select Monitor Location:", unique_locations)
loc_data = df[df['Location_Label'] == selected_loc]

# Date Selector
if not loc_data.empty:
    min_date = loc_data['Date'].min()
    max_date = loc_data['Date'].max()
    selected_date = st.sidebar.date_input("📅 Select Date:", min_value=min_date, max_value=max_date, value=min_date)
    specific_day = loc_data[loc_data['Date'].dt.date == selected_date]
else:
    st.warning("No data for this location.")
    st.stop()

# --- 3. MAIN DASHBOARD ---
st.title("🌲 West Coast SmokeSignal: AI Forecasting")

# Check for required columns before creating metrics
required_cols = ['Actual_PM25', 'Predicted_PM25']
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    st.error(f"⚠️ **Column Name Mismatch!** The code is looking for columns named: `{required_cols}`")
    st.error(f"❌ **Missing:** {missing_cols}")
    st.info(f"✅ **Available Columns in your CSV:** {df.columns.tolist()}")
    st.warning("Please rename the columns in your CSV or update the code to match the names above.")
    st.stop() # Stop execution here to prevent the crash

# A. TOP METRICS
col1, col2, col3, col4 = st.columns(4)
if not specific_day.empty:
    row = specific_day.iloc[0]
    
    pred = row.get('Predicted_PM25', 0)
    actual = row.get('Actual_PM25', 0)
    error = row.get('Prediction_Error', 0)
    
    if pred < 12: status_color = "green"
    elif pred < 35: status_color = "orange"
    else: status_color = "red"

    col1.metric("🤖 AI Forecast (PM2.5)", f"{pred:.1f}")
    col2.metric("📉 Actual Value", f"{actual:.1f}")
    col3.metric("⚠️ Model Error", f"{error:.1f}", delta_color="inverse")
    col4.markdown(f"**Risk Level:** :{status_color}[**{'HAZARDOUS' if pred > 35 else 'SAFE'}**]")

# --- 4. ACCURACY SECTION ---
st.divider()
st.subheader("🔍 Model Diagnostics")

tab1, tab2 = st.tabs(["📉 Time Series", "🎯 Error Distribution"])

with tab1:
    # SAFE MELT: We already checked columns exist above, so this is now safe
    chart_data = loc_data.melt(id_vars=['Date'], value_vars=['Actual_PM25', 'Predicted_PM25'], var_name='Type', value_name='PM25')
    
    line_chart = alt.Chart(chart_data).mark_line().encode(
        x='Date:T',
        y='PM25:Q',
        color=alt.Color('Type', scale=alt.Scale(domain=['Actual_PM25', 'Predicted_PM25'], range=['#1f77b4', '#d62728'])),
        tooltip=['Date', 'Type', 'PM25']
    ).properties(height=350, title="Actual vs Predicted")
    
    st.altair_chart(line_chart, use_container_width=True)

with tab2:
    if 'Prediction_Error' in loc_data.columns:
        residual_chart = alt.Chart(loc_data).mark_bar().encode(
            x='Date:T',
            y='Prediction_Error:Q',
            color=alt.condition(
                alt.datum.Prediction_Error > 0,
                alt.value("orange"),
                alt.value("blue")
            ),
            tooltip=['Date', 'Prediction_Error']
        ).properties(height=300)
        st.altair_chart(residual_chart, use_container_width=True)

# --- 5. EXPLAINABILITY ---
st.divider()
st.subheader("🧠 XAI: Explainability Engine")

col_xai_1, col_xai_2 = st.columns([1, 2])

with col_xai_1:
    if not specific_day.empty:
        # Robust .get() calls so it won't crash if these explainability columns are missing
        smoke_val = row.get('Smoke_Yesterday', 0)
        velocity_val = row.get('Velocity_Yesterday', 0)
        
        st.markdown("**🔥 Smoke Intensity (Yesterday)**")
        st.progress(min(smoke_val / 4.0, 1.0))
        st.caption(f"Value: {smoke_val}")

with col_xai_2:
    # Feature Importance Chart
    features = {
        "Smoke Density (Lag 1)": row.get('Smoke_Yesterday', 0) * 10, 
        "Pollution Velocity": row.get('Velocity_Yesterday', 0),
        "Local Trends": row.get('Actual_PM25', 0) * 0.5 
    }
    feat_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Impact_Score'])
    
    bar_chart = alt.Chart(feat_df).mark_bar().encode(
        x='Impact_Score:Q',
        y=alt.Y('Feature:N', sort='-x'),
        color=alt.Color('Impact_Score', scale=alt.Scale(scheme='reds'))
    )
    st.altair_chart(bar_chart, use_container_width=True)
