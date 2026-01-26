%%writefile app.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="SmokeSignal AI", layout="wide")

@st.cache_data
def load_data():
    # Load your results (adjust path if needed)
    df = pd.read_csv("final_predictions.csv.zip")
    # Create a friendly "Location" label combining Lat/Lon
    df['Location_Label'] = df['Lat'].astype(str) + ", " + df['Lon'].astype(str)
    df['Date'] = pd.to_datetime(df.get('Date', '2023-01-01')) # Ensure Date exists
    return df

try:
    df = load_data()
    st.success("Data successfully loaded!")
except:
    st.error("Could not find 'final_predictions.csv'. Please run the previous notebook steps first!")
    st.stop()

# --- 2. THE SIDEBAR (The Control Center) ---
st.sidebar.title("🔍 Hazard Controls")
st.sidebar.markdown("Filter to see specific fire events.")

# Select Location
unique_locations = df['Location_Label'].unique()
selected_loc = st.sidebar.selectbox("Select Monitor Location:", unique_locations)

# Filter Data for that Location
loc_data = df[df['Location_Label'] == selected_loc]

# Select Date
min_date = loc_data['Date'].min()
max_date = loc_data['Date'].max()
selected_date = st.sidebar.date_input("Select Date:", min_value=min_date, max_value=max_date, value=min_date)

# Get specific row for that day
specific_day = loc_data[loc_data['Date'].dt.date == selected_date]

# --- 3. MAIN DASHBOARD ---
st.title("🌲 SmokeSignal: AI Wildfire Forecaster")

# METRICS ROW
col1, col2, col3 = st.columns(3)
if not specific_day.empty:
    pred = specific_day.iloc[0]['Predicted_PM25']
    actual = specific_day.iloc[0]['Actual_PM25']
    error = specific_day.iloc[0]['Error']
    
    col1.metric("Predicted Air Quality (PM2.5)", f"{pred:.1f}")
    col2.metric("Actual Air Quality", f"{actual:.1f}", delta=f"{error:.1f} error", delta_color="inverse")
    
    # Hazard Alert Logic
    status = "🚨 HAZARDOUS" if pred > 35 else "✅ SAFE"
    status_color = "red" if pred > 35 else "green"
    col3.markdown(f"### Status: :{status_color}[{status}]")
else:
    st.warning("No data found for this specific date.")

# VISUAL 1: PERFORMANCE CHART
st.subheader("📉 Actual vs. Predicted (Time Series)")
st.line_chart(loc_data.set_index("Date")[['Actual_PM25', 'Predicted_PM25']])

# --- 4. EXPLAINABILITY & MAP (The "Why") ---
st.divider()
col_map, col_explain = st.columns([1, 1])

with col_map:
    st.subheader("📍 Monitor Location")
    # Streamlit needs 'lat' and 'lon' columns exactly named
    map_data = loc_data[['Lat', 'Lon']].rename(columns={'Lat': 'lat', 'Lon': 'lon'})
    st.map(map_data.iloc[[0]]) # Show just the one point

with col_explain:
    st.subheader("🤖 Why is the air bad?")
    
    if not specific_day.empty:
        # NOTE: In a real app, you would load the full 'explainer' object.
        # For this demo, we simulate the logic based on your 'Smoke_Yesterday' feature.
        
        smoke_val = specific_day.iloc[0].get('Smoke_Yesterday', 0)
        velocity_val = specific_day.iloc[0].get('Velocity_Yesterday', 0)
        
        st.write(f"**Key Driver Analysis:**")
        st.info(f"💨 **Smoke Intensity (Yesterday):** {smoke_val} (Scale 0-3)")
        st.info(f"📈 **Pollution Velocity:** {velocity_val:.1f} (Rate of change)")
        
        if smoke_val > 0:
            st.warning(" The AI detected significant smoke plumes yesterday, driving the risk up.")
        elif velocity_val > 10:
            st.warning(" Pollution is rising rapidly (High Velocity), suggesting a new ignition.")
        else:
            st.success(" Factors suggest stable atmospheric conditions.")
