import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import zipfile  # <--- Added to handle Mac zip files

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="SmokeSignal AI", layout="wide")

@st.cache_data
def load_data():
    # We use zipfile to ignore the hidden __MACOSX folder that Mac creates
    try:
        with zipfile.ZipFile("data.zip", "r") as z:
            # Get list of all files inside the zip
            all_files = z.namelist()
            # Filter: Find the file that ends with .csv and is NOT the Mac hidden folder
            csv_files = [f for f in all_files if f.endswith('.csv') and not f.startswith('__MACOSX')]
            
            if not csv_files:
                st.error("Error: No CSV file found inside data.zip")
                st.stop()
            
            # Pick the first valid CSV file found (e.g., 'final_predictions_full copy.csv')
            target_file = csv_files[0]
            
            # Open that specific file
            with z.open(target_file) as f:
                df = pd.read_csv(f)
    except FileNotFoundError:
        st.error("CRITICAL ERROR: Could not find 'data.zip'. Please check GitHub file list.")
        st.stop()
    
    # Create a friendly "Location" label combining Lat/Lon
    # Force Lat/Lon to string to avoid errors
    df['Location_Label'] = df['Lat'].astype(str) + ", " + df['Lon'].astype(str)
    
    # Ensure Date column is standard datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        st.error("Column 'Date' not found in the CSV.")
        
    return df

# Main Data Loading Block
try:
    df = load_data()
    st.sidebar.success("Data System: Online ✅")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
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
if not loc_data.empty:
    min_date = loc_data['Date'].min()
    max_date = loc_data['Date'].max()
    selected_date = st.sidebar.date_input("Select Date:", min_value=min_date, max_value=max_date, value=min_date)
    
    # Get specific row for that day
    specific_day = loc_data[loc_data['Date'].dt.date == selected_date]
else:
    st.warning("No data available for this location.")
    st.stop()

# --- 3. MAIN DASHBOARD ---
st.title("🌲 SmokeSignal: AI Wildfire Forecaster")

# METRICS ROW
col1, col2, col3 = st.columns(3)
if not specific_day.empty:
    # Use .get() to avoid errors if columns are missing
    pred = specific_day.iloc[0].get('Predicted_PM25', 0)
    actual = specific_day.iloc[0].get('Actual_PM25', 0)
    error = specific_day.iloc[0].get('Error', 0)
    
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
if not loc_data.empty:
    chart_data = loc_data.set_index("Date")[['Actual_PM25', 'Predicted_PM25']]
    st.line_chart(chart_data)

# --- 4. EXPLAINABILITY & MAP (The "Why") ---
st.divider()
col_map, col_explain = st.columns([1, 1])

with col_map:
    st.subheader("📍 Monitor Location")
    # Streamlit needs 'lat' and 'lon' columns exactly named (case sensitive)
    # We create a copy to avoid SettingWithCopy warnings
    map_data = loc_data[['Lat', 'Lon']].copy()
    map_data = map_data.rename(columns={'Lat': 'lat', 'Lon': 'lon'})
    st.map(map_data.iloc[[0]]) # Show just the one point

with col_explain:
    st.subheader("🤖 Why is the air bad?")
    
    if not specific_day.empty:
        # Simulate explainability based on features
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
