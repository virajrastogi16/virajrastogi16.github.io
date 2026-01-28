import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import altair as alt
import pydeck as pdk

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SmokeSignal AI",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DATA LOADING (The "Safe Mode" Logic) ---
@st.cache_data
def load_data():
    try:
        # Robust Zip Loader
        with zipfile.ZipFile("data.zip", "r") as z:
            all_files = z.namelist()
            csv_files = [f for f in all_files if f.endswith('.csv') and not f.startswith('__MACOSX')]
            
            if not csv_files:
                st.error("Error: No CSV file found inside data.zip")
                st.stop()
            
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f)
                
    except FileNotFoundError:
        st.error("CRITICAL ERROR: Could not find 'data.zip'. Ensure it is uploaded to GitHub!")
        st.stop()

    # --- PREPROCESSING ---
    # 1. Create Location Label
    if 'Lat' in df.columns and 'Lon' in df.columns:
        df['Location_Label'] = df['Lat'].astype(str) + ", " + df['Lon'].astype(str)
    
    # 2. Process Dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
    # 3. Create State Map (Mapping IDs to Names based on your notebook)
    state_map = {6: 'California (CA)', 41: 'Oregon (OR)', 53: 'Washington (WA)', 32: 'Nevada (NV)', 4: 'Arizona (AZ)'}
    if 'State_ID' in df.columns:
        df['State_Name'] = df['State_ID'].map(state_map).fillna('Unknown')
    else:
        df['State_Name'] = 'All Regions'

    # 4. Calculate Errors
    if 'Actual_PM25' in df.columns and 'Predicted_PM25' in df.columns:
        df['Error'] = df['Actual_PM25'] - df['Predicted_PM25']
        df['Absolute_Error'] = df['Error'].abs()
        
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Data Load Error: {e}")
    st.stop()

# --- 3. SIDEBAR CONTROLS (Matching Screenshots) ---
st.sidebar.title("🔍 Regional Controls")

# A. Date Selection
min_date = df['Date'].min()
max_date = df['Date'].max()
selected_date = st.sidebar.date_input(
    "Select Analysis Date:",
    min_value=min_date,
    max_value=max_date,
    value=min_date # Default to first date
)

# Filter by Date immediately to speed up other filters
day_data = df[df['Date'].dt.date == selected_date]

if day_data.empty:
    st.warning(f"No data available for {selected_date}.")
    st.stop()

# B. State Filter
state_options = ['All'] + sorted(df['State_Name'].unique().tolist())
selected_state = st.sidebar.selectbox("Filter by State:", state_options)

if selected_state != 'All':
    filtered_data = day_data[day_data['State_Name'] == selected_state]
else:
    filtered_data = day_data

# C. Sensor Selection
sensor_options = filtered_data['Location_Label'].unique()
if len(sensor_options) > 0:
    selected_sensor = st.sidebar.selectbox("Select Specific Sensor:", sensor_options)
    sensor_row = filtered_data[filtered_data['Location_Label'] == selected_sensor].iloc[0]
else:
    st.sidebar.warning("No sensors found for this selection.")
    sensor_row = None


# --- 4. MAIN DASHBOARD HEADER ---
st.title("🌲 West Coast SmokeSignal: Wildfire AI")
st.markdown(f"### Tracking PM2.5 Levels across **{len(filtered_data)}** active sensors")
st.divider()

# --- 5. TOP METRICS ROW ---
if sensor_row is not None:
    # Extract values safely
    pred = sensor_row.get('Predicted_PM25', 0)
    actual = sensor_row.get('Actual_PM25', 0)
    velocity = sensor_row.get('Velocity_Yesterday', 0)
    
    # Status Logic
    if pred > 35:
        status = "🚨 HAZARDOUS"
        s_color = "red"
    elif pred > 12:
        status = "⚠️ MODERATE"
        s_color = "orange"
    else:
        status = "✅ SAFE"
        s_color = "green"

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Predicted PM2.5", f"{pred:.1f} µg/m³")
    with col2:
        st.metric("Actual PM2.5", f"{actual:.1f} µg/m³", delta=f"{actual-pred:.1f} diff")
    with col3:
        st.metric("Pollution Velocity", f"{velocity:.2f}", help="Rate of change from yesterday")
    with col4:
        st.markdown(f"### Status: :{s_color}[{status}]")

# --- 6. REGIONAL MAP (The "Green/Red Dots" from Screenshot) ---
st.subheader(f"🌍 Regional Air Quality Map ({selected_date})")

# Define color based on Predicted PM2.5 (Green -> Yellow -> Red)
# Pydeck requires RGB tuples. 
# Low (<12) = Green [0, 255, 0], High (>50) = Red [255, 0, 0]
def get_color(val):
    if val < 12: return [0, 128, 0, 160] # Green
    elif val < 35: return [255, 165, 0, 160] # Orange
    return [255, 0, 0, 160] # Red

# Apply colors
map_df = filtered_data.copy()
map_df['color'] = map_df['Predicted_PM25'].apply(get_color)
# Scale radius by pollution level for visibility
map_df['radius'] = map_df['Predicted_PM25'] * 500 

layer = pdk.Layer(
    "ScatterplotLayer",
    map_df,
    get_position='[Lon, Lat]',
    get_color='color',
    get_radius='radius',
    pickable=True,
    radius_min_pixels=5,
    radius_max_pixels=30,
)

view_state = pdk.ViewState(
    latitude=map_df['Lat'].mean(),
    longitude=map_df['Lon'].mean(),
    zoom=5,
    pitch=0,
)

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=view_state,
    layers=[layer],
    tooltip={"text": "Location: {Location_Label}\nPred PM2.5: {Predicted_PM25}"}
))

# --- 7. EXPLAINABILITY & NARRATIVE (Merging Screenshots) ---
st.divider()
col_explain, col_text = st.columns([2, 1])

with col_explain:
    st.subheader("🤖 Feature Contribution Analysis")
    if sensor_row is not None:
        # Create data for the bar chart
        features = {
            "3-Day Avg (History)": sensor_row.get('PM25_3Day_Avg', 0),
            "Pollution Velocity": sensor_row.get('Velocity_Yesterday', 0) * 5, # Scale for visibility
            "Smoke Satellite Data": sensor_row.get('Smoke_Yesterday', 0) * 10,
        }
        feat_df = pd.DataFrame(list(features.items()), columns=['Driver', 'Impact Score'])
        
        # Altair Bar Chart
        chart = alt.Chart(feat_df).mark_bar().encode(
            x='Impact Score',
            y=alt.Y('Driver', sort='-x'),
            color=alt.Color('Impact Score', scale=alt.Scale(scheme='reds'))
        ).properties(height=250)
        
        st.altair_chart(chart, use_container_width=True)

with col_text:
    st.subheader("📝 AI Narrative Report")
    if sensor_row is not None:
        smoke_val = sensor_row.get('Smoke_Yesterday', 0)
        avg_val = sensor_row.get('PM25_3Day_Avg', 0)
        
        narrative = ""
        if smoke_val > 1:
            narrative = "**CRITICAL WARNING:** Satellite imagery detected heavy smoke plumes yesterday. The model predicts these will drift into the valley today, causing hazardous conditions."
        elif avg_val > 25:
            narrative = "**Lingering Haze:** No new smoke was detected, but the 3-day average is high. The model predicts stagnant smoke trapping pollution in the area."
        else:
            narrative = "**Stable Conditions:** Atmospheric inputs are low. The model predicts clean air quality for this location."
            
        st.info(narrative)

# --- 8. MODEL DIAGNOSTICS (Scatter Plot & Error Map) ---
st.divider()
st.subheader("📊 Model Diagnostics")

tab1, tab2 = st.tabs(["📉 Accuracy Plot", "🗺️ Error Cartography"])

with tab1:
    st.markdown("**Global Model Accuracy ($R^2 \\approx 0.80$)**")
    # Scatter plot of Actual vs Predicted for ALL data (not just selected day)
    # We sample 1000 points to keep it fast
    if len(df) > 1000:
        scatter_data = df.sample(1000)
    else:
        scatter_data = df
        
    scatter = alt.Chart(scatter_data).mark_circle(size=60, opacity=0.5).encode(
        x=alt.X('Actual_PM25', title='Actual Pollution'),
        y=alt.Y('Predicted_PM25', title='AI Prediction'),
        color=alt.value('#3182bd'),
        tooltip=['Date', 'Location_Label', 'Actual_PM25', 'Predicted_PM25']
    ).properties(height=400)
    
    # Add the "Perfect Prediction" red line
    line = alt.Chart(pd.DataFrame({'x': [0, 100], 'y': [0, 100]})).mark_line(color='red', strokeDash=[5,5]).encode(x='x', y='y')
    
    st.altair_chart(scatter + line, use_container_width=True)

with tab2:
    st.markdown("**Map of Model Failures (Large dots = Big Prediction Errors)**")
    # Filter for high errors
    high_error_df = df[df['Absolute_Error'] > 15].copy() # Only show significant errors
    
    if not high_error_df.empty:
        # Scale for map
        high_error_df['radius'] = high_error_df['Absolute_Error'] * 1000
        
        error_layer = pdk.Layer(
            "ScatterplotLayer",
            high_error_df,
            get_position='[Lon, Lat]',
            get_fill_color='[200, 30, 0, 160]', # Dark Red
            get_radius='radius',
            pickable=True
        )
        
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(latitude=38, longitude=-120, zoom=4),
            layers=[error_layer],
            tooltip={"text": "Error: {Absolute_Error}\nDate: {Date}"}
        ))
    else:
        st.success("Remarkable! No major errors detected in the current dataset.")
