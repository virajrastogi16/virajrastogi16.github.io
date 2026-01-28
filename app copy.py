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

# --- 2. DATA LOADING (Robust & Safe) ---
@st.cache_data
def load_data():
    try:
        # robustly handle the mac zip file issue
        with zipfile.ZipFile("data.zip", "r") as z:
            all_files = z.namelist()
            # find the actual csv, ignoring mac metadata folders
            csv_files = [f for f in all_files if f.endswith('.csv') and not f.startswith('__MACOSX')]
            
            if not csv_files:
                st.error("Error: No CSV file found inside data.zip")
                st.stop()
            
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f)
                
    except FileNotFoundError:
        st.error("CRITICAL ERROR: Could not find 'data.zip'. Please ensure it is uploaded to your GitHub repository.")
        st.stop()

    # --- PREPROCESSING ---
    # 1. Location Label
    if 'Lat' in df.columns and 'Lon' in df.columns:
        df['Location_Label'] = df['Lat'].astype(str) + ", " + df['Lon'].astype(str)
    
    # 2. Date Parsing
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
    # 3. State Mapping (Fixes missing state names)
    # Mapping ID to Name based on typical West Coast sensor IDs
    state_map = {
        6: 'California (CA)', 
        41: 'Oregon (OR)', 
        53: 'Washington (WA)', 
        32: 'Nevada (NV)', 
        4: 'Arizona (AZ)'
    }
    if 'State_ID' in df.columns:
        df['State_Name'] = df['State_ID'].map(state_map).fillna('Other')
    else:
        df['State_Name'] = 'All Regions'

    # 4. Error Calculation for Diagnostics
    if 'Actual_PM25' in df.columns and 'Predicted_PM25' in df.columns:
        df['Error'] = df['Actual_PM25'] - df['Predicted_PM25']
        df['Absolute_Error'] = df['Error'].abs()
        
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Data Load Error: {e}")
    st.stop()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("🔍 Regional Controls")

# A. Date Selection
min_date = df['Date'].min()
max_date = df['Date'].max()
selected_date = st.sidebar.date_input(
    "Select Analysis Date:",
    min_value=min_date,
    max_value=max_date,
    value=min_date 
)

# Filter Master Data by Date FIRST
day_data = df[df['Date'].dt.date == selected_date]

if day_data.empty:
    st.warning(f"No data available for {selected_date}.")
    st.stop()

# B. State Filter
state_options = ['All'] + sorted(df['State_Name'].unique().tolist())
selected_state = st.sidebar.selectbox("Filter by State:", state_options)

# Apply State Filter
if selected_state != 'All':
    filtered_data = day_data[day_data['State_Name'] == selected_state]
else:
    filtered_data = day_data

# C. Sensor Selection (Populated by the filtered data)
# This fixes the "only 1 element" issue by showing all valid sensors for that state/day
sensor_options = filtered_data['Location_Label'].unique()
if len(sensor_options) > 0:
    selected_sensor = st.sidebar.selectbox("Select Specific Sensor:", sensor_options)
    # Get the specific row for metrics
    sensor_row = filtered_data[filtered_data['Location_Label'] == selected_sensor].iloc[0]
else:
    st.sidebar.warning("No sensors found for this selection.")
    sensor_row = None


# --- 4. MAIN DASHBOARD ---
st.title("🌲 West Coast SmokeSignal: Wildfire AI")
st.markdown(f"### Tracking PM2.5 Levels across **{len(filtered_data)}** active sensors")
st.divider()

# --- TOP METRICS ROW ---
if sensor_row is not None:
    # Safely get values
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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted PM2.5", f"{pred:.1f} µg/m³")
    c2.metric("Actual PM2.5", f"{actual:.1f} µg/m³", delta=f"{actual-pred:.1f}")
    c3.metric("Pollution Velocity", f"{velocity:.2f}")
    c4.markdown(f"### Status: :{s_color}[{status}]")

st.divider()

# --- 5. THE THREE TABS ---
tab_map, tab_explain, tab_diag = st.tabs(["🌍 Regional Map", "🤖 Explainability", "📊 Model Diagnostics"])

# ==========================================
# TAB 1: REGIONAL MAP (The Colorful Dots)
# ==========================================
with tab_map:
    st.subheader(f"Regional Air Quality Map ({selected_date})")
    
    if not filtered_data.empty:
        # 1. Color Logic (Green -> Yellow -> Red)
        def get_color(val):
            if val < 12: return [0, 128, 0, 160] # Green
            elif val < 35: return [255, 165, 0, 160] # Orange
            return [255, 0, 0, 160] # Red

        # Prepare map data
        map_df = filtered_data.copy()
        map_df['color'] = map_df['Predicted_PM25'].apply(get_color)
        # Dynamic radius based on pollution (bigger dots for worse air)
        map_df['radius'] = map_df['Predicted_PM25'].clip(lower=10) * 200

        # 2. Pydeck Layer
        layer = pdk.Layer(
            "ScatterplotLayer",
            map_df,
            get_position='[Lon, Lat]',
            get_color='color',
            get_radius='radius',
            pickable=True,
            radius_min_pixels=5,
            radius_max_pixels=50,
        )

        # 3. Dynamic View State (Fixes "I cannot see the map" issue)
        # Centers the map automatically on the available data points
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
            tooltip={"text": "Location: {Location_Label}\nPred PM2.5: {Predicted_PM25:.1f}"}
        ))
    else:
        st.info("No data points to display on map.")

# ==========================================
# TAB 2: EXPLAINABILITY
# ==========================================
with tab_explain:
    col_x1, col_x2 = st.columns([2, 1])
    
    with col_x1:
        st.subheader("Why did the AI predict this?")
        if sensor_row is not None:
            # Create feature data
            features = {
                "3-Day Avg (History)": sensor_row.get('PM25_3Day_Avg', 0),
                "Pollution Velocity": sensor_row.get('Velocity_Yesterday', 0) * 5, 
                "Smoke Satellite Data": sensor_row.get('Smoke_Yesterday', 0) * 10,
            }
            feat_df = pd.DataFrame(list(features.items()), columns=['Driver', 'Impact Score'])
            
            # Altair Bar Chart
            chart = alt.Chart(feat_df).mark_bar().encode(
                x='Impact Score',
                y=alt.Y('Driver', sort='-x'),
                color=alt.Color('Impact Score', scale=alt.Scale(scheme='reds'))
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)
            
    with col_x2:
        st.subheader("📝 AI Narrative Report")
        if sensor_row is not None:
            smoke = sensor_row.get('Smoke_Yesterday', 0)
            avg = sensor_row.get('PM25_3Day_Avg', 0)
            
            if smoke > 1:
                st.warning("**CRITICAL DRIVER:** Satellite imagery detected heavy smoke plumes yesterday. The model predicts drift into this area.")
            elif avg > 25:
                st.info("**LINGERING HAZE:** No new smoke detected, but the 3-day average is high. Model predicts stagnation.")
            else:
                st.success("**STABLE:** Atmospheric inputs are low. Model predicts clean air.")

# ==========================================
# TAB 3: MODEL DIAGNOSTICS
# ==========================================
with tab_diag:
    st.subheader("Model Performance & Diagnostics")
    
    # 1. Accuracy Plot (Global)
    st.markdown("#### 1. Accuracy: Actual vs. Predicted (Global)")
    # Sample data to keep chart fast
    if len(df) > 1000:
        scatter_data = df.sample(1000)
    else:
        scatter_data = df
        
    scatter = alt.Chart(scatter_data).mark_circle(size=60, opacity=0.5).encode(
        x=alt.X('Actual_PM25', title='Actual Pollution'),
        y=alt.Y('Predicted_PM25', title='AI Prediction'),
        color=alt.value('#3182bd'),
        tooltip=['Date', 'Location_Label', 'Actual_PM25', 'Predicted_PM25']
    ).properties(height=350)
    
    line = alt.Chart(pd.DataFrame({'x': [0, 100], 'y': [0, 100]})).mark_line(color='red', strokeDash=[5,5]).encode(x='x', y='y')
    st.altair_chart(scatter + line, use_container_width=True)
    st.caption("Points on the red line represent perfect predictions.")
    
    st.divider()

    # 2. Timeline & Error Map
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("#### 2. Forecast Tracking over Time")
        # Get history for the SELECTED sensor
        if sensor_row is not None:
            sensor_loc = sensor_row['Location_Label']
            history = df[df['Location_Label'] == sensor_loc].sort_values('Date')
            
            # Melt for multi-line chart
            chart_hist = history.melt(id_vars=['Date'], value_vars=['Actual_PM25', 'Predicted_PM25'], var_name='Type', value_name='PM25')
            
            line_chart = alt.Chart(chart_hist).mark_line().encode(
                x='Date:T',
                y='PM25:Q',
                color=alt.Color('Type', scale=alt.Scale(domain=['Actual_PM25', 'Predicted_PM25'], range=['#1f77b4', '#ff7f0e'])),
                tooltip=['Date', 'Type', 'PM25']
            ).properties(height=300)
            st.altair_chart(line_chart, use_container_width=True)
        else:
            st.info("Select a sensor to see its history.")

    with col_d2:
        st.markdown("#### 3. Error Cartography (Where does AI struggle?)")
        # Filter for high errors
        high_error_df = df[df['Absolute_Error'] > 15].copy()
        
        if not high_error_df.empty:
            high_error_df['radius'] = high_error_df['Absolute_Error'] * 500
            
            err_layer = pdk.Layer(
                "ScatterplotLayer",
                high_error_df,
                get_position='[Lon, Lat]',
                get_fill_color='[200, 30, 0, 180]', # Dark Red
                get_radius='radius',
                pickable=True
            )
            
            # Center on errors
            err_view = pdk.ViewState(
                latitude=high_error_df['Lat'].mean(),
                longitude=high_error_df['Lon'].mean(),
                zoom=4,
                pitch=0
            )
            
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=err_view,
                layers=[err_layer],
                tooltip={"text": "Error: {Absolute_Error:.1f}"}
            ))
        else:
            st.success("No major errors (>15) detected.")
