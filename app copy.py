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
        # Open zip file
        with zipfile.ZipFile("data.zip", "r") as z:
            all_files = z.namelist()
            csv_files = [f for f in all_files if f.endswith('.csv') and not f.startswith('__MACOSX')]
            
            if not csv_files:
                st.error("Error: No CSV file found inside data.zip")
                st.stop()
            
            # Load CSV
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f)
                
    except FileNotFoundError:
        st.error("CRITICAL ERROR: 'data.zip' not found. Please upload it to GitHub.")
        st.stop()

    # --- CRITICAL FIX: CLEAN COLUMN NAMES ---
    # Remove invisible spaces from column names (Fixes the ValueError)
    df.columns = df.columns.str.strip()

    # --- PREPROCESSING ---
    # 1. Date Parsing
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        st.error("❌ Column 'Date' is missing from the dataset. Check your CSV.")
        st.stop()

    # 2. Location & State
    # Ensure Lat/Lon are numeric (Fixes Map Invisible Issue)
    df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
    df['Lon'] = pd.to_numeric(df['Lon'], errors='coerce')
    
    # Drop rows with invalid coordinates to prevent Map crash
    df = df.dropna(subset=['Lat', 'Lon'])

    df['Location_Label'] = df['Lat'].astype(str) + ", " + df['Lon'].astype(str)
    
    state_map = {
        6: 'California (CA)', 41: 'Oregon (OR)', 53: 'Washington (WA)', 
        32: 'Nevada (NV)', 4: 'Arizona (AZ)'
    }
    if 'State_ID' in df.columns:
        df['State_Name'] = df['State_ID'].map(state_map).fillna('Other')
    else:
        df['State_Name'] = 'All Regions'

    # 3. Error Calculation
    if 'Actual_PM25' in df.columns and 'Predicted_PM25' in df.columns:
        df['Error'] = df['Actual_PM25'] - df['Predicted_PM25']
        df['Absolute_Error'] = df['Error'].abs()
        
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Data Loading Failed: {e}")
    st.stop()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("🔍 Regional Controls")

# DEBUGGER: Show user if data is empty
if df.empty:
    st.sidebar.error("⚠️ The dataset is empty!")
    st.stop()

# A. Date Selection
min_date = df['Date'].min()
max_date = df['Date'].max()

# Default to the most recent date to ensure we see data
selected_date = st.sidebar.date_input(
    "Select Analysis Date:",
    min_value=min_date,
    max_value=max_date,
    value=max_date 
)

# Filter by Date
day_data = df[df['Date'].dt.date == selected_date]

# B. State Filter
state_options = ['All'] + sorted(df['State_Name'].unique().tolist())
selected_state = st.sidebar.selectbox("Filter by State:", state_options)

if selected_state != 'All':
    filtered_data = day_data[day_data['State_Name'] == selected_state]
else:
    filtered_data = day_data

# C. Sensor Selection
sensor_options = filtered_data['Location_Label'].unique()
sensor_row = None

if len(sensor_options) > 0:
    selected_sensor = st.sidebar.selectbox("Select Specific Sensor:", sensor_options)
    sensor_subset = filtered_data[filtered_data['Location_Label'] == selected_sensor]
    if not sensor_subset.empty:
        sensor_row = sensor_subset.iloc[0]
else:
    if not day_data.empty:
        st.sidebar.warning(f"No sensors found in {selected_state} on this date.")
    else:
        st.sidebar.warning(f"No data available for {selected_date}.")

# --- 4. MAIN DASHBOARD ---
st.title("🌲 West Coast SmokeSignal: Wildfire AI")
st.markdown(f"### Tracking PM2.5 Levels across **{len(filtered_data)}** active sensors")

# Top Metrics
if sensor_row is not None:
    pred = sensor_row.get('Predicted_PM25', 0)
    actual = sensor_row.get('Actual_PM25', 0)
    velocity = sensor_row.get('Velocity_Yesterday', 0)
    
    if pred > 35: status, s_color = "🚨 HAZARDOUS", "red"
    elif pred > 12: status, s_color = "⚠️ MODERATE", "orange"
    else: status, s_color = "✅ SAFE", "green"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted PM2.5", f"{pred:.1f} µg/m³")
    c2.metric("Actual PM2.5", f"{actual:.1f} µg/m³", delta=f"{actual-pred:.1f}")
    c3.metric("Pollution Velocity", f"{velocity:.2f}")
    c4.markdown(f"### Status: :{s_color}[{status}]")

st.divider()

# --- 5. TABS ---
tab_map, tab_explain, tab_diag = st.tabs(["🌍 Regional Map", "🤖 Explainability", "📊 Model Diagnostics"])

# TAB 1: MAP (Fixed Visibility)
with tab_map:
    st.subheader(f"Regional Air Quality Map ({selected_date})")
    
    if not filtered_data.empty:
        # Check for NaN values in Lat/Lon again just to be safe
        map_data = filtered_data.dropna(subset=['Lat', 'Lon', 'Predicted_PM25'])

        if not map_data.empty:
            # Color Logic
            def get_color(val):
                if val < 12: return [0, 128, 0, 160] # Green
                elif val < 35: return [255, 165, 0, 160] # Orange
                return [255, 0, 0, 160] # Red

            map_data['color'] = map_data['Predicted_PM25'].apply(get_color)
            map_data['radius'] = map_data['Predicted_PM25'].clip(lower=10) * 400

            layer = pdk.Layer(
                "ScatterplotLayer",
                map_data,
                get_position='[Lon, Lat]',
                get_fill_color='color',
                get_radius='radius',
                pickable=True
            )

            # Hard-coded view state fallback to ensure map isn't blank
            view_lat = map_data['Lat'].mean()
            view_lon = map_data['Lon'].mean()
            
            view_state = pdk.ViewState(
                latitude=view_lat,
                longitude=view_lon,
                zoom=5,
                pitch=0,
            )

            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"text": "Location: {Location_Label}\nPred: {Predicted_PM25:.1f}"}
            ))
        else:
            st.warning("Data exists but lacks valid coordinates for mapping.")
    else:
        st.info("No data available for the selected filters.")

# TAB 2: EXPLAINABILITY
with tab_explain:
    c_x1, c_x2 = st.columns([2, 1])
    with c_x1:
        st.subheader("Feature Contribution")
        if sensor_row is not None:
            features = {
                "3-Day Avg": sensor_row.get('PM25_3Day_Avg', 0),
                "Velocity": sensor_row.get('Velocity_Yesterday', 0) * 5, 
                "Smoke Sat.": sensor_row.get('Smoke_Yesterday', 0) * 10,
            }
            feat_df = pd.DataFrame(list(features.items()), columns=['Driver', 'Impact'])
            
            chart = alt.Chart(feat_df).mark_bar().encode(
                x='Impact',
                y=alt.Y('Driver', sort='-x'),
                color=alt.Color('Impact', scale=alt.Scale(scheme='reds'))
            )
            st.altair_chart(chart, use_container_width=True)
            
    with c_x2:
        st.subheader("AI Narrative")
        if sensor_row is not None:
            smoke = sensor_row.get('Smoke_Yesterday', 0)
            if smoke > 1: st.warning("Heavy smoke detected.")
            else: st.success("Conditions stable.")

# TAB 3: DIAGNOSTICS (Fixed ValueError)
with tab_diag:
    st.subheader("Model Performance")
    
    # 1. Global Accuracy
    if len(df) > 500:
        scatter_data = df.sample(500)
    else:
        scatter_data = df
        
    scatter = alt.Chart(scatter_data).mark_circle(size=60, opacity=0.5).encode(
        x='Actual_PM25',
        y='Predicted_PM25',
        tooltip=['Date', 'Actual_PM25', 'Predicted_PM25']
    ).properties(height=350)
    st.altair_chart(scatter, use_container_width=True)
    
    st.divider()
    col_d1, col_d2 = st.columns(2)

    # 2. Timeline (The Chart that Crashed)
    with col_d1:
        st.markdown("#### Forecast Tracking")
        if sensor_row is not None:
            loc = sensor_row['Location_Label']
            hist = df[df['Location_Label'] == loc].sort_values('Date')
            
            # SAFE MELT: Explicitly check columns before melting
            if 'Actual_PM25' in hist.columns and 'Predicted_PM25' in hist.columns:
                chart_data = hist[['Date', 'Actual_PM25', 'Predicted_PM25']].melt(
                    id_vars=['Date'], 
                    var_name='Type', 
                    value_name='PM25'
                )
                
                line = alt.Chart(chart_data).mark_line().encode(
                    x='Date:T',
                    y='PM25:Q',
                    color='Type:N'
                ).properties(height=300)
                st.altair_chart(line, use_container_width=True)
            else:
                st.error("Missing columns for timeline visualization.")
        else:
            st.info("Select a sensor.")

    # 3. Error Map
    with col_d2:
        st.markdown("#### Error Cartography")
        high_err = df[df['Absolute_Error'] > 15].dropna(subset=['Lat', 'Lon'])
        
        if not high_err.empty:
            err_layer = pdk.Layer(
                "ScatterplotLayer",
                high_err,
                get_position='[Lon, Lat]',
                get_fill_color='[200, 30, 0, 180]',
                get_radius=5000, # Large radius for visibility
                pickable=True
            )
            
            # Default view if empty
            lat_view = high_err['Lat'].mean() if not high_err.empty else 38
            lon_view = high_err['Lon'].mean() if not high_err.empty else -120

            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(latitude=lat_view, longitude=lon_view, zoom=4),
                layers=[err_layer],
                tooltip={"text": "Error: {Absolute_Error:.1f}"}
            ))
        else:
            st.success("No major errors detected.")
