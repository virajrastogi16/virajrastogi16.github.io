import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import altair as alt
import pydeck as pdk
import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SmokeSignal AI",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DATA LOADING & CLEANING ---
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
        st.error("CRITICAL ERROR: 'data.zip' not found on GitHub. Please upload it.")
        st.stop()

    # --- CLEANING ---
    df.columns = df.columns.str.strip()

    # 1. Date Parsing
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        st.error("❌ 'Date' column missing.")
        st.stop()

    # 2. Location Cleaning
    df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
    df['Lon'] = pd.to_numeric(df['Lon'], errors='coerce')
    df = df.dropna(subset=['Lat', 'Lon'])

    df['Location_Label'] = df['Lat'].astype(str) + ", " + df['Lon'].astype(str)
    
    # 3. State Mapping
    state_map = {6:'California (CA)', 41:'Oregon (OR)', 53:'Washington (WA)', 32:'Nevada (NV)', 4:'Arizona (AZ)'}
    if 'State_ID' in df.columns:
        df['State_Name'] = df['State_ID'].map(state_map).fillna('Other')
    else:
        df['State_Name'] = 'All Regions'

    # 4. Error Metrics
    if 'Actual_PM25' in df.columns and 'Predicted_PM25' in df.columns:
        df['Error'] = df['Actual_PM25'] - df['Predicted_PM25']
        df['Absolute_Error'] = df['Error'].abs()
        
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Data Loading Error: {e}")
    st.stop()

# --- 3. SIDEBAR CONTROLS (UPDATED FOR DEFAULT DATE) ---
st.sidebar.title("🔍 Controls")

# A. Date Filter (Defaults to Aug 31, 2023 for demo)
min_d, max_d = df['Date'].min(), df['Date'].max()

# --- TARGET DATE SETTING ---
target_default = datetime.date(2023, 8, 31)
# Check if target is valid, otherwise fallback to max date
if min_d.date() <= target_default <= max_d.date():
    default_date = target_default
else:
    default_date = max_d

selected_date = st.sidebar.date_input("Analysis Date:", value=default_date, min_value=min_d, max_value=max_d)

# B. State Filter (Default to California for demo)
state_options = ['All'] + sorted(df['State_Name'].unique().tolist())

# Try to set default index to California if it exists
try:
    default_state_idx = state_options.index("California (CA)")
except ValueError:
    default_state_idx = 0

selected_state = st.sidebar.selectbox("Filter by State:", state_options, index=default_state_idx)

# Filter Logic
day_data = df[df['Date'].dt.date == selected_date]
if selected_state != 'All':
    filtered_data = day_data[day_data['State_Name'] == selected_state]
else:
    filtered_data = day_data

if filtered_data.empty:
    st.warning("No data for these filters. Try a different date.")

# C. Sensor Selector
sensor_options = filtered_data['Location_Label'].unique()
sensor_row = None
if len(sensor_options) > 0:
    selected_sensor = st.sidebar.selectbox("Select Sensor:", sensor_options)
    sensor_subset = filtered_data[filtered_data['Location_Label'] == selected_sensor]
    if not sensor_subset.empty:
        sensor_row = sensor_subset.iloc[0]

# --- 4. MAIN HEADER ---
st.title("🌲 West Coast SmokeSignal: Wildfire AI")
st.markdown(f"**Tracking {len(filtered_data)} sensors on {selected_date}**")

# Metrics Row
if sensor_row is not None:
    pred = sensor_row.get('Predicted_PM25', 0)
    actual = sensor_row.get('Actual_PM25', 0)
    
    if pred > 35: 
        status, s_color, emoji = "HAZARDOUS", "red", "🚨"
    elif pred > 12: 
        status, s_color, emoji = "MODERATE", "orange", "⚠️"
    else: 
        status, s_color, emoji = "SAFE", "green", "✅"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted PM2.5", f"{pred:.1f}")
    c2.metric("Actual PM2.5", f"{actual:.1f}", delta=f"{actual-pred:.1f}")
    c3.metric("Pollution Velocity", f"{sensor_row.get('Velocity_Yesterday', 0):.2f}")
    c4.markdown(f"### {emoji} :{s_color}[{status}]")

st.divider()

# --- 5. TABS INTERFACE ---
tab_map, tab_explain, tab_diag = st.tabs(["🌍 Regional Map", "🤖 Explainability", "📊 Diagnostics"])

# ================= TAB 1: MAP =================
with tab_map:
    st.subheader("Regional Air Quality Map")
    
    if not filtered_data.empty:
        # 1. Colors
        def get_color(val):
            if val < 12: return [0, 128, 0, 200]
            elif val < 35: return [255, 165, 0, 200]
            return [200, 30, 0, 200]

        map_df = filtered_data.copy()
        map_df['color'] = map_df['Predicted_PM25'].apply(get_color)
        map_df['radius'] = map_df['Predicted_PM25'].clip(lower=5) * 500 

        # 2. View State
        view_state = pdk.ViewState(
            latitude=map_df['Lat'].mean(),
            longitude=map_df['Lon'].mean(),
            zoom=5,
            pitch=0
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            map_df,
            get_position='[Lon, Lat]',
            get_fill_color='color',
            get_radius='radius',
            pickable=True
        )

        # 3. Chart
        st.pydeck_chart(pdk.Deck(
            initial_view_state=view_state,
            layers=[layer],
            tooltip={
                "html": "<b>Location:</b> {Location_Label}<br/>"
                        "<b>Predicted:</b> {Predicted_PM25}<br/>"
                        "<b>Actual:</b> {Actual_PM25}",
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }
        ))
    else:
        st.info("No data for map.")

# ================= TAB 2: EXPLAINABILITY =================
with tab_explain:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### Feature Contribution")
        if sensor_row is not None:
            feats = {
                "Yesterday's Pollution": sensor_row.get('PM25_Yesterday', 0),
                "3-Day Avg": sensor_row.get('PM25_3Day_Avg', 0),
                "Velocity": sensor_row.get('Velocity_Yesterday', 0) * 5, 
                "Smoke Sat.": sensor_row.get('Smoke_Yesterday', 0) * 10
            }
            feat_df = pd.DataFrame(list(feats.items()), columns=['Feature', 'Impact'])
            
            c = alt.Chart(feat_df).mark_bar().encode(
                x='Impact', 
                y=alt.Y('Feature', sort='-x'),
                color=alt.Color('Impact', scale=alt.Scale(scheme='reds'))
            ).properties(height=350)
            st.altair_chart(c, use_container_width=True)
            
    with c2:
        st.markdown("#### Narrative")
        if sensor_row is not None:
            if sensor_row.get('Smoke_Yesterday', 0) > 1:
                st.warning("Detected heavy smoke plumes drifting into area.")
            elif sensor_row.get('PM25_3Day_Avg', 0) > 25:
                st.info("Stagnant air trapping existing pollution.")
            else:
                st.success("Stable atmospheric conditions.")

# ================= TAB 3: DIAGNOSTICS =================
with tab_diag:
    st.markdown("#### Global Accuracy")
    
    # Safe sampling
    if not df.empty:
        chart_data = df.sample(min(1000, len(df)))
        max_val = max(chart_data['Actual_PM25'].max(), chart_data['Predicted_PM25'].max())
        
        scatter = alt.Chart(chart_data).mark_circle(size=60).encode(
            x='Actual_PM25', y='Predicted_PM25', tooltip=['Date', 'Actual_PM25']
        ).properties(height=500)
        
        line = alt.Chart(pd.DataFrame({'x':[0, max_val], 'y':[0, max_val]})).mark_line(color='red').encode(x='x', y='y')
        st.altair_chart(scatter + line, use_container_width=True)

    st.divider()
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("#### Time Series")
        if sensor_row is not None:
            loc = sensor_row['Location_Label']
            hist = df[df['Location_Label'] == loc].sort_values('Date')
            
            # SAFE MELT
            hist_clean = hist.reset_index(drop=True)
            if 'Actual_PM25' in hist_clean.columns and 'Predicted_PM25' in hist_clean.columns:
                hist_mini = hist_clean[['Date', 'Actual_PM25', 'Predicted_PM25']]
                m = hist_mini.melt(id_vars='Date', var_name='Type', value_name='PM25')
                
                l = alt.Chart(m).mark_line().encode(
                    x='Date:T', y='PM25:Q', color='Type:N'
                ).properties(height=400)
                st.altair_chart(l, use_container_width=True)
            else:
                st.error("Data missing columns for timeline.")
        else:
            st.info("Select a sensor.")
    
    with col_d2:
        st.markdown("#### Error Map")
        errs = df[df['Absolute_Error'] > 15].dropna(subset=['Lat', 'Lon'])
        if not errs.empty:
            err_layer = pdk.Layer(
                "ScatterplotLayer", errs,
                get_position='[Lon, Lat]',
                get_fill_color='[200, 30, 0, 200]',
                get_radius=5000, pickable=True
            )
            
            view_err = pdk.ViewState(
                latitude=errs['Lat'].mean(),
                longitude=errs['Lon'].mean(),
                zoom=4
            )
            
            st.pydeck_chart(pdk.Deck(
                initial_view_state=view_err,
                layers=[err_layer],
                tooltip={"text": "Error: {Absolute_Error:.1f}"}
            ))
        else:
            st.success("No major errors.")
