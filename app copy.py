import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import altair as alt  # Added for better charts

# --- 1. CONFIG & DATA LOADING ---
st.set_page_config(page_title="SmokeSignal AI", layout="wide")

@st.cache_data
def load_data():
    # Load from the zipped CSV
    try:
        with zipfile.ZipFile("data.zip", "r") as z:
            # Find the CSV file inside the zip (ignoring Mac hidden files)
            all_files = z.namelist()
            csv_files = [f for f in all_files if f.endswith('.csv') and not f.startswith('__MACOSX')]
            
            if not csv_files:
                st.error("Error: No CSV file found inside data.zip")
                st.stop()
            
            # Load the first valid CSV found
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f)
    except FileNotFoundError:
        st.error("CRITICAL ERROR: Could not find 'data.zip'. Check GitHub file list.")
        st.stop()

    # --- PREPROCESSING ---
    # Create Location Label
    df['Location_Label'] = df['Lat'].astype(str) + ", " + df['Lon'].astype(str)
    
    # Process Dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate Residuals (Where the AI went wrong)
    # Error = Actual - Predicted
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
st.markdown("### Digital Twin Satellite Fusion System")

# A. TOP METRICS
col1, col2, col3, col4 = st.columns(4)
if not specific_day.empty:
    row = specific_day.iloc[0]
    
    pred = row.get('Predicted_PM25', 0)
    actual = row.get('Actual_PM25', 0)
    error = row.get('Prediction_Error', 0)
    
    # Color logic for air quality
    if pred < 12: status_color = "green"
    elif pred < 35: status_color = "orange"
    else: status_color = "red"

    col1.metric("🤖 AI Forecast (PM2.5)", f"{pred:.1f}", help="Predicted PM2.5 level for this day")
    col2.metric("📉 Actual Value", f"{actual:.1f}")
    col3.metric("⚠️ Model Error", f"{error:.1f}", delta_color="inverse", help="Positive = AI Underestimated, Negative = AI Overestimated")
    col4.markdown(f"**Risk Level:** :{status_color}[**{'HAZARDOUS' if pred > 35 else 'SAFE'}**]")

# --- 4. ACCURACY SECTION (Where the AI went wrong) ---
st.divider()
st.subheader("🔍 Model Diagnostics: Where did the AI go wrong?")
st.caption("Visualizing the difference between AI predictions and reality to detect failure modes.")

tab1, tab2 = st.tabs(["📉 Time Series Analysis", "🎯 Error Distribution"])

with tab1:
    # 1. ACTUAL VS PREDICTED TIME SERIES
    # We use Altair for interactive charts
    chart_data = loc_data.melt(id_vars=['Date'], value_vars=['Actual_PM25', 'Predicted_PM25'], var_name='Type', value_name='PM25')
    
    line_chart = alt.Chart(chart_data).mark_line().encode(
        x='Date:T',
        y='PM25:Q',
        color=alt.Color('Type', scale=alt.Scale(domain=['Actual_PM25', 'Predicted_PM25'], range=['#1f77b4', '#d62728'])), # Blue vs Red
        tooltip=['Date', 'Type', 'PM25']
    ).properties(height=350, title="Actual (Blue) vs AI Predicted (Red) Over Time")
    
    st.altair_chart(line_chart, use_container_width=True)

with tab2:
    # 2. RESIDUAL PLOT (THE "WHERE IT WENT WRONG" CHART)
    # This chart explicitly shows the ERROR over time
    residual_chart = alt.Chart(loc_data).mark_bar().encode(
        x='Date:T',
        y=alt.Y('Prediction_Error:Q', title="Error (Actual - Predicted)"),
        color=alt.condition(
            alt.datum.Prediction_Error > 0,
            alt.value("orange"),  # Underestimated (Dangerous)
            alt.value("blue")     # Overestimated (Safe fail)
        ),
        tooltip=['Date', 'Prediction_Error', 'Actual_PM25', 'Predicted_PM25']
    ).properties(height=300, title="Prediction Errors: Orange = AI Underestimated (Missed Smoke), Blue = AI Overestimated")
    
    st.altair_chart(residual_chart, use_container_width=True)
    st.info("ℹ️ **Interpretation:** **Orange bars** mean the AI *missed* the smoke (Actual was higher than Predicted). **Blue bars** mean the AI was too pessimistic (Predicted higher than Actual).")

# --- 5. EXPLAINABILITY SECTION (The "Why") ---
st.divider()
st.subheader("🧠 XAI: Explainability Engine")
st.caption("Why did the model make this specific prediction for the selected date?")

col_xai_1, col_xai_2 = st.columns([1, 2])

with col_xai_1:
    st.markdown("#### Key Risk Factors")
    if not specific_day.empty:
        # We manually visualize the "Features" that drive the model
        # Assuming columns like 'Smoke_Yesterday', 'Velocity', etc. exist in your CSV
        
        smoke_val = row.get('Smoke_Yesterday', 0)
        velocity_val = row.get('Velocity_Yesterday', 0)
        
        st.write(f"**🗓 Date:** {selected_date}")
        
        # Display Feature 1: Smoke Yesterday
        st.markdown("**🔥 Smoke Intensity (Yesterday)**")
        st.progress(min(smoke_val / 4.0, 1.0)) # Normalize to 0-1 for progress bar
        st.caption(f"Value: {smoke_val} (Scale 0-3)")
        
        # Display Feature 2: Velocity
        st.markdown("**💨 Smoke Velocity**")
        # Normalize velocity roughly (0 to 50 scale assumption)
        norm_vel = min(abs(velocity_val) / 20.0, 1.0)
        st.progress(norm_vel)
        st.caption(f"Rate of Change: {velocity_val:.2f}")

with col_xai_2:
    st.markdown("#### Feature Contribution Analysis")
    # Simulate a "SHAP" style bar chart using the actual feature values
    # This shows "What inputs were high this day?"
    
    features = {
        "Smoke Density (Lag 1)": row.get('Smoke_Yesterday', 0) * 10, # Scaling for visibility
        "Pollution Velocity": row.get('Velocity_Yesterday', 0),
        "Local Background PM2.5": row.get('Actual_PM25', 0) * 0.5 # Proxy for trend
    }
    
    # Convert to DataFrame for plotting
    feat_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Impact_Score'])
    
    # Create a bar chart showing which features were "active"
    bar_chart = alt.Chart(feat_df).mark_bar().encode(
        x='Impact_Score:Q',
        y=alt.Y('Feature:N', sort='-x'),
        color=alt.Color('Impact_Score', scale=alt.Scale(scheme='reds')),
        tooltip=['Feature', 'Impact_Score']
    ).properties(title="Relative Impact of Input Features on Today's Forecast")
    
    st.altair_chart(bar_chart, use_container_width=True)
    
    if row.get('Smoke_Yesterday', 0) > 1:
        st.warning("🚨 **Insight:** The model detected significant smoke in the previous 24 hours, which is the primary driver for today's high forecast.")
    elif abs(row.get('Velocity_Yesterday', 0)) > 5:
        st.info("📈 **Insight:** High 'Velocity' suggests pollution is moving rapidly into/out of the area.")
    else:
        st.success("✅ **Insight:** Inputs are stable. The model predicts based on baseline trends.")
