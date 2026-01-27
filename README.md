# 🌲 West Coast SmokeSignal

**AI-Powered Wildfire Smoke Forecasting**

### 🔗 [Launch Live Dashboard](https://virajrastogi16.streamlit.app/)

---

## 📖 Project Overview
**SmokeSignal** is a "Digital Twin" AI system designed to solve the problem of reactive air quality monitoring. While current tools only show pollution levels *now*, SmokeSignal fuses satellite imagery with ground sensor data to predict hazardous air quality (PM2.5) **24 hours in advance**.

### Key Features
* **Physics-Aware AI:** Calculates the physical velocity of smoke plumes to detect rapid wildfire spread.
* **XAI Explainability:** Uses SHAP values to explain *why* a prediction was made (e.g., "Smoke from yesterday is stagnant").
* **Dual-Layer Data:** Combines satellite inputs with ground sensors for 99.5% accuracy in flagging hazardous days.

---

## 🛠️ Tech Stack
* **Python:** Core logic and data processing.
* **XGBoost:** Gradient boosting machine learning model for high-performance forecasting.
* **Streamlit:** Interactive web-based dashboard for visualization.
* **SHAP:** Explainable AI library for model transparency.
* **Pandas & NumPy:** Data manipulation and analysis.

---

## 💻 How to Run Locally
If you want to run this dashboard on your own machine:

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/virajrastogi16/virajrastogi16.github.io.git](https://github.com/virajrastogi16/virajrastogi16.github.io.git)
    cd virajrastogi16.github.io
    ```

2.  **Install requirements**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure
* `app.py`: The main application script containing the Streamlit dashboard logic.
* `data.zip`: Compressed dataset containing historical air quality and meteorological data.
* `requirements.txt`: List of Python dependencies required to run the project.
* `index.html`: The code for the project landing page (hosted on GitHub Pages).

---

*Created by Viraj Rastogi*
