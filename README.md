# 🚨 Chicago Traffic Safety ML & Analytics Dashboard

## 📖 Overview
Developed an interactive data visualization and analytics web application using Python and Streamlit to analyze Chicago traffic accidents. The project integrates data preprocessing and geospatial analysis using Pandas and GeoPandas for coordinate mapping and distance calculations (CRS transformations), alongside an unsupervised Machine Learning pipeline via Scikit-Learn (K-Means clustering, Label Encoding, Standard Scaling) to automatically classify accident risk zones. Additionally, a statistical forecasting simulator using Statsmodels (OLS Multiple Linear Regression) is included to predict casualty numbers based on environmental variables such as speed limits and vehicle involvement.

## ✨ Key Features
* **Interactive Filtering & Export:** Filter accidents dynamically using a sidebar slider based on posted speed limits and download the cleaned CSV dataset directly from the UI.
* **Geospatial Mapping:** Visualize accident hotspots on an interactive map. It calculates the physical distance (in km) of each crash from the Chicago city center using GeoPandas and spatial re-projection (`EPSG:3857`).
* **Risk Factor Analysis:** Analyzes categorical data, such as lighting conditions, to calculate total accidents and injuries using Pandas aggregation.
* **Unsupervised Machine Learning:** Utilizes Scikit-Learn's K-Means clustering algorithm (k=3) to automatically classify crashes into distinct risk zones (e.g., Medium Risk, Critical Speed Risk, Low Risk Congestion) based on standardized speed and distance variables.
* **Statistical Forecasting Simulator:** Features an interactive police simulator backed by an OLS Multiple Linear Regression model (Statsmodels) to mathematically estimate potential casualties based on the speed limit and the number of vehicles involved in the crash.

## 🛠️ Technologies Used
* **Language:** Python
* **Web Framework:** Streamlit
* **Data Processing & Geospatial:** Pandas, GeoPandas, Shapely
* **Machine Learning:** Scikit-Learn (K-Means, LabelEncoder, StandardScaler)
* **Statistical Modeling:** Statsmodels (OLS Regression)
* **Visualization:** Plotly Express

## 🚀 How to Run the Project Locally

1. **Clone the repository:**
```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
```
2. **Install the required dependencies:**
Make sure you have Python installed, then run the following command to install the required libraries:
```
pip install -r requirements.txt
```
3. **Add the dataset:**
Ensure the chicago_accidents.csv file is placed in the root directory of the project (or update the path in main.py).
4. **Launch the app:**
```
streamlit run main.py
```
The app will automatically open in your default web browser at http://localhost:8501.
