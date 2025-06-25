# Electric-Load-Forecasting-Using-Data-Mining-Techniques

Project Overview
This repository contains the implementation of an end-to-end data mining and machine learning project focused on electric load forecasting using a Kaggle dataset. The dataset includes hourly electricity demand and weather measurements for ten major U.S. cities. The project encompasses three main components:

Cluster Analysis: Identifying groups of similar consumption-weather patterns across cities and time periods.
Predictive Modeling: Building and evaluating machine learning models to forecast future electricity demand.
Front-End Interface: Developing a user-friendly web interface for data input, model controls, and result visualization.

Repository Structure

/data/: Directory for storing the Kaggle dataset (not included in the repository; see Dataset Description for download instructions).
/notebooks/: Jupyter notebooks for data preprocessing, clustering, and predictive modeling.
preprocessing.ipynb: Data loading, cleaning, feature engineering, and anomaly detection.
clustering.ipynb: Cluster analysis using K-Means, DBSCAN, and Hierarchical Clustering.
forecasting.ipynb: Predictive modeling with regression, time series, and ensemble methods.


/frontend/: Source code for the single-page React application.
src/: React components, including input forms, visualization plots, and user controls.


/docs/: Project documentation.
report.md: Summary of methods, results, and discussion.


requirements.txt: Python dependencies for the data analysis and modeling components.
README.md: This file.

Dataset Description

Source: The dataset must be downloaded from Kaggle (replace Download_Dataset with the actual Kaggle dataset link).
Features:
Timestamp (date and hour)
City name
Temperature (°F)
Humidity (%)
Wind speed (mph)
Hourly electricity demand (MWh)
Optional: Other weather variables (e.g., pressure, precipitation) if available.



Installation

Clone the Repository:
git clone https://github.com/your-username/electric-load-forecasting.git
cd electric-load-forecasting


Set Up Python Environment:

Install Python 3.8+.
Create and activate a virtual environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt




Set Up Front-End Environment:

Navigate to the /frontend directory:cd frontend


Install Node.js dependencies:npm install




Download Dataset:

Download the dataset from Kaggle and place the CSV files in the /data/ directory.



Usage

Data Preprocessing and Analysis:

Open the Jupyter notebooks in the /notebooks/ directory.
Run preprocessing.ipynb to load, clean, and preprocess the dataset.
Run clustering.ipynb for cluster analysis and visualization.
Run forecasting.ipynb to train and evaluate predictive models.


Run the Front-End Interface:

Navigate to the /frontend/ directory:cd frontend


Start the React application:npm start


Open http://localhost:3000 in a web browser to interact with the interface.


Explore the Interface:

Use the input form to select a city and date range.
Adjust model parameters (e.g., number of clusters, look-back window) via sliders or dropdowns.
Visualize clusters and forecasted demand through interactive plots.



Project Components
1. Data Preprocessing

Tasks:
Merge CSV files for all cities into a unified dataset.
Handle missing values (impute or remove).
Extract time-based features (hour, day, month, season).
Normalize/scale continuous variables.
Detect and handle anomalies using statistical or machine learning methods (e.g., z-score, Isolation Forest).


Output: Cleaned dataset ready for analysis.

2. Clustering Task

Objective: Segment hourly observations into clusters based on weather and consumption patterns.
Methods:
Dimensionality reduction (PCA or t-SNE).
Clustering algorithms: K-Means, DBSCAN, Hierarchical Clustering.


Evaluation: Silhouette score, cluster stability.
Output: Visualizations and a report section characterizing clusters (e.g., "high-demand hot afternoons").

3. Predictive Modeling

Objective: Forecast next-day hourly electricity demand.
Models:
Linear/Polynomial Regression
Time Series (ARIMA/SARIMA)
Machine Learning (Random Forest, XGBoost)
Neural Networks (LSTM or Feedforward ANN)


Evaluation Metrics: MAE, RMSE, MAPE.
Baseline: Naive forecast (previous day’s same hour).
Ensemble: At least one ensemble method (e.g., XGBoost, stacking).
Output: Jupyter notebook with code, visualizations, and performance summary.

4. Front-End Interface

Framework: React single-page application.
Features:
Input form for city, date range, and model parameters.
Interactive visualizations (cluster scatter plots, forecast time-series).
User controls (sliders, dropdowns, checkboxes) for model tuning.
Help section with instructions and technical details.


Output: Deployable web interface.

Submission Requirements

Jupyter Notebooks: Preprocessing, clustering, and forecasting code.
Front-End Code: React application source files.
Report: Markdown or PDF summarizing methods, results, and discussion.

Dependencies

Python: pandas, numpy, scikit-learn, xgboost, tensorflow, matplotlib, seaborn, jupyter.
Front-End: React, Tailwind CSS, Chart.js (or similar for visualizations).
See requirements.txt for the full list of Python dependencies.

