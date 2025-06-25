# Electric-Load-Forecasting-Using-Data-Mining-Techniques 📈 

## Project Overview
This repository contains the implementation of an end-to-end data mining and machine learning project focused on electric load forecasting using a Kaggle dataset. The dataset includes hourly electricity demand and weather measurements for ten major U.S. cities. The project encompasses three main components:

**Cluster Analysis**: Identifying groups of similar consumption-weather patterns across cities and time periods.
**Predictive Modeling**: Building and evaluating machine learning models to forecast future electricity demand.
**Front-End Interface**: Developing a user-friendly web interface for data input, model controls, and result visualization.


### Repository Structure
-**analyzer.py**: 1. Data loading, cleaning, feature engineering, and anomaly detection. 
             2. Cluster analysis using K-Means, DBSCAN, and Hierarchical Clustering. 
             3. Predictive modeling with regression, time series, and ensemble methods.

-**app.py**: Develops a connection between the frontend and backend. Flask is used in this file.

-**index.html, styles.css, scripts.js**: Covers the frontend to display the graphs.

-**README.md**: This file.


### Dataset Description
Source: The dataset must be downloaded from Kaggle (replace Download_Dataset with the actual Kaggle dataset link).
Features: 1. Timestamp (date and hour)
          2. City name
          3. Temperature (°F)
          4. Humidity (%)
          5. Wind speed (mph)
          6. Hourly electricity demand (MWh)
Optional: Other weather variables (e.g., pressure, precipitation) if available.

**Download Dataset**: Download the dataset from Kaggle as a zip file.

