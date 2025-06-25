from flask import Flask, render_template, request, jsonify
from analyzer import ElectricityDemandAnalyzer
import pandas as pd
import os
import traceback

app = Flask(__name__)

# Initialize the analyzer
analyzer = ElectricityDemandAnalyzer(zip_path='archive.zip')

# Load data on startup
try:
    analyzer.load_and_merge_data()
    analyzer.preprocess_data()
    analyzer.detect_anomalies()
except Exception as e:
    print(f"Error initializing data: {str(e)}")

@app.route('/')
def index():
    """Render the main page"""
    if analyzer.df is None:
        return render_template('index.html', cities=[], min_date='', max_date='', error="Data loading failed")
    cities = sorted(analyzer.df['city'].unique().tolist())
    min_date = analyzer.df['timestamp'].min().strftime('%Y-%m-%d')
    max_date = analyzer.df['timestamp'].max().strftime('%Y-%m-%d')
    return render_template('index.html', cities=cities, min_date=min_date, max_date=max_date)

@app.route('/api/clustering', methods=['POST'])
def clustering():
    """Perform clustering and return visualization data"""
    try:
        data = request.json
        k_opt = int(data.get('k_opt', 4))
        sample_size = int(data.get('sample_size', 10000))

        if k_opt < 2 or k_opt > 10:
            return jsonify({'error': 'Number of clusters must be between 2 and 10'}), 400
        if sample_size < 1000 or sample_size > 50000:
            return jsonify({'error': 'Sample size must be between 1000 and 50000'}), 400

        result = analyzer.clustering_analysis(sample_size=sample_size, k_opt=k_opt)
        plot_json = analyzer.plot_clusters()

        return jsonify({
            'elbow_plot': result['elbow_plot'],
            'scatter_plot': plot_json,
            'silhouette_scores': {
                'kmeans': result['kmeans_silhouette'],
                'dbscan': result['dbscan_silhouette'],
                'hierarchical': result['hierarchical_silhouette']
            },
            'interpretation': analyzer._interpret_clusters(analyzer.cluster_results, [
                'temperature_f', 'humidity_pct', 'wind_speed_mph',
                'cloud_cover_pct', 'pressure_mb', 'precip_intensity_inhr',
                'electricity_demand_mwh', 'hour'
            ])
        })
    except Exception as e:
        print(f"Clustering error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f"Clustering failed: {str(e)}"}), 500

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """Perform forecasting and return visualization data"""
    try:
        data = request.json
        city = data.get('city', 'nyc')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        lag_days = int(data.get('lag_days', 1))
        model_name = data.get('model_name', 'Random Forest')

        # Validate inputs
        if city not in analyzer.df['city'].unique():
            return jsonify({'error': f"Invalid city: {city}"}), 400
        if model_name not in ['Linear Regression', 'Random Forest', 'XGBoost', 'LSTM']:
            return jsonify({'error': f"Invalid model: {model_name}"}), 400
        if lag_days < 1 or lag_days > 7:
            return jsonify({'error': 'Look-back window must be between 1 and 7 days'}), 400

        # Calculate test_days from date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if start >= end:
            return jsonify({'error': 'End date must be after start date'}), 400
        test_days = (end - start).days + 1
        if test_days < 1 or test_days > 90:
            return jsonify({'error': 'Date range must be between 1 and 90 days'}), 400

        # Perform forecasting
        results = analyzer.forecast_demand(city=city, test_days=test_days, model_name=model_name, lag_days=lag_days)
        plot_json = analyzer.plot_forecast(city=city)

        return jsonify({
            'metrics': results.get(model_name, {'MAE': None, 'RMSE': None, 'MAPE': None, 'R2': None}),
            'plot': plot_json
        })
    except Exception as e:
        print(f"Forecasting error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)