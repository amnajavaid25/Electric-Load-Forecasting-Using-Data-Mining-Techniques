import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go  # Added for Scatterpolar
import zipfile
import json
from io import TextIOWrapper
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
import warnings
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

class ElectricityDemandAnalyzer:
    def __init__(self, zip_path='archive.zip'):
        self.zip_path = zip_path
        self.df = None
        self.cleaned_df = None
        self.anomaly_report = None
        self.cluster_results = None
        self.forecast_results = None
        self.log_offset = None
        self.n_steps = None

    def load_and_merge_data(self):
        """Load and merge electricity and weather data from zip archive"""
        electricity_df = self._load_electricity_data()
        weather_df = self._load_weather_data()

        self.df = pd.merge(
            weather_df,
            electricity_df,
            on=['city', 'timestamp'],
            how='inner'
        )

        self._add_time_features()
        return self.df

    def _load_electricity_data(self):
        """Process electricity CSVs from zip archive"""
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            with zip_ref.open('cleaned_subregion_data.csv') as file:
                subregion_df = pd.read_csv(TextIOWrapper(file, 'utf-8'))
                if 'demand' in subregion_df.columns:
                    subregion_df['electricity_demand_mwh'] = subregion_df['demand']
                subregion_df['timestamp'] = pd.to_datetime(subregion_df['utc_time'])
                subregion_df = subregion_df[['timestamp', 'city', 'electricity_demand_mwh']]

            with zip_ref.open('cleaned_balance_data.csv') as file:
                balance_df = pd.read_csv(TextIOWrapper(file, 'utf-8'))
                if 'demand' in balance_df.columns:
                    balance_df['electricity_demand_mwh'] = balance_df['demand']
                balance_df['timestamp'] = pd.to_datetime(balance_df['utc_time'])
                balance_df = balance_df[['timestamp', 'city', 'electricity_demand_mwh']]

        return pd.concat([subregion_df, balance_df], ignore_index=True)

    def _load_weather_data(self):
        """Process weather JSON files from zip archive"""
        weather_dfs = []
        weather_files = [
            'phoenix.json', 'san_antonio.json', 'san_diego.json', 'san_jose.json',
            'seattle.json', 'dallas.json', 'houston.json', 'la.json', 'nyc.json',
            'philadelphia.json'
        ]

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            for weather_file in weather_files:
                try:
                    with zip_ref.open(weather_file) as file:
                        data = json.load(TextIOWrapper(file, 'utf-8'))

                        if 'hourly' in data and 'data' in data['hourly']:
                            df = pd.DataFrame(data['hourly']['data'])
                        else:
                            df = pd.DataFrame(data)

                        city = weather_file.split('.')[0]
                        df['city'] = city
                        df['timestamp'] = pd.to_datetime(df['time'], unit='s')

                        column_mapping = {
                            'temperature': 'temperature_f',
                            'dewPoint': 'dew_point_f',
                            'humidity': 'humidity_pct',
                            'windSpeed': 'wind_speed_mph',
                            'cloudCover': 'cloud_cover_pct',
                            'pressure': 'pressure_mb',
                            'precipIntensity': 'precip_intensity_inhr'
                        }

                        for original, new in column_mapping.items():
                            if original in df.columns and new not in df.columns:
                                df[new] = df[original]

                        keep_cols = ['timestamp', 'city'] + list(column_mapping.values())
                        df = df[[col for col in keep_cols if col in df.columns]]

                        weather_dfs.append(df)

                except Exception as e:
                    print(f"Warning: Error processing {weather_file}: {str(e)}")
                    continue

        return pd.concat(weather_dfs, ignore_index=True).drop_duplicates()

    def _add_time_features(self):
        """Add temporal features to the dataframe"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_merge_data() first.")

        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['day_of_month'] = self.df['timestamp'].dt.day
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)

        self.df['season'] = self.df['month'].apply(lambda x:
            'winter' if x in [12, 1, 2] else
            'spring' if x in [3, 4, 5] else
            'summer' if x in [6, 7, 8] else 'autumn')

        self.df['time_of_day'] = pd.cut(self.df['hour'],
                                      bins=[0, 6, 12, 18, 24],
                                      labels=['night', 'morning', 'afternoon', 'evening'],
                                      right=False)

    def preprocess_data(self):
        """Handle missing values and scale features"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_merge_data() first.")

        nan_cols = self.df.columns[self.df.isnull().any()].tolist()
        if nan_cols:
            print(f"Columns with NaN values before preprocessing: {nan_cols}")
            print("NaN counts per column:")
            print(self.df[nan_cols].isnull().sum())

        initial_count = len(self.df)
        self.df = self.df.dropna()
        print(f"Dropped {initial_count - len(self.df)} rows with missing values")

        if self.df.isnull().any().any():
            raise ValueError("Data still contains NaN values after preprocessing")

        self._scale_features()
        return self.df

    def _scale_features(self):
        """Normalize and scale features"""
        temp_scaler = StandardScaler()
        self.df[['temperature_f', 'dew_point_f']] = temp_scaler.fit_transform(
            self.df[['temperature_f', 'dew_point_f']])

        weather_scaler = MinMaxScaler()
        weather_features = ['humidity_pct', 'wind_speed_mph', 'pressure_mb',
                          'cloud_cover_pct', 'precip_intensity_inhr']
        for feat in weather_features:
            if feat in self.df.columns:
                self.df[feat] = weather_scaler.fit_transform(self.df[[feat]])

        if self.df['electricity_demand_mwh'].min() > 0:
            self.df['demand_log'] = np.log(self.df['electricity_demand_mwh'])

    def create_aggregates(self):
        """Create daily and weekly aggregated datasets"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_merge_data() first.")

        return self._create_daily_aggregates(), self._create_weekly_aggregates()

    def _create_daily_aggregates(self):
        daily = self.df.groupby(['city', pd.Grouper(key='timestamp', freq='D')]).agg({
            'temperature_f': ['mean', 'min', 'max'],
            'humidity_pct': 'mean',
            'wind_speed_mph': 'mean',
            'electricity_demand_mwh': ['sum', 'mean', 'max'],
            'day_of_week': 'first',
            'month': 'first',
            'season': 'first'
        })

        daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
        daily.reset_index(inplace=True)
        daily['demand_variation'] = daily['electricity_demand_mwh_max'] / daily['electricity_demand_mwh_mean']
        return daily

    def _create_weekly_aggregates(self):
        weekly = self.df.groupby(['city', pd.Grouper(key='timestamp', freq='W-MON')]).agg({
            'temperature_f': ['mean', 'min', 'max'],
            'humidity_pct': 'mean',
            'wind_speed_mph': 'mean',
            'electricity_demand_mwh': ['sum', 'mean', 'max', 'min'],
            'month': 'first',
            'season': 'first'
        })

        weekly.columns = ['_'.join(col).strip() for col in weekly.columns.values]
        weekly.reset_index(inplace=True)
        weekly['demand_range'] = weekly['electricity_demand_mwh_max'] - weekly['electricity_demand_mwh_min']
        weekly['temp_range'] = weekly['temperature_f_max'] - weekly['temperature_f_min']
        return weekly

    def detect_anomalies(self):
        """Detect anomalies using multiple methods"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_merge_data() first.")

        self._detect_statistical_anomalies()
        self._detect_ml_anomalies()
        self._detect_physical_anomalies()
        self._detect_temporal_anomalies()
        self._handle_anomalies()
        return self.anomaly_report

    def _detect_statistical_anomalies(self, threshold=3.5):
        numeric_cols = ['temperature_f', 'humidity_pct', 'wind_speed_mph',
                       'electricity_demand_mwh', 'pressure_mb']

        z_scores = np.abs(stats.zscore(self.df[numeric_cols]))
        self.df['z_score_anomaly'] = (z_scores > threshold).any(axis=1)

        anomalies = pd.DataFrame()
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            anomalies[col] = ~self.df[col].between(lower_bound, upper_bound)

        self.df['iqr_anomaly'] = anomalies.any(axis=1)

    def _detect_ml_anomalies(self, contamination=0.01):
        features = ['temperature_f', 'humidity_pct', 'wind_speed_mph',
                   'electricity_demand_mwh', 'hour', 'day_of_week']

        scaler = StandardScaler()
        X = scaler.fit_transform(self.df[features].fillna(self.df[features].median()))

        clf = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
        self.df['ml_anomaly'] = clf.fit_predict(X)
        self.df['ml_anomaly'] = self.df['ml_anomaly'].map({1: False, -1: True})

    def _detect_physical_anomalies(self):
        self.df['physical_anomaly'] = False

        self.df.loc[self.df['temperature_f'] > 140, 'physical_anomaly'] = True
        self.df.loc[self.df['temperature_f'] < -50, 'physical_anomaly'] = True
        self.df.loc[~self.df['humidity_pct'].between(0, 100), 'physical_anomaly'] = True
        self.df.loc[self.df['wind_speed_mph'] < 0, 'physical_anomaly'] = True

        demand_thresholds = {
            'nyc': (500, 15000),
            'la': (400, 12000),
            'phoenix': (300, 10000),
            'san_antonio': (200, 8000),
            'san_diego': (300, 9000),
            'san_jose': (200, 7000),
            'seattle': (300, 10000),
            'dallas': (400, 11000),
            'houston': (500, 13000),
            'philadelphia': (400, 12000)
        }

        for city, (low, high) in demand_thresholds.items():
            city_mask = self.df['city'] == city
            self.df.loc[city_mask & ~self.df['electricity_demand_mwh'].between(low, high), 'physical_anomaly'] = True

    def _detect_temporal_anomalies(self):
        self.df['temporal_anomaly'] = False

        for city in self.df['city'].unique():
            city_mask = self.df['city'] == city
            demand = self.df.loc[city_mask, 'electricity_demand_mwh']

            rolling_mean = demand.rolling(window=24, min_periods=1).mean()
            rolling_std = demand.rolling(window=24, min_periods=1).std()

            self.df.loc[city_mask & (np.abs(demand - rolling_mean) > 3 * rolling_std), 'temporal_anomaly'] = True

    def _handle_anomalies(self):
        self.df['total_anomalies'] = self.df[['z_score_anomaly', 'iqr_anomaly',
                                             'ml_anomaly', 'physical_anomaly',
                                             'temporal_anomaly']].any(axis=1)

        self.anomaly_report = {
            'total_records': len(self.df),
            'total_anomalies': self.df['total_anomalies'].sum(),
            'anomaly_rate': self.df['total_anomalies'].mean() * 100,
            'by_type': {
                'z_score': self.df['z_score_anomaly'].sum(),
                'iqr': self.df['iqr_anomaly'].sum(),
                'ml': self.df['ml_anomaly'].sum(),
                'physical': self.df['physical_anomaly'].sum(),
                'temporal': self.df['temporal_anomaly'].sum()
            },
            'by_city': self.df.groupby('city')['total_anomalies'].sum().to_dict()
        }

        self.df['action'] = 'keep'
        self.df.loc[self.df['physical_anomaly'], 'action'] = 'remove'
        self.df.loc[self.df['temporal_anomaly'], 'action'] = 'investigate'

        stat_anomalies = self.df['z_score_anomaly'] | self.df['iqr_anomaly']
        self.df.loc[stat_anomalies & ~self.df['physical_anomaly'], 'action'] = 'impute'

        self.cleaned_df = self.df[self.df['action'] != 'remove'].copy()

        for col in ['temperature_f', 'humidity_pct', 'wind_speed_mph', 'electricity_demand_mwh']:
            if col in self.cleaned_df.columns:
                city_medians = self.cleaned_df.groupby('city')[col].median()
                for city in city_medians.index:
                    mask = (self.cleaned_df['city'] == city) & (self.cleaned_df['action'] == 'impute')
                    self.cleaned_df.loc[mask, col] = city_medians[city]

    def visualize_anomalies(self, city='nyc', feature='electricity_demand_mwh', save_path=None):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_merge_data() first.")

        city_data = self.df[self.df['city'] == city].sort_values('timestamp')

        fig = px.line(city_data, x='timestamp', y=feature, title=f'Anomaly Detection for {feature} in {city}')
        anomaly_types = ['z_score_anomaly', 'iqr_anomaly', 'ml_anomaly', 'physical_anomaly', 'temporal_anomaly']
        colors = ['red', 'green', 'purple', 'orange', 'cyan']

        for atype, color in zip(anomaly_types, colors):
            anomalies = city_data[city_data[atype]]
            fig.add_scatter(x=anomalies['timestamp'], y=anomalies[feature], mode='markers',
                            marker=dict(color=color), name=atype)

        fig.update_layout(xaxis_title='Date', yaxis_title=feature, showlegend=True)
        if save_path:
            fig.write_image(save_path)
        return fig.to_json()

    def save_results(self, output_path='electricity_demand_analysis_results.csv'):
        if self.cleaned_df is None:
            raise ValueError("No cleaned data available. Run detect_anomalies() first.")

        self.cleaned_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    def clustering_analysis(self, sample_size=10000, k_opt=4):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_merge_data() first.")

        features = [
            'temperature_f', 'humidity_pct', 'wind_speed_mph',
            'cloud_cover_pct', 'pressure_mb', 'precip_intensity_inhr',
            'electricity_demand_mwh', 'hour'
        ]

        print("NaN values before sampling:")
        print(self.df[features].isnull().sum())

        df_clean = self.df.dropna(subset=features)
        print(f"Original: {len(self.df)}, Clean: {len(df_clean)}")

        if len(df_clean) < sample_size:
            print(f"Warning: Sample size reduced to {len(df_clean)} due to available data")
            sample_size = len(df_clean)

        df_sample = df_clean.sample(n=sample_size, random_state=42)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_sample[features])

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        df_sample['pca_x'] = X_pca[:, 0]
        df_sample['pca_y'] = X_pca[:, 1]

        inertias = []
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        fig_elbow = px.line(x=range(2, 10), y=inertias, markers=True,
                            title="Elbow Method for K",
                            labels={'x': 'k', 'y': 'Inertia'})
        print("Elbow plot generated")

        kmeans = KMeans(n_clusters=k_opt, random_state=42)
        labels_kmeans = kmeans.fit_predict(X_scaled)
        df_sample['cluster'] = labels_kmeans

        silhouette_kmeans = silhouette_score(X_scaled, labels_kmeans)
        print(f"K-Means Silhouette Score: {silhouette_kmeans:.2f}")

        dbscan = DBSCAN(eps=1.5, min_samples=10)
        labels_db = dbscan.fit_predict(X_scaled)

        mask = labels_db != -1
        if len(set(labels_db[mask])) > 1:
            score_db = silhouette_score(X_scaled[mask], labels_db[mask])
            print(f"DBSCAN Silhouette Score (excluding noise): {score_db:.2f}")
        else:
            print("DBSCAN did not form valid clusters.")
            score_db = None

        linkage_matrix = linkage(X_scaled, method='ward')
        h_opt = 3
        agglomerative = AgglomerativeClustering(n_clusters=h_opt)
        labels_h = agglomerative.fit_predict(X_scaled)

        silhouette_h = silhouette_score(X_scaled, labels_h)
        print(f"Hierarchical Clustering Silhouette Score: {silhouette_h:.2f}")

        self.cluster_results = df_sample
        self._interpret_clusters(df_sample, features)

        return {
            'elbow_plot': fig_elbow.to_json(),
            'kmeans_silhouette': silhouette_kmeans,
            'dbscan_silhouette': score_db,
            'hierarchical_silhouette': silhouette_h
        }

    def _interpret_clusters(self, clustered_df, features):
        print("\n=== Cluster Interpretation ===")

        cluster_means = clustered_df.groupby('cluster')[features].mean()
        cluster_sizes = clustered_df['cluster'].value_counts().sort_index()

        interpretation = []
        for cluster_id in cluster_means.index:
            cluster_data = cluster_means.loc[cluster_id]
            size = cluster_sizes[cluster_id]
            size_pct = (size / len(clustered_df)) * 100

            temp = cluster_data['temperature_f']
            temp_desc = "very hot" if temp > 1 else "hot" if temp > 0.5 else "moderate" if temp > -0.5 else "cool" if temp > -1 else "cold"
            demand = cluster_data['electricity_demand_mwh']
            demand_desc = "very high" if demand > 1 else "high" if demand > 0.5 else "moderate" if demand > -0.5 else "low" if demand > -1 else "very low"
            hour = cluster_data['hour']
            time_desc = "morning" if 5 <= hour < 12 else "afternoon" if 12 <= hour < 17 else "evening" if 17 <= hour < 21 else "night"
            humidity = cluster_data['humidity_pct']
            humid_desc = "humid" if humidity > 0.5 else "moderate humidity" if humidity > -0.5 else "dry"
            wind = cluster_data['wind_speed_mph']
            wind_desc = "windy" if wind > 0.5 else "moderate wind" if wind > -0.5 else "calm"

            label = f"{demand_desc}-demand {temp_desc} {time_desc}s"
            top_cities = clustered_df[clustered_df['cluster'] == cluster_id]['city'].value_counts().head(3).to_dict()

            interpretation.append({
                'cluster_id': int(cluster_id),
                'size': int(size),
                'size_pct': float(size_pct),
                'temperature': {'description': temp_desc, 'value': float(temp)},
                'demand': {'description': demand_desc, 'value': float(demand)},
                'time': {'description': time_desc, 'hour': int(hour)},
                'weather': {'humidity': humid_desc, 'wind': wind_desc},
                'summary_label': label,
                'top_cities': top_cities
            })

        self._plot_cluster_characteristics(cluster_means)
        return interpretation

    def _plot_cluster_characteristics(self, cluster_means):
        features_to_plot = [
            'temperature_f', 'humidity_pct', 'wind_speed_mph',
            'electricity_demand_mwh', 'hour'
        ]

        angles = np.linspace(0, 2 * np.pi, len(features_to_plot), endpoint=False).tolist()
        angles += angles[:1]

        data = []
        for cluster_id in cluster_means.index:
            values = cluster_means.loc[cluster_id, features_to_plot].tolist()
            values += values[:1]
            data.append(go.Scatterpolar(
                r=values,
                theta=features_to_plot + [features_to_plot[0]],
                fill='toself',
                name=f'Cluster {cluster_id}',
                opacity=0.1
            ))

        fig = go.Figure(data=data)
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title='Cluster Characteristics Comparison'
        )
        return fig.to_json()

    def prepare_forecasting_data(self, lag_days=1):
        if self.cleaned_df is None:
            raise ValueError("No cleaned data available. Run detect_anomalies() first.")

        df = self.cleaned_df.copy()

        for i in range(1, lag_days + 1):
            df[f'demand_lag_{i}'] = df.groupby('city')['electricity_demand_mwh'].shift(i * 24)

        df = df.dropna()

        demand_min = df['electricity_demand_mwh'].min()
        if demand_min <= 0:
            self.log_offset = 1 - demand_min
            df['demand_log'] = np.log(df['electricity_demand_mwh'] + self.log_offset)
        else:
            self.log_offset = 0
            df['demand_log'] = np.log(df['electricity_demand_mwh'])

        features = [
            'temperature_f', 'humidity_pct', 'wind_speed_mph',
            'hour', 'day_of_week', 'month', 'is_weekend'
        ] + [f'demand_lag_{i}' for i in range(1, lag_days + 1)]

        target = 'demand_log'

        return df, features, target

    def train_test_split(self, df, test_days=30):
        df = df.sort_values('timestamp')
        split_date = df['timestamp'].max() - pd.Timedelta(days=test_days)

        train = df[df['timestamp'] <= split_date]
        test = df[df['timestamp'] > split_date]

        return train, test

    def evaluate_model(self, y_true, y_pred):
        if hasattr(self, 'log_offset') and self.log_offset is not None and self.log_offset > 0:
            y_true = np.exp(y_true) - self.log_offset
            y_pred = np.exp(y_pred) - self.log_offset
        else:
            y_true = np.exp(y_true)
            y_pred = np.exp(y_pred)

        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'R2': r2_score(y_true, y_pred)
        }
        return metrics

    def train_linear_regression(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def train_random_forest(self, X_train, y_train):
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    def train_xgboost(self, X_train, y_train):
        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            reg_alpha=1,
            reg_lambda=1,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def train_lstm(self, X_train, y_train, n_steps=24):
        """Fixed LSTM implementation with proper reshaping"""
        X_np = np.nan_to_num(X_train.values)
        y_np = y_train.values.reshape(-1, 1)

        n_samples = (X_np.shape[0] // n_steps) * n_steps
        X_trimmed = X_np[:n_samples]
        y_trimmed = y_np[:n_samples]

        X_reshaped = X_trimmed.reshape(-1, n_steps, X_train.shape[1])
        y_reshaped = y_trimmed.reshape(-1, n_steps, 1)

        model = Sequential([
            LSTM(64, activation='relu', input_shape=(n_steps, X_train.shape[1])),
            Dense(1)
        ])
        model.compile(optimizer=Adam(0.001), loss='mse')

        model.fit(
            X_reshaped, y_reshaped,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=1,
            callbacks=[EarlyStopping(patience=3)]
        )

        self.n_steps = n_steps
        return model

    def plot_clusters(self, city=None):
        if not hasattr(self, 'cluster_results'):
            self.clustering_analysis()

        df = self.cluster_results
        if city:
            df = df[df['city'] == city]

        fig = px.scatter(
            df,
            x='pca_x',
            y='pca_y',
            color='cluster',
            hover_data=['temperature_f', 'electricity_demand_mwh', 'hour'],
            title=f"Demand Patterns {'('+city+')' if city else ''}"
        )
        fig.update_layout(
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2"
        )
        return fig.to_json()

    def plot_forecast(self, city='nyc'):
        if not hasattr(self, 'forecast_results'):
            raise ValueError("Run forecast_demand() first")

        df = pd.DataFrame({
            'timestamp': self.forecast_results['timestamps'],
            'actual': self.forecast_results['actual'],
            'predicted': self.forecast_results['predicted']
        })

        fig = px.line(
            df,
            x='timestamp',
            y=['actual', 'predicted'],
            title=f"Electricity Demand Forecast - {city}",
            labels={'value': 'Demand (MWh)', 'timestamp': 'Date'}
        )

        fig.update_layout(
            hovermode="x unified",
            legend_title_text='Series'
        )
        return fig.to_json()

    def forecast_demand(self, city='nyc', test_days=30, model_name='Random Forest', lag_days=1):
        """Robust forecasting with model selection"""
        try:
            if city not in self.df['city'].unique():
                raise ValueError(f"Invalid city: {city}")

            df, features, target = self.prepare_forecasting_data(lag_days=lag_days)
            df_city = df[df['city'] == city]

            if len(df_city) < test_days * 24:
                raise ValueError(f"Not enough data for {city} for {test_days} days")

            train, test = self.train_test_split(df_city, test_days)
            X_train, y_train = train[features], train[target]
            X_test, y_test = test[features], test[target]

            model_funcs = {
                'Linear Regression': self.train_linear_regression,
                'Random Forest': self.train_random_forest,
                'XGBoost': self.train_xgboost,
                'LSTM': self.train_lstm
            }

            if model_name not in model_funcs:
                raise ValueError(f"Invalid model: {model_name}")

            if model_name == 'LSTM':
                model = model_funcs[model_name](X_train, y_train, n_steps=24)
                n_steps = self.n_steps
                pad_size = (n_steps - (len(X_test) % n_steps)) % n_steps
                X_padded = np.pad(X_test.values, ((0, pad_size), (0, 0)), mode='edge')
                X_reshaped = X_padded.reshape(-1, n_steps, X_test.shape[1])
                y_pred = model.predict(X_reshaped).flatten()[:len(X_test)]
            else:
                model = model_funcs[model_name](X_train, y_train)
                y_pred = model.predict(X_test)

            metrics = self.evaluate_model(y_test, y_pred)

            self.forecast_results = {
                'timestamps': test['timestamp'],
                'actual': y_test,
                'predicted': y_pred
            }

            return {model_name: metrics}

        except Exception as e:
            print(f"Forecasting failed: {str(e)}")
            return {model_name: {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'R2': np.nan}}