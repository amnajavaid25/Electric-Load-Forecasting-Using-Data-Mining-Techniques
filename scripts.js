// static/js/scripts.js
document.addEventListener('DOMContentLoaded', () => {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toast_message');
    const clusteringResults = document.getElementById('clustering_results');
    const forecastResults = document.getElementById('forecast_results');
    const clusteringLoader = document.getElementById('clustering_loader');
    const forecastLoader = document.getElementById('forecast_loader');

    // Show toast notification
    function showToast(message, isError = true) {
        console.log(isError ? 'Error:' : 'Success:', message);
        toastMessage.textContent = message;
        toast.classList.remove('hidden', 'bg-green-600');
        toast.classList.add(isError ? 'bg-red-600' : 'bg-green-600', 'show');
        setTimeout(() => {
            toast.classList.remove('show');
            toast.classList.add('hidden');
        }, 5000);
    }

    // Run clustering
    document.getElementById('run_clustering').addEventListener('click', async () => {
        const k_opt = document.getElementById('k_opt').value;
        clusteringLoader.classList.remove('hidden');

        try {
            const response = await fetch('/api/clustering', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ k_opt: k_opt, sample_size: 10000 })
            });

            const result = await response.json();
            clusteringLoader.classList.add('hidden');

            if (response.ok) {
                console.log('Clustering response:', result);
                // Render plots
                try {
                    Plotly.newPlot('elbow_plot', JSON.parse(result.elbow_plot));
                    Plotly.newPlot('cluster_plot', JSON.parse(result.scatter_plot));
                } catch (plotErr) {
                    console.error('Plotly rendering error:', plotErr);
                    showToast('Failed to render plots');
                    return;
                }

                // Display silhouette scores
                const scoresDiv = document.getElementById('silhouette_scores');
                scoresDiv.innerHTML = `
                    <h3 class="text-lg font-semibold">Silhouette Scores</h3>
                    <p>K-Means: ${result.silhouette_scores.kmeans.toFixed(2)}</p>
                    <p>DBSCAN: ${result.silhouette_scores.dbscan ? result.silhouette_scores.dbscan.toFixed(2) : 'N/A'}</p>
                    <p>Hierarchical: ${result.silhouette_scores.hierarchical.toFixed(2)}</p>
                `;

                // Display cluster interpretation
                const interpDiv = document.getElementById('cluster_interpretation');
                interpDiv.innerHTML = '<h3 class="text-lg font-semibold">Cluster Interpretation</h3>';
                result.interpretation.forEach(cluster => {
                    interpDiv.innerHTML += `
                        <div class="mt-2">
                            <h4>Cluster ${cluster.cluster_id} (n=${cluster.size}, ${cluster.size_pct.toFixed(1)}%)</h4>
                            <p>Temperature: ${cluster.temperature.description} (${cluster.temperature.value.toFixed(2)} std)</p>
                            <p>Demand: ${cluster.demand.description} (${cluster.demand.value.toFixed(2)} std)</p>
                            <p>Time: ${cluster.time.description} (~${cluster.time.hour}:00)</p>
                            <p>Weather: ${cluster.weather.humidity}, ${cluster.weather.wind}</p>
                            <p>Summary: ${cluster.summary_label}</p>
                            <p>Top Cities: ${Object.entries(cluster.top_cities).map(([city, count]) => `${city}: ${count}`).join(', ')}</p>
                        </div>
                    `;
                });

                clusteringResults.classList.remove('hidden');
                showToast('Clustering completed successfully', false);
            } else {
                showToast(result.error);
            }
        } catch (err) {
            clusteringLoader.classList.add('hidden');
            showToast('Failed to run clustering: ' + err.message);
        }
    });

    // Run forecast
    document.getElementById('run_forecast').addEventListener('click', async () => {
        const city = document.getElementById('city').value;
        const start_date = document.getElementById('start_date').value;
        const end_date = document.getElementById('end_date').value;
        const lag_days = document.getElementById('lag_days').value;
        forecastLoader.classList.remove('hidden');

        // Get selected models
        const modelCheckboxes = [
            'linear_regression', 'random_forest', 'xgboost', 'lstm'
        ].map(id => document.getElementById(id));
        const selectedModels = modelCheckboxes
            .filter(cb => cb.checked)
            .map(cb => cb.value);

        if (selectedModels.length === 0) {
            forecastLoader.classList.add('hidden');
            showToast('Please select at least one forecasting model');
            return;
        }

        forecastResults.classList.add('hidden');
        const metricsDiv = document.getElementById('forecast_metrics');
        metricsDiv.innerHTML = '';

        for (const model_name of selectedModels) {
            try {
                const response = await fetch('/api/forecast', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        city,
                        start_date,
                        end_date,
                        lag_days,
                        model_name
                    })
                });

                const result = await response.json();
                if (response.ok) {
                    console.log(`${model_name} forecast response:`, result);
                    // Render forecast plot
                    try {
                        Plotly.newPlot('forecast_plot', JSON.parse(result.plot));
                    } catch (plotErr) {
                        console.error('Plotly rendering error:', plotErr);
                        showToast('Failed to render forecast plot');
                        continue;
                    }

                    // Display metrics
                    metricsDiv.innerHTML += `
                        <h3 class="text-lg font-semibold">${model_name} Metrics</h3>
                        <p>MAE: ${result.metrics.MAE ? result.metrics.MAE.toFixed(2) : 'N/A'}</p>
                        <p>RMSE: ${result.metrics.RMSE ? result.metrics.RMSE.toFixed(2) : 'N/A'}</p>
                        <p>MAPE: ${result.metrics.MAPE ? result.metrics.MAPE.toFixed(2) : 'N/A'}%</p>
                        <p>R2: ${result.metrics.R2 ? result.metrics.R2.toFixed(2) : 'N/A'}</p>
                    `;

                    forecastResults.classList.remove('hidden');
                } else {
                    showToast(`${model_name}: ${result.error}`);
                }
            } catch (err) {
                showToast(`${model_name}: Failed to run forecast: ${err.message}`);
            }
        }
        forecastLoader.classList.add('hidden');
        if (forecastResults.classList.contains('hidden')) {
            showToast('No forecast results to display');
        } else {
            showToast('Forecast completed successfully', false);
        }
    });
});