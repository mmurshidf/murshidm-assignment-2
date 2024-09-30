let kMeansChart, dataPoints, clusterCenters, clusterLabels;
let iterationStep = 0;
let algorithmConverged = false;

function initializeChart() {
    const chartContext = document.getElementById('kMeansChartCanvas').getContext('2d');
    kMeansChart = new Chart(chartContext, {
        type: 'scatter',
        data: {
            datasets: _getInitialChartDatasets(),
        },
        options: _getChartDisplayOptions(),
    });
    console.log('Chart initialized');
}

function _getInitialChartDatasets() {
    return [{
        label: 'Data Points',
        data: [],
        backgroundColor: 'rgba(0, 0, 255, 0.5)'
    }, {
        label: 'Centroids',
        data: [],
        backgroundColor: 'rgba(255, 0, 0, 1)',
        pointRadius: 8
    }];
}

function _getChartDisplayOptions() {
    return {
        scales: {
            x: {
                type: 'linear',
                position: 'bottom',
                min: 0,
                max: 100
            },
            y: {
                min: 0,
                max: 100
            }
        }
    };
}

function refreshChart() {
    console.log('Updating chart');
    kMeansChart.data.datasets[0].data = _mapDataPoints(dataPoints, clusterLabels);
    kMeansChart.data.datasets[1].data = clusterCenters ? _mapCentroids(clusterCenters) : [];
    kMeansChart.update();
    console.log(`Chart updated with ${dataPoints.length} points and ${clusterCenters ? clusterCenters.length : 0} centroids`);
}

function _mapDataPoints(points, labels) {
    return points.map((point, index) => ({
        x: point[0],
        y: point[1],
        backgroundColor: labels ? _assignColor(labels[index]) : 'rgba(0, 0, 255, 0.5)'
    }));
}

function _mapCentroids(centers) {
    return centers.map(center => ({
        x: center[0],
        y: center[1]
    }));
}

function _assignColor(label) {
    const colorPalette = [
        'rgba(255, 0, 0, 0.5)', 'rgba(0, 255, 0, 0.5)',
        'rgba(0, 0, 255, 0.5)', 'rgba(255, 255, 0, 0.5)',
        'rgba(255, 0, 255, 0.5)', 'rgba(0, 255, 255, 0.5)'
    ];
    return colorPalette[label % colorPalette.length];
}

function updateControlButtonStates() {
    const selectedInitMethod = document.getElementById('initializationMethodSelect').value;
    const numberOfClusters = parseInt(document.getElementById('clusterCountInput').value);
    const stepButton = document.getElementById('stepButton');
    const convergeButton = document.getElementById('convergeButton');

    if (selectedInitMethod === 'manual') {
        const centroidsPlaced = clusterCenters.length === numberOfClusters;
        stepButton.disabled = !centroidsPlaced;
        convergeButton.disabled = !centroidsPlaced;
        _updateStatus(centroidsPlaced ? 'Ready to start clustering' : `Place ${numberOfClusters - clusterCenters.length} more centroid(s)`);
    } else {
        stepButton.disabled = false;
        convergeButton.disabled = false;
    }
}

function generateNewDataset() {
    console.log('Generating new data');
    fetch('/generate_data', { method: 'POST' })
        .then(response => response.json())
        .then(_handleNewDataset)
        .catch(error => console.error('Error generating data:', error));
}

function _handleNewDataset(newData) {
    dataPoints = newData;
    clusterCenters = null;
    clusterLabels = null;
    iterationStep = 0;
    algorithmConverged = false;
    console.log(`Generated ${dataPoints.length} data points`);
    refreshChart();
    _updateStatus('New data generated. Click Step to start clustering.');
}

function executeKMeansStep() {
    if (!_canExecuteStep()) return;

    const kMeansRequestPayload = _prepareKMeansRequest();
    fetch('/run_kmeans_step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(kMeansRequestPayload)
    })
    .then(response => response.json())
    .then(_processKMeansStepResponse)
    .catch(error => console.error('Error in KMeans step:', error));
}

function _canExecuteStep() {
    if (!dataPoints || algorithmConverged) return false;

    const selectedInitMethod = document.getElementById('initializationMethodSelect').value;
    const numberOfClusters = parseInt(document.getElementById('clusterCountInput').value);

    if (selectedInitMethod === 'manual' && clusterCenters.length !== numberOfClusters) {
        _updateStatus(`Place ${numberOfClusters - clusterCenters.length} more centroid(s) before starting`);
        return false;
    }
    return true;
}

function _prepareKMeansRequest() {
    const selectedInitMethod = document.getElementById('initializationMethodSelect').value;
    const numberOfClusters = parseInt(document.getElementById('clusterCountInput').value);
    
    return {
        data: dataPoints,
        k: numberOfClusters,
        initMethod: selectedInitMethod,
        step: iterationStep,
        initialCentroids: selectedInitMethod === 'manual' ? clusterCenters : null
    };
}

function _processKMeansStepResponse(result) {
    clusterCenters = result.centroids;
    clusterLabels = result.labels;
    iterationStep = result.step;
    algorithmConverged = result.converged;
    refreshChart();
    _updateStatus(algorithmConverged ? 'KMeans has converged!' : `Step ${iterationStep} completed`);
}

function executeKMeansConvergence() {
    if (!_canExecuteStep()) return;

    const kMeansRequestPayload = _prepareKMeansRequest();
    fetch('/run_kmeans_converge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(kMeansRequestPayload)
    })
    .then(response => response.json())
    .then(_processKMeansConvergenceResponse)
    .catch(error => console.error('Error in KMeans converge:', error));
}

function _processKMeansConvergenceResponse(result) {
    clusterCenters = result.centroids;
    clusterLabels = result.labels;
    iterationStep = result.step;
    algorithmConverged = true;
    refreshChart();
    _updateStatus('KMeans has converged!');
}

function resetKMeans() {
    clusterCenters = [];
    clusterLabels = null;
    iterationStep = 0;
    algorithmConverged = false;
    refreshChart();
    _updateStatus('Reset complete. Generate new data or start clustering.');
    updateControlButtonStates();
}

function _updateStatus(statusMessage) {
    document.getElementById('statusMessage').textContent = statusMessage;
}

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('generateDataButton').addEventListener('click', generateNewDataset);
    document.getElementById('stepButton').addEventListener('click', executeKMeansStep);
    document.getElementById('convergeButton').addEventListener('click', executeKMeansConvergence);
    document.getElementById('resetButton').addEventListener('click', resetKMeans);
    document.getElementById('initializationMethodSelect').addEventListener('change', updateControlButtonStates);
    document.getElementById('clusterCountInput').addEventListener('change', updateControlButtonStates);

    document.getElementById('kMeansChartCanvas').addEventListener('click', function(event) {
        if (document.getElementById('initializationMethodSelect').value === 'manual') {
            _placeCentroid(event);
        }
    });

    initializeChart();
    generateNewDataset();
    updateControlButtonStates();
});

function _placeCentroid(event) {
    const canvasRect = kMeansChart.canvas.getBoundingClientRect();
    const xCoord = kMeansChart.scales.x.getValueForPixel(event.clientX - canvasRect.left);
    const yCoord = kMeansChart.scales.y.getValueForPixel(event.clientY - canvasRect.top);
    const numberOfClusters = parseInt(document.getElementById('clusterCountInput').value);

    if (clusterCenters.length < numberOfClusters) {
        clusterCenters.push([xCoord, yCoord]);
        refreshChart();
        _updateStatus(`Centroid ${clusterCenters.length} of ${numberOfClusters} placed`);
        updateControlButtonStates();
    } else {
        _updateStatus(`All ${numberOfClusters} centroids have been placed`);
    }
}