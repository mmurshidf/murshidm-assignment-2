from flask import Flask, render_template, jsonify, request
import numpy as np

class BaseKMeans:
    def __init__(self, k=3, max_iters=300):
        self.k = k
        self.max_iters = max_iters
        self.state = {
            'centroids': None,
            'labels': None,
            'n_iter': 0
        }

    def step(self, X):
        if self.state['centroids'] is None:
            self.initialize(X)
        else:
            old_centroids = self.state['centroids'].copy()
            self.state['labels'] = self._assign_labels(X)
            self._update_centroids(X)
            self.state['n_iter'] += 1
            if np.all(old_centroids == self.state['centroids']):
                return False
        return True

    def _assign_labels(self, X):
        return np.array([np.argmin([np.linalg.norm(x - c) for c in self.state['centroids']]) for x in X])

    def _update_centroids(self, X):
        for i in range(self.k):
            if np.sum(self.state['labels'] == i) > 0:
                self.state['centroids'][i] = np.mean(X[self.state['labels'] == i], axis=0)

    def predict(self, X):
        return self._assign_labels(X)

class RandomKMeans(BaseKMeans):
    def initialize(self, X, manual_centroids=None):
        self.state['centroids'] = X[np.random.choice(X.shape[0], self.k, replace=False)]


class KMeansPlusPlus(BaseKMeans):
    def initialize(self, X, manual_centroids=None):
        centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, self.k):
            dists = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
            probs = dists**2 / np.sum(dists**2)
            centroids.append(X[np.random.choice(X.shape[0], p=probs)])
        self.state['centroids'] = np.array(centroids)

class FarthestFirstKMeans(BaseKMeans):
    def initialize(self, X, manual_centroids=None):
        centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, self.k):
            dists = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
            centroids.append(X[np.argmax(dists)])
        self.state['centroids'] = np.array(centroids)

class ManualKMeans(BaseKMeans):
    def initialize(self, X, manual_centroids=None):
        if manual_centroids is not None:
            self.state['centroids'] = np.array(manual_centroids)

app = Flask(__name__)

kmeans_instance = None
current_k = None
has_converged = False

def get_kmeans_instance(k, init_method):
    if init_method == 'random':
        return RandomKMeans(k=k)
    elif init_method == 'kmeans++':
        return KMeansPlusPlus(k=k)
    elif init_method == 'farthest_first':
        return FarthestFirstKMeans(k=k)
    elif init_method == 'manual':
        return ManualKMeans(k=k)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_data', methods=['POST'])
def generate_data():
    _reset_kmeans_state()

    data = _generate_random_data()

    print(f"Generated {len(data)} data points")
    return jsonify(data.tolist())

def _reset_kmeans_state():
    global kmeans_instance, current_k, has_converged
    kmeans_instance = None
    current_k = None
    has_converged = False

def _generate_random_data(num_points=100):
    return np.random.rand(num_points, 2) * 100

@app.route('/run_kmeans_step', methods=['POST'])
def run_kmeans_step():
    data, k, init_method, step, initial_centroids = _extract_step_params(request.json)

    if _should_initialize(kmeans_instance, k, step):
        _initialize_kmeans(data, k, init_method, initial_centroids)

    if not has_converged:
        _perform_kmeans_step(data)

    result = _prepare_step_result(step)
    print(f"Step result: {len(result['centroids'])} centroids, {len(result['labels'])} labels, Converged: {has_converged}")
    
    return jsonify(result)

def _extract_step_params(request_json):
    data = np.array(request_json['data'])
    k = int(request_json['k'])
    init_method = request_json['initMethod']
    step = request_json['step']
    initial_centroids = request_json.get('initialCentroids', None)
    return data, k, init_method, step, initial_centroids

def _should_initialize(kmeans_instance, k, step):
    return step == 0 or kmeans_instance is None or k != current_k

def _initialize_kmeans(data, k, init_method, initial_centroids):
    global kmeans_instance, current_k, has_converged

    kmeans_instance = get_kmeans_instance(k, init_method)
    current_k = k
    has_converged = False

    if init_method == 'manual' and initial_centroids is not None:
        kmeans_instance.state['centroids'] = np.array(initial_centroids)
    else:
        kmeans_instance.initialize(data)

def _perform_kmeans_step(data):
    global has_converged
    has_converged = not kmeans_instance.step(data)

def _prepare_step_result(step):
    return {
        'centroids': kmeans_instance.state['centroids'].tolist(),
        'labels': kmeans_instance.state['labels'].tolist(),
        'step': step + 1,
        'converged': has_converged
    }

@app.route('/run_kmeans_converge', methods=['POST'])
def run_kmeans_converge():
    global kmeans_instance, current_k, has_converged

    data, k, init_method, initial_centroids = _extract_kmeans_params(request.json)

    print(f"Running KMeans to convergence. Method: {init_method}, K: {k}")

    if _should_initialize_kmeans(kmeans_instance, k):
        _initialize_kmeans_instance(data, k, init_method, initial_centroids)

    result = _run_until_convergence(data)

    print(f"Converged after {result['step']} steps with {len(result['centroids'])} centroids.")
    return jsonify(result)

def _extract_kmeans_params(request_json):
    data = np.array(request_json['data'])
    k = int(request_json['k'])
    init_method = request_json['initMethod']
    initial_centroids = request_json.get('initialCentroids', None)
    return data, k, init_method, initial_centroids

def _should_initialize_kmeans(kmeans_instance, k):
    return kmeans_instance is None or k != current_k

def _initialize_kmeans_instance(data, k, init_method, initial_centroids):
    global kmeans_instance, current_k, has_converged

    kmeans_instance = get_kmeans_instance(k, init_method)
    current_k = k
    has_converged = False

    if init_method == 'manual' and initial_centroids:
        kmeans_instance.state['centroids'] = np.array(initial_centroids)
    else:
        kmeans_instance.initialize(data)

def _run_until_convergence(data):
    global has_converged
    step = 0

    while not has_converged:
        has_converged = not kmeans_instance.step(data)
        step += 1

    return {
        'centroids': kmeans_instance.state['centroids'].tolist(),
        'labels': kmeans_instance.state['labels'].tolist(),
        'step': step,
        'converged': has_converged
    }

if __name__ == '__main__':
    app.run(debug=True, port=3000)
