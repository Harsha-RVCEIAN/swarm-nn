# backend/controller.py

import numpy as np

from nn.inference import Predictor
from aco.aco import ACO
from aco.params import NUM_SERVERS, RETRAIN_INTERVAL
from data.buffer import Buffer

from backend.utils import validate_state
from backend.logger import (
    log_state,
    log_prediction,
    log_probabilities,
    log_selection,
    log_actual_rt,
    log_pheromone,
    log_retraining
)

predictor = Predictor(input_dim=5)
buffer = Buffer()

# No ACO instance here — api.py's BackendSimulation owns the single ACO.
# Both functions accept it as a parameter so there is only ever one
# pheromone state in the whole process.


def process_request(state_matrix, aco: ACO):
    state_matrix = validate_state(state_matrix)
    log_state(state_matrix)

    state_matrix_nn = state_matrix.copy()
    queue_norm = state_matrix_nn[:, 4] / (state_matrix_nn[:, 4] + 5.0)
    wait_norm  = state_matrix_nn[:, 3] / 500.0
    state_matrix_nn[:, 3] = wait_norm
    state_matrix_nn[:, 4] = queue_norm

    predicted_rt = predictor.predict(state_matrix_nn)
    print("[DEBUG] Predicted RT:", predicted_rt)
    predicted_rt = np.array(predicted_rt, dtype=float)
    predicted_rt = np.clip(predicted_rt, 50.0, None)

    # Force differentiation if NN outputs are too similar
    if np.std(predicted_rt) < 50.0:
        predicted_rt += np.random.normal(0, 200.0, size=predicted_rt.shape)
        predicted_rt = np.clip(predicted_rt, 50.0, None)

    normalized_rt = predicted_rt.copy()
    log_prediction(predicted_rt)

    cpu_util      = state_matrix[:, 0]
    wait_time     = state_matrix[:, 3]
    queue_lengths = state_matrix[:, 4]

    probabilities = aco.compute_probabilities(normalized_rt, cpu_util, wait_time, queue_lengths)
    probabilities = probabilities / (np.sum(probabilities) + 1e-12)
    print("[DEBUG] Probabilities:", probabilities)
    log_probabilities(probabilities)

    selected_server = aco.select_server(probabilities)
    log_selection(selected_server)

    return selected_server, predicted_rt, probabilities


def process_feedback(state_matrix, selected_server, actual_rt, aco: ACO):
    state_matrix = validate_state(state_matrix)
    actual_rt = float(np.clip(actual_rt, 50.0, 2500.0))  # ← add this line
    log_actual_rt(actual_rt)

    aco.update(selected_server, actual_rt)
    log_pheromone(aco.get_pheromone())

    buffer.add(state_matrix[selected_server], actual_rt)

    if buffer.size() >= RETRAIN_INTERVAL and buffer.size() % RETRAIN_INTERVAL == 0:
        retrain_model()


def retrain_model():
    log_retraining()
    try:
        data = buffer.get_recent_data()
        if len(data) < 10:
            return
        X = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        y = np.clip(np.nan_to_num(y, nan=500.0), 50.0, 5000.0)
        from nn.train import train_incremental
        train_incremental(X, y)
    except Exception as e:
        import logging
        logging.getLogger("NN-ACO").error("Retrain skipped: %s", e, exc_info=True)