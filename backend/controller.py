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


# -----------------------------
# Initialize core components
# -----------------------------
predictor = Predictor(input_dim=5)
aco = ACO(num_servers=NUM_SERVERS)
buffer = Buffer()


# -----------------------------
# MAIN DECISION FUNCTION
# -----------------------------
def process_request(state_matrix):

    state_matrix = validate_state(state_matrix)
    log_state(state_matrix)

    # -----------------------------
    # Step 1: NN Feature Engineering & Prediction
    # -----------------------------
    state_matrix_nn = state_matrix.copy()
    queue_norm = state_matrix_nn[:, 4] / (state_matrix_nn[:, 4] + 5.0)
    wait_norm = state_matrix_nn[:, 3] / 500.0
    
    state_matrix_nn[:, 3] = wait_norm
    state_matrix_nn[:, 4] = queue_norm
    
    predicted_rt = predictor.predict(state_matrix_nn)
    print("[DEBUG] Predicted RT:", predicted_rt)
    predicted_rt = np.array(predicted_rt, dtype=float)

    # FIX: Pass raw predicted RT (in milliseconds) directly to ACO
    # Just ensure it's strictly positive
    predicted_rt = np.clip(predicted_rt, 1.0, None)
    
    # 🔥 FIX: Prevent NN flat-output collapse (force differentiation)
    if np.std(predicted_rt) < 1.0:
        predicted_rt += np.random.normal(0, 5.0, size=predicted_rt.shape)
        predicted_rt = np.clip(predicted_rt, 1.0, None)
        
    normalized_rt = predicted_rt.copy()

    log_prediction(predicted_rt)

    # -----------------------------
    # Step 2: ACO Probabilities
    # -----------------------------
    # Extract features for heuristic penalties
    cpu_util = state_matrix[:, 0]
    wait_time = state_matrix[:, 3]
    queue_lengths = state_matrix[:, 4]

    probabilities = aco.compute_probabilities(normalized_rt, cpu_util, wait_time, queue_lengths)

    # 🔥 FIX 4: safety normalization
    probabilities = probabilities / (np.sum(probabilities) + 1e-12)
    print("[DEBUG] Probabilities:", probabilities)
    log_probabilities(probabilities)

    # -----------------------------
    # Step 3: Select Server
    # -----------------------------
    selected_server = aco.select_server(probabilities)
    log_selection(selected_server)

    return selected_server, predicted_rt, probabilities


# -----------------------------
# FEEDBACK LOOP
# -----------------------------
def process_feedback(state_matrix, selected_server, actual_rt):

    state_matrix = validate_state(state_matrix)
    log_actual_rt(actual_rt)

    # -----------------------------
    # Step 1: Update ACO
    # -----------------------------
    aco.update(selected_server, actual_rt)
    log_pheromone(aco.get_pheromone())

    # -----------------------------
    # Step 2: Store experience
    # -----------------------------
    buffer.add(state_matrix[selected_server], actual_rt)

    # -----------------------------
    # Step 3: Retrain trigger
    # -----------------------------
    if buffer.size() >= RETRAIN_INTERVAL and buffer.size() % RETRAIN_INTERVAL == 0:
        retrain_model()


# -----------------------------
# RETRAINING FUNCTION
# -----------------------------
def retrain_model():

    log_retraining()

    data = buffer.get_recent_data()

    if len(data) == 0:
        return

    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])

    from nn.train import train_incremental
    train_incremental(X, y)