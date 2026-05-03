# backend/utils.py

import numpy as np
import joblib
import os


# -----------------------------
# Load scaler (kept for compatibility)
# -----------------------------
SCALER_PATH = "nn/scaler.pkl"

if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    scaler = None


# -----------------------------
# Validate state matrix
# -----------------------------
def validate_state(state_matrix):
    """
    Ensures state is valid:
    - 2D structure
    - correct number of features (5)
    """

    state_matrix = np.array(state_matrix, dtype=float)

    if state_matrix.ndim != 2:
        raise ValueError("State must be 2D: (num_servers, num_features)")

    if state_matrix.shape[1] != 5:
        raise ValueError("Each server must have 5 features")

    return state_matrix


# -----------------------------
# Preprocess input for NN
# -----------------------------
def preprocess_state(state_matrix):
    """
    🔥 FIX: DO NOT SCALE HERE
    Scaling must happen ONLY inside nn/inference.py

    This prevents:
    ❌ double scaling
    ❌ distorted inputs
    ❌ model collapse
    """

    state_matrix = validate_state(state_matrix)

    # Return raw state (scaling handled later)
    return state_matrix


# -----------------------------
# Safe division (avoid zero issues)
# -----------------------------
def safe_inverse(values, epsilon=1e-6):
    """
    Computes 1/x safely
    """
    values = np.clip(values, epsilon, None)
    return 1.0 / values


# -----------------------------
# Normalize probabilities
# -----------------------------
def normalize(probabilities):
    """
    Ensures probabilities sum to 1
    """

    probabilities = np.array(probabilities, dtype=float)

    total = np.sum(probabilities)

    if total == 0:
        return np.ones_like(probabilities) / len(probabilities)

    return probabilities / total


# -----------------------------
# Convert server objects → matrix
# -----------------------------
def servers_to_matrix(servers):
    """
    Converts list of server dicts/objects to matrix
    """

    matrix = []

    for s in servers:
        matrix.append([
            s["CPU_utilization"],
            s["Memory_utilization"],
            s["Bandwidth_utilization"],
            s["Queue_pressure"],
            s["Active_users"]
        ])

    return np.array(matrix, dtype=float)


# -----------------------------
# Convert numpy to list (for API response)
# -----------------------------
def to_list(arr):
    """
    Converts numpy array → Python list
    """
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr