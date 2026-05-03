# nn/inference.py

import os
import numpy as np
import torch
import joblib

from nn.model import ResponseTimeNN


# -----------------------------
# PATHS
# -----------------------------
MODEL_PATH = "nn/model.pt"
SCALER_PATH = "nn/scaler.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# PREDICTOR CLASS
# -----------------------------
class Predictor:
    """
    Loads trained model + scaler
    Provides prediction for backend
    """

    def __init__(self, input_dim=5):
        self.input_dim = input_dim

        # -----------------------------
        # Load model
        # -----------------------------
        self.model = ResponseTimeNN(input_dim=input_dim).to(DEVICE)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Train first.")

        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()

        # -----------------------------
        # Load feature scaler
        # -----------------------------
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("Scaler not found. Train first.")

        self.scaler = joblib.load(SCALER_PATH)

    # -----------------------------
    # Predict function
    # -----------------------------
    def predict(self, state_matrix):
        """
        Input:
            state_matrix → (num_servers, 5)

        Output:
            predicted_rt → (num_servers,)
        """

        state_matrix = np.array(state_matrix, dtype=np.float32)

        if state_matrix.ndim != 2 or state_matrix.shape[1] != self.input_dim:
            raise ValueError("Invalid input shape. Expected (num_servers, 5)")

        # 🔥 Clamp input to training distribution
        state_matrix[:, 0] = np.clip(state_matrix[:, 0], 0, 100)     # CPU
        state_matrix[:, 1] = np.clip(state_matrix[:, 1], 0, 8000)    # Memory
        state_matrix[:, 2] = np.clip(state_matrix[:, 2], 0, 1000)    # Bandwidth
        state_matrix[:, 3] = np.clip(state_matrix[:, 3], 0, 5000)    # Waiting time
        state_matrix[:, 4] = np.clip(state_matrix[:, 4], 0, 5000)    # Users

        # -----------------------------
        # Scale input
        # -----------------------------
        state_scaled = self.scaler.transform(state_matrix)

        x = torch.tensor(state_scaled, dtype=torch.float32).to(DEVICE)

        # -----------------------------
        # Predict
        # -----------------------------
        with torch.no_grad():
            preds = self.model(x).cpu().numpy().flatten()

        # 🔥 DEBUG
        print("RAW:", preds[:4])

        # 🔥 IMPORTANT: NO inverse scaling anymore
        preds_real = preds

        # 🔥 Clamp to realistic RT range
        preds_real = np.clip(preds_real, 50, 5000)

        print("FINAL:", preds_real[:4])

        return preds_real