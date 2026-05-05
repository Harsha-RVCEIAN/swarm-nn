# nn/inference.py

import os
import json
import numpy as np
import torch
import joblib

from nn.model import ResponseTimeNN

MODEL_PATH  = "nn/model.pt"
SCALER_PATH = "nn/scaler.pkl"
SCALE_PATH  = "nn/rt_scale.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:

    def __init__(self, input_dim=5):
        self.input_dim = input_dim

        self.model = ResponseTimeNN(input_dim=input_dim).to(DEVICE)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Train first.")
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()

        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("Scaler not found. Train first.")
        self.scaler = joblib.load(SCALER_PATH)

        if os.path.exists(SCALE_PATH):
            scale = json.load(open(SCALE_PATH))
            self.rt_mean = scale['mean']
            self.rt_std  = scale['std']
        else:
            self.rt_mean, self.rt_std = 0.0, 1.0

    def predict(self, state_matrix):
        state_matrix = np.array(state_matrix, dtype=np.float32)

        if state_matrix.ndim != 2 or state_matrix.shape[1] != self.input_dim:
            raise ValueError("Invalid input shape. Expected (num_servers, 5)")

        state_matrix[:, 0] = np.clip(state_matrix[:, 0], 0, 100)
        state_matrix[:, 1] = np.clip(state_matrix[:, 1], 0, 8000)
        state_matrix[:, 2] = np.clip(state_matrix[:, 2], 0, 1000)
        state_matrix[:, 3] = np.clip(state_matrix[:, 3], 0, 5000)
        state_matrix[:, 4] = np.clip(state_matrix[:, 4], 0, 5000)

        state_scaled = self.scaler.transform(state_matrix)
        x = torch.tensor(state_scaled, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            preds = self.model(x).cpu().numpy().flatten()

        # Undo target normalisation
        preds_real = preds * self.rt_std + self.rt_mean
        preds_real = np.clip(preds_real, 50, None) 

        print("RAW:", preds[:4])
        print("FINAL:", preds_real[:4])

        return preds_real