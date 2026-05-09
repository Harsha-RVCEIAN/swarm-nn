# nn/train.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import joblib
import json

from nn.model import ResponseTimeNN


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH   = "data/processed/clean_dataset.csv"
MODEL_PATH  = "nn/model.pt"
SCALER_PATH = "nn/scaler.pkl"
SCALE_PATH  = "nn/rt_scale.json"

BATCH_SIZE = 32
EPOCHS     = 100
LR         = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# LOAD DATA
# -----------------------------
def load_dataset():
    df = pd.read_csv(DATA_PATH)

    df = df.rename(columns={
        "CPU_Utilization (%)":                  "CPU_utilization",
        "Memory_Consumption (MB)":              "Memory_utilization",
        "Network_Bandwidth_Utilization (Mbps)": "Bandwidth_utilization",
        "Task_Waiting_Time (ms)":               "Queue_pressure",
        "Number_of_Active_Users":               "Active_users",
        "response_time(ms)":                    "Response_Time"
    })

    features = [
        "CPU_utilization",
        "Memory_utilization",
        "Bandwidth_utilization",
        "Queue_pressure",
        "Active_users"
    ]
    target = "Response_Time"

    missing = [col for col in features + [target] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[features].astype(np.float32).values
    y = df[target].astype(np.float32).values.reshape(-1, 1)

    y_mean = float(y.mean())
    y_std  = float(y.std())
    y = (y - y_mean) / y_std

    return X, y, y_mean, y_std


# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(X, fit=True):
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)

    return X_scaled


# -----------------------------
# LOAD SCALE  ← new helper
# -----------------------------
def load_scale():
    """
    Load rt_scale.json. If it doesn't exist, create it with
    safe defaults derived from the typical RT range (50–2500 ms).
    This prevents train_incremental from crashing on a fresh install.
    """
    if os.path.exists(SCALE_PATH):
        with open(SCALE_PATH, 'r') as f:
            scale = json.load(f)
        return scale['mean'], scale['std']

    # File missing — use defaults and persist them so next call succeeds
    y_mean, y_std = 500.0, 300.0
    os.makedirs('nn', exist_ok=True)
    with open(SCALE_PATH, 'w') as f:
        json.dump({'mean': y_mean, 'std': y_std}, f)
    print(f"[WARN] nn/rt_scale.json not found — created with defaults "
          f"(mean={y_mean}, std={y_std}). "
          f"Run `python -m nn.train` for accurate values.")
    return y_mean, y_std


# -----------------------------
# TRAIN
# -----------------------------
def train():
    print("[INFO] Training model...")

    X, y, y_mean, y_std = load_dataset()

    # Save scale so incremental training and inference can undo normalisation
    os.makedirs('nn', exist_ok=True)
    with open(SCALE_PATH, 'w') as f:
        json.dump({'mean': y_mean, 'std': y_std}, f)

    X = preprocess(X, fit=True)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset    = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE)

    model    = ResponseTimeNN(input_dim=5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.SmoothL1Loss(beta=0.5)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds  = model(xb)
            loss   = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += loss_fn(model(xb), yb).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print("[DONE] Training complete")


# -----------------------------
# INCREMENTAL TRAIN
# -----------------------------
def train_incremental(X_new, y_new):
    if len(X_new) == 0:
        return

    # ── load scale — never crashes now ────────────────────
    y_mean, y_std = load_scale()

    # ── load feature scaler ───────────────────────────────
    if not os.path.exists(SCALER_PATH):
        print("[WARN] nn/scaler.pkl not found — skipping incremental retrain. "
              "Run `python -m nn.train` first.")
        return

    scaler   = joblib.load(SCALER_PATH)
    X_new    = np.nan_to_num(X_new, nan=0.0, posinf=1.0, neginf=0.0)
    X_scaled = scaler.transform(X_new)

    y_new = np.array(y_new, dtype=np.float32).reshape(-1, 1)
    y_new = np.clip(y_new, 50.0, 5000.0)
    y_new = (y_new - y_mean) / y_std

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_new,    dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ResponseTimeNN(input_dim=5).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.SmoothL1Loss(beta=0.5)

    for epoch in range(3):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds  = model(xb)
            loss   = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    print("[DONE] Incremental update done")


if __name__ == "__main__":
    train()