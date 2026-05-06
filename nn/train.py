# nn/train.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import joblib

from nn.model import ResponseTimeNN


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/processed/clean_dataset.csv"
MODEL_PATH = "nn/model.pt"
SCALER_PATH = "nn/scaler.pkl"

BATCH_SIZE = 32
EPOCHS = 100
LR = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# LOAD DATA
# -----------------------------
def load_dataset():
    df = pd.read_csv(DATA_PATH)

    df = df.rename(columns={
        "CPU_Utilization (%)": "CPU_utilization",
        "Memory_Consumption (MB)": "Memory_utilization",
        "Network_Bandwidth_Utilization (Mbps)": "Bandwidth_utilization",
        "Task_Waiting_Time (ms)": "Queue_pressure",
        "Number_of_Active_Users": "Active_users",
        "response_time(ms)": "Response_Time"
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

    # Add these 3 lines
    y_mean = y.mean()
    y_std  = y.std()
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
# TRAIN
# -----------------------------
def train():
    print("[INFO] Training model...")

    X, y, y_mean, y_std = load_dataset()
    # Save them so inference can undo the scaling
    import json
    json.dump({'mean': float(y_mean), 'std': float(y_std)}, open('nn/rt_scale.json','w'))
    X = preprocess(X, fit=True)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = ResponseTimeNN(input_dim=5).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    best_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            preds = model(xb)
            loss = loss_fn(preds, yb)

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
                preds = model(xb)
                val_loss += loss_fn(preds, yb).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print("[DONE] Training complete")


# -----------------------------
# INCREMENTAL TRAIN (FIXED)
# -----------------------------
def train_incremental(X_new, y_new):
    if len(X_new) == 0:
        return

    # Load scale params to normalize targets same as training
    import json
    scale = json.load(open('nn/rt_scale.json'))
    y_mean, y_std = scale['mean'], scale['std']

    scaler = joblib.load(SCALER_PATH)
    X_new  = np.nan_to_num(X_new, nan=0.0, posinf=1.0, neginf=0.0)
    X_scaled = scaler.transform(X_new)

    y_new = np.array(y_new, dtype=np.float32).reshape(-1, 1)
    y_new = np.clip(y_new, 50.0, 5000.0)
    y_new = (y_new - y_mean) / y_std          # ← normalize same as training

    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_new, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ResponseTimeNN(input_dim=5).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    for epoch in range(3):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            preds = model(xb)
            loss = loss_fn(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    print("[DONE] Incremental update done")


if __name__ == "__main__":
    train()