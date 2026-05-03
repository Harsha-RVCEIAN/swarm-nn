# preprocessing/feature_engineering.py

import pandas as pd
import os


# -----------------------------
# PATHS
# -----------------------------
INPUT_PATH = "data/processed/clean_dataset.csv"
OUTPUT_PATH = "data/processed/clean_dataset_fe.csv"


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def engineer_features():
    print("⚙️ Performing feature engineering...")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Dataset not found at {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print(f"Original shape: {df.shape}")

    # -----------------------------
    # Basic validation
    # -----------------------------
    required_cols = [
        "CPU_utilization",
        "Memory_utilization",
        "Bandwidth_utilization",
        "Queue_pressure",
        "Active_users",
        "Response_Time"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # -----------------------------
    # Feature 1: Load Index
    # -----------------------------
    df["Load_Index"] = (
        0.4 * df["CPU_utilization"] +
        0.3 * df["Memory_utilization"] +
        0.2 * df["Bandwidth_utilization"] +
        0.1 * df["Queue_pressure"]
    )

    # -----------------------------
    # Feature 2: Resource Imbalance
    # -----------------------------
    df["Resource_Imbalance"] = (
        abs(df["CPU_utilization"] - df["Memory_utilization"]) +
        abs(df["Memory_utilization"] - df["Bandwidth_utilization"] / 10)
    )

    # -----------------------------
    # Final check
    # -----------------------------
    print(f"New shape: {df.shape}")
    print("New columns:", df.columns.tolist())

    # -----------------------------
    # Save dataset
    # -----------------------------
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Feature-engineered dataset saved at {OUTPUT_PATH}")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    engineer_features()