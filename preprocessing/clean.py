# preprocessing/clean.py

import pandas as pd
import os


# -----------------------------
# PATHS
# -----------------------------
RAW_PATH = "data/raw/cloud_dataset.csv"
OUTPUT_PATH = "data/processed/clean_dataset.csv"


# -----------------------------
# REQUIRED COLUMNS (final system)
# -----------------------------
REQUIRED_COLUMNS = {
    "CPU_Utilization (%)": "CPU_utilization",
    "Memory_Utilization (%)": "Memory_utilization",
    "Network_Bandwidth_Utilization (Mbps)": "Bandwidth_utilization",
    "Queue_Pressure": "Queue_pressure",
    "Number_of_Active_Users": "Active_users",
    "Task_Execution_Time (ms)": "Response_Time"  # target
}


# -----------------------------
# CLEAN FUNCTION
# -----------------------------
def clean_dataset():
    print("🧹 Cleaning dataset...")

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    print(f"Original shape: {df.shape}")

    # -----------------------------
    # Step 1: Keep only required columns
    # -----------------------------
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[list(REQUIRED_COLUMNS.keys())]

    # -----------------------------
    # Step 2: Rename columns
    # -----------------------------
    df = df.rename(columns=REQUIRED_COLUMNS)

    # -----------------------------
    # Step 3: Remove null values
    # -----------------------------
    df = df.dropna()

    # -----------------------------
    # Step 4: Remove duplicates
    # -----------------------------
    df = df.drop_duplicates()

    # -----------------------------
    # Step 5: Ensure numeric types
    # -----------------------------
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # -----------------------------
    # Final check
    # -----------------------------
    print(f"Cleaned shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    # -----------------------------
    # Save cleaned dataset
    # -----------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Clean dataset saved at {OUTPUT_PATH}")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    clean_dataset()