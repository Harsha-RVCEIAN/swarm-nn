# config.py

"""
Global configuration for SWARM+NN system
Keep ONLY system-level parameters here
"""

# -----------------------------
# Simulation Settings
# -----------------------------

# Number of servers in system
NUM_SERVERS = 4

# Total simulation steps
NUM_STEPS = 200

# Delay between steps (for readability/logging)
STEP_DELAY = 0.05


# -----------------------------
# Feature Scaling (for fake simulation only)
# -----------------------------
# These define realistic ranges for generated states

CPU_MAX = 100                # %
MEMORY_MAX = 100             # %
BANDWIDTH_MAX = 1000         # Mbps
QUEUE_MAX = 50               # queue pressure
USERS_MAX = 200              # number of users


# -----------------------------
# Noise Control (simulation realism)
# -----------------------------

# Random noise added to response time
NOISE_LEVEL = 10


# -----------------------------
# Retraining Control (global override if needed)
# -----------------------------

ENABLE_RETRAINING = True


# -----------------------------
# Debug / Logging Control
# -----------------------------

PRINT_STEP_LOGS = True
PRINT_SUMMARY = True


# -----------------------------
# Paths (optional centralization)
# -----------------------------

DATA_PATH = "data/processed/clean_dataset.csv"
MODEL_PATH = "nn/model.pt"
SCALER_PATH = "nn/scaler.pkl"
LOG_DIR = "logs"