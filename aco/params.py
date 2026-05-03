# aco/params.py

"""
ACO Configuration Parameters
Modify these values to tune system behavior
"""

# -----------------------------
# ACO Core Parameters
# -----------------------------

# Influence of pheromone (experience)
ALPHA = 1.0

# Influence of heuristic (NN prediction)
BETA = 5.0

# Evaporation rate (0 < RHO < 1)
RHO = 0.1

# Reward factor (higher → stronger reinforcement)
Q = 1.0


# -----------------------------
# System Parameters
# -----------------------------

# Number of servers (must match simulation)
NUM_SERVERS = 4


# -----------------------------
# Safety / Stability
# -----------------------------

# Prevent division by zero
EPSILON = 1e-6


# -----------------------------
# Retraining Control
# -----------------------------

# How often to retrain NN (in number of requests)
RETRAIN_INTERVAL = 50


# -----------------------------
# Debug Flags
# -----------------------------

# Print probabilities each step
DEBUG_PROBABILITIES = False

# Print pheromone updates
DEBUG_PHEROMONE = False