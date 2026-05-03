# backend/logger.py

import logging
import os
from datetime import datetime


# -----------------------------
# Create logs directory
# -----------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with timestamp
log_filename = datetime.now().strftime("system_%Y%m%d_%H%M%S.log")
log_path = os.path.join(LOG_DIR, log_filename)


# -----------------------------
# Configure logger
# -----------------------------
logger = logging.getLogger("NN_ACO_SYSTEM")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Format
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# -----------------------------
# Logging functions
# -----------------------------

def log_state(state_matrix):
    logger.info(f"STATE: {state_matrix}")


def log_prediction(predicted_rt):
    logger.info(f"PREDICTED_RT: {predicted_rt}")


def log_probabilities(probabilities):
    logger.info(f"PROBABILITIES: {probabilities}")


def log_selection(selected_server):
    logger.info(f"SELECTED_SERVER: {selected_server}")


def log_actual_rt(actual_rt):
    logger.info(f"ACTUAL_RT: {actual_rt}")


def log_pheromone(pheromone):
    logger.info(f"PHEROMONE: {pheromone}")


def log_retraining():
    logger.info("RETRAINING TRIGGERED")


def log_error(error_message):
    logger.error(f"ERROR: {error_message}")