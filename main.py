# main.py

import numpy as np
import time

from backend.controller import process_request, process_feedback
from aco.params import NUM_SERVERS

# NEW: real simulation
from simulation.engine import SimulationEngine


# -----------------------------
# CONFIG
# -----------------------------
NUM_STEPS = 200

# Toggle mode
USE_REAL_SIMULATION = True   # change to False if you want old fake mode


# -----------------------------
# FAKE STATE GENERATOR (KEEPED)
# -----------------------------
def generate_state(num_servers):
    return np.random.rand(num_servers, 5) * np.array([
        100,   # CPU %
        100,   # Memory %
        1000,  # Bandwidth
        50,    # Queue pressure
        200    # Active users
    ])


# -----------------------------
# FAKE EXECUTION (KEEPED)
# -----------------------------
def simulate_execution(server_state):

    cpu, mem, bw, queue, users = server_state

    response_time = (
        0.5 * cpu +
        0.3 * mem +
        0.1 * queue * 10 +
        0.05 * users +
        np.random.rand() * 10
    )

    return response_time


# -----------------------------
# OLD SIMULATION (UNCHANGED)
# -----------------------------
def run_fake_simulation():

    print("[INFO] Running FAKE simulation...\n")

    for step in range(NUM_STEPS):

        state_matrix = generate_state(NUM_SERVERS)

        selected_server, predicted_rt, probabilities = process_request(state_matrix)

        actual_rt = simulate_execution(state_matrix[selected_server])

        process_feedback(state_matrix, selected_server, actual_rt)

        print(f"Step {step+1}")
        print(f"Selected Server: {selected_server}")
        print(f"Predicted RT: {predicted_rt}")
        print(f"Actual RT: {actual_rt:.2f}")
        print(f"Probabilities: {probabilities}\n")

        time.sleep(0.05)


# -----------------------------
# REAL SIMULATION (NEW)
# -----------------------------
def run_real_simulation(scenario):

    print(f"[INFO] Initializing {scenario.upper()} simulation...\n")

    engine = SimulationEngine(num_servers=NUM_SERVERS)
    engine.run(steps=NUM_STEPS, scenario=scenario)


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":

    if USE_REAL_SIMULATION:
        print("Running all 3 validation scenarios...\n")
        run_real_simulation('normal')
        print("\n" + "="*50 + "\n")
        run_real_simulation('burst')
        print("\n" + "="*50 + "\n")
        run_real_simulation('adversarial')
    else:
        run_fake_simulation()