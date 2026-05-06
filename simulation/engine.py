# simulation/engine.py

from simulation.server import Server
from simulation.workload import WorkloadGenerator
from simulation.state_builder import StateBuilder

from backend.controller import process_request, process_feedback

import numpy as np


class SimulationEngine:

    def __init__(self, num_servers=4):

        self.servers = [
            Server(i, service_rate=np.random.uniform(0.5, 2.5))
            for i in range(num_servers)
        ]

        self.workload = WorkloadGenerator(arrival_rate=1.0)

        self.state_builder = StateBuilder(self.servers)

        self.time = 0
        self.completed_requests = []
        self.entropy_history = []
        self.variance_history = []
        self.regret_history = []
        self.cumulative_regret = 0.0

    # -----------------------------
    # Run simulation
    # -----------------------------
    def run(self, steps=100, scenario='normal'):

        print(f"[INFO] Starting {scenario.upper()} Simulation...\n")

        for t in range(steps):
            self.time = t

            # SCENARIO OVERRIDES
            if scenario == 'burst':
                if 100 <= t < 150:
                    self.workload.arrival_rate = 3.0  # Burst load
                else:
                    self.workload.arrival_rate = 1.0
            
            if scenario == 'adversarial':
                if t == 100:
                    # Artificially degrade the fastest server
                    fastest = min(self.servers, key=lambda s: s.service_rate)
                    fastest.service_rate *= 0.1  # 10x slower suddenly

            # -----------------------------
            # Step 1: Generate requests
            # -----------------------------
            new_requests = self.workload.generate(t)

            # -----------------------------
            # Step 2: Assign requests
            # -----------------------------
            for request in new_requests:

                state_matrix = self.state_builder.build()

                selected_server, predicted_rt, probs = process_request(state_matrix)
                
                # Track entropy
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                self.entropy_history.append(entropy)

                # attach decision-time state
                request["decision_state"] = state_matrix.copy()
                request["predicted_rt"] = predicted_rt.copy()

                self.servers[selected_server].add_request(request)

            # Track load variance
            queues = [len(s.queue) for s in self.servers]
            self.variance_history.append(np.var(queues))

            # -----------------------------
            # Step 3: Process servers (FIXED)
            # -----------------------------
            for i, server in enumerate(self.servers):

                completed_request, actual_rt = server.step(t)

                # 🔥 CRITICAL FIX
                if actual_rt is not None:
                    actual_rt = max(actual_rt, 50.0)

                if completed_request is not None:

                    state_matrix = completed_request["decision_state"]
                    predicted_rt = completed_request["predicted_rt"]

                    process_feedback(state_matrix, i, actual_rt)

                    self.completed_requests.append(actual_rt)
                    
                    # Track Regret
                    best_rt = np.min(state_matrix[:, 3])
                    regret = actual_rt - best_rt
                    self.cumulative_regret += regret
                    avg_regret = self.cumulative_regret / len(self.completed_requests)
                    self.regret_history.append(avg_regret)

            # -----------------------------
            # Debug logs
            # -----------------------------
            if t % 10 == 0:
                avg_rt = self.get_average_rt()

                print(f"Time {t} | Avg RT: {avg_rt:.2f}")

                queues = [len(s.queue) for s in self.servers]
                print(f"[DEBUG] Step {t} | Queues: {queues}")

        print(f"n[DONE] {scenario.upper()} Simulation Complete")
        self.plot_metrics(scenario)

    # -----------------------------
    # Metrics
    # -----------------------------
    def get_average_rt(self):
        if len(self.completed_requests) == 0:
            return 0
        return sum(self.completed_requests) / len(self.completed_requests)

    # -----------------------------
    # Plotting
    # -----------------------------
    def plot_metrics(self, scenario='normal'):
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(18, 5))
            
            # Plot Entropy
            plt.subplot(1, 3, 1)
            plt.plot(self.entropy_history, color='blue', alpha=0.7)
            plt.title(f'[{scenario.upper()}] Probability Entropy')
            plt.xlabel('Request / Step')
            plt.ylabel('Entropy (bits)')
            plt.grid(True)
            
            # Plot Variance
            plt.subplot(1, 3, 2)
            plt.plot(self.variance_history, color='red', alpha=0.7)
            plt.title(f'[{scenario.upper()}] Load Variance')
            plt.xlabel('Step')
            plt.ylabel('Queue Length Variance')
            plt.grid(True)

            # Plot Regret
            plt.subplot(1, 3, 3)
            plt.plot(self.regret_history, color='purple', alpha=0.7)
            plt.title(f'[{scenario.upper()}] Average Regret')
            plt.xlabel('Completed Request')
            plt.ylabel('Avg Regret (ms)')
            plt.grid(True)
            
            import os
            os.makedirs('output', exist_ok=True)
            filename = os.path.join('output', f'simulation_metrics_{scenario}.png')
            plt.savefig(filename)
            print(f"[PLOT] Metrics plotted and saved to '{filename}'")
            # plt.show() # Disabled to prevent blocking execution
        except ImportError:
            print("[WARN] matplotlib not installed. Skipping plots.")
            print("To view plots, run: pip install matplotlib")