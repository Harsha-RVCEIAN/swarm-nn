# aco/aco.py  — COMPLETE FILE
"""
ACO - pheromone + probability selection
"""

import numpy as np


class ACO:
    def __init__(self, num_servers, alpha=1.0, beta=3.0, rho=0.1, Q=1.0,
                 tau_min=0.01, tau_max=10.0):
        self.num_servers = num_servers
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.tau_min = tau_min
        self.tau_max = tau_max

        self.pheromone = np.ones(num_servers)
        self.cooldown = np.zeros(num_servers)
        self.moving_avg_rt = 500.0
        self.step_count = 0

    def compute_probabilities(self, predicted_rt, cpu_utilization=None,
                               waiting_time=None, queue_lengths=None):
        predicted_rt = np.array(predicted_rt, dtype=float)
        self.step_count += 1

        # Clamp NN output to realistic range
        predicted_rt = np.clip(predicted_rt, 50, 5000)

        # Heuristic: lower predicted RT = higher desirability
        eta = 1.0 / (predicted_rt + 1e-6)

        # ACO score
        score = (self.pheromone ** self.alpha) * (eta ** self.beta)

        # Zero out servers in cooldown
        for i in range(self.num_servers):
            if self.cooldown[i] > 0:
                score[i] *= 0.05   # heavily penalized but not dead
                self.cooldown[i] -= 1

        if score.sum() == 0 or np.isnan(score.sum()):
            prob = np.ones(self.num_servers) / self.num_servers
        else:
            prob = score / score.sum()

        return prob

    def select_server(self, probabilities):
        return np.random.choice(self.num_servers, p=probabilities)

    def update(self, selected_server, actual_rt):
        actual_rt = max(actual_rt, 1.0)

        # Update moving average
        self.moving_avg_rt = 0.9 * self.moving_avg_rt + 0.1 * actual_rt

        # Evaporate
        self.pheromone = (1 - self.rho) * self.pheromone

        # Deposit reward (better RT = more pheromone)
        reward = np.exp(-actual_rt / 500.0)
        self.pheromone[selected_server] += reward

        # Isolate on extreme latency
        # Only isolate extreme outliers, shorter cooldown
        if actual_rt > 3500.0:
            self.pheromone[selected_server] *= 0.3  # was 0.1
            self.cooldown[selected_server] = 2       # was 20

        # Bounds
        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)

    def get_pheromone(self):
        return self.pheromone.copy()   # ← trailing comma was here — FIXED