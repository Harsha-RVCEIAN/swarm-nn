"""
🐜 Ant Colony Optimization - pheromone + probability selection
Decision module for server/action selection (ENHANCED STABLE VERSION)
"""

import numpy as np


class ACO:
    def __init__(
        self,
        num_servers,
        alpha=0.6,
        beta=2.0,
        rho=0.05,
        Q=1.0,
        epsilon=0.05,
        tau_min=1e-4,
        tau_max=10.0
    ):
        self.num_servers = num_servers
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.epsilon = epsilon
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Initialize pheromone
        self.pheromone = np.ones(num_servers)

        # -----------------------------
        # RESEARCH-GRADE STATE TRACKERS
        # -----------------------------
        self.W = 10
        self.usage_history = []
        self.cooldown = np.zeros(num_servers)
        self.prev_queue = np.zeros(num_servers)
        self.moving_avg_rt = 500.0
        self.gamma = 2.0
        
        # 🔥 FINAL STABILITY CONTROLS
        self.T = 1.0
        self.prev_prob = np.ones(num_servers) / num_servers
        self.step_count = 0
        self.prev_advantage = 0.0
        self.gamma = 0.3

    # -----------------------------
    # Step 1: Compute probabilities
    # -----------------------------
    def compute_probabilities(self, predicted_rt, cpu_utilization=None, waiting_time=None, queue_lengths=None):
        predicted_rt = np.array(predicted_rt, dtype=float)

        # 1. Warm-up
        if self.step_count < 50:
            prob = np.ones(self.num_servers) / self.num_servers
            self.prev_prob = prob.copy()
            self.step_count += 1
            return prob

        self.step_count += 1

        # 2. Clamp NN output
        predicted_rt = np.clip(predicted_rt, 100, 2000)

        # 🔥 Continuous NN Uncertainty Handling
        uncertainty = np.std(predicted_rt)
        self.beta = 2.0 * (uncertainty / (uncertainty + 50.0))
        self.beta = np.clip(self.beta, 0.8, 2.0)

        # 3. Heuristic
        eta = np.exp(-predicted_rt / 400.0)

        # 4. Combine with pheromone
        score = (self.pheromone ** self.alpha) * (eta ** self.beta)
        if score.sum() == 0 or np.isnan(score.sum()):
            prob = np.ones(self.num_servers) / self.num_servers
        else:
            prob = score / score.sum()

        # 5. Temporal smoothing (Adaptive)
        prob = (1.0 - self.gamma) * self.prev_prob + self.gamma * prob

        # 🔥 Feasibility constraint (Strict)
        mask = predicted_rt < 1000
        prob = prob * mask.astype(float)
        
        # 🔥 Cooldown Enforcement
        for i in range(self.num_servers):
            if self.cooldown[i] > 0:
                prob[i] = 0.0
                self.cooldown[i] -= 1

        if prob.sum() == 0:
            prob = np.ones_like(prob) / len(prob)
        else:
            prob = prob / prob.sum()

        # 🔥 Simple Dynamic Cap
        confidence = 1.0 - (uncertainty / 100.0)
        confidence = np.clip(confidence, 0.0, 1.0)
        dynamic_cap = 0.45 + 0.3 * confidence

        # 🔥 Simplex Projection for Cap
        while True:
            over = prob > dynamic_cap
            if not np.any(over):
                break

            excess = prob[over] - dynamic_cap
            prob[over] = dynamic_cap

            redistribute = excess.sum()
            under = prob < dynamic_cap
            if np.sum(under) == 0:
                break

            prob[under] += redistribute / np.sum(under)

        self.prev_prob = prob.copy()

        return prob

    # -----------------------------
    # Step 2: Select server
    # -----------------------------
    def select_server(self, probabilities):
        return np.random.choice(self.num_servers, p=probabilities)

    # -----------------------------
    # Step 3: Update pheromone
    # -----------------------------
    def update(self, selected_server, actual_rt):

        actual_rt = max(actual_rt, 1.0)

        # 🔥 Baseline Correction (Relative learner)
        alpha_ma = 0.1
        self.moving_avg_rt = (1 - alpha_ma) * self.moving_avg_rt + alpha_ma * actual_rt
        
        advantage = self.moving_avg_rt - actual_rt
        
        # 🔥 Shock Detection
        shock = actual_rt > 2.0 * self.moving_avg_rt
        self.gamma = 0.6 if shock else 0.3
        self.rho = 0.2 if shock else 0.05
        
        # 🔥 Smoothed Advantage Signal
        smoothed_advantage = 0.8 * self.prev_advantage + 0.2 * advantage
        self.prev_advantage = smoothed_advantage

        # 🔥 Time-decay learning with relative advantage
        decay = np.exp(-self.step_count / 200.0)
        reward = decay * np.exp(smoothed_advantage / 300.0) * self.Q

        # 🔥 Evaporation
        self.pheromone = (1 - self.rho) * self.pheromone

        # 🔥 Deposit
        self.pheromone[selected_server] += reward

        # 🔥 Hard penalty condition (Temporary Isolation)
        if actual_rt > 2000.0:
            self.pheromone[selected_server] *= 0.1
            self.cooldown[selected_server] = 20  # isolate for 20 requests

        # 🔥 HARD BOUND
        self.pheromone = np.clip(self.pheromone, 0.1, 0.6)

        # 🔥 Normalize
        self.pheromone = self.pheromone / self.pheromone.sum()

    # -----------------------------
    # Debug helper
    # -----------------------------
    def get_pheromone(self):
        return self.pheromone.copy()