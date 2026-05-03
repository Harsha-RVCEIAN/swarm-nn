# simulation/workload.py

import random


class WorkloadGenerator:
    """
    Generates incoming requests based on arrival rate (lambda)
    """

    def __init__(self, arrival_rate=0.5):
        """
        arrival_rate (λ):
            Average number of requests per time step

            Example:
                0.5 → 1 request every 2 steps (low load)
                1.0 → ~1 request per step (medium load)
                2.0 → ~2 requests per step (high load)
        """
        self.arrival_rate = arrival_rate

    # -----------------------------
    # Generate requests for a time step
    # -----------------------------
    def generate(self, current_time):
        """
        Returns:
            list of requests arriving at this time step
        """

        num_requests = self._poisson_sample(self.arrival_rate)

        requests = []
        for _ in range(num_requests):
            request = {
                "arrival_time": current_time
            }
            requests.append(request)

        return requests

    # -----------------------------
    # Poisson sampling (simple version)
    # -----------------------------
    def _poisson_sample(self, lam):
        """
        Generate Poisson-distributed number using
        exponential inter-arrival approximation
        """

        L = pow(2.71828, -lam)
        k = 0
        p = 1

        while p > L:
            k += 1
            p *= random.random()

        return max(0, k - 1)