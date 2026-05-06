# simulation/server.py

import random
from collections import deque


MAX_QUEUE = 50  # 🔥 prevents system collapse


class Server:
    """
    Simulates a server with queue and processing capability
    """

    def __init__(self, server_id, service_rate):

        self.server_id = server_id
        self.service_rate = service_rate

        self.queue = deque()
        self.current_request = None
        self.remaining_time = 0

        # Metrics
        self.cpu_utilization = 0
        self.memory_utilization = 0
        self.bandwidth_utilization = 0
        self.active_users = 0
        
    # -----------------------------
    # Add request (bounded queue)
    # -----------------------------
    def add_request(self, request):
        if len(self.queue) < MAX_QUEUE:
            self.queue.append(request)

    # -----------------------------
    # Process one step
    # -----------------------------
    def step(self, current_time):

        completed_rt = None
        completed_request = None

        # -----------------------------
        # Start new request if idle
        # -----------------------------
        if self.current_request is None and self.queue:

            self.current_request = self.queue.popleft()

            exec_time = random.uniform(200, 800) / self.service_rate
            self.remaining_time = exec_time

            # Store timing
            self.current_request["start_time"] = current_time
            self.current_request["exec_time"] = exec_time

        # -----------------------------
        # Process current request
        # -----------------------------
        if self.current_request is not None:

            self.remaining_time -= 200  # 🔥 Increased processing power per step to handle arrival rate

            if self.remaining_time <= 0:

                req = self.current_request

                arrival = req["arrival_time"]
                start = req["start_time"]
                exec_time = req["exec_time"]

                # 🔥 REAL waiting time (correct)
                waiting_time_steps = start - arrival
                waiting_time_ms = waiting_time_steps * 200  # convert steps to ms

                completed_rt = waiting_time_ms + exec_time
                completed_request = req

                self.current_request = None

        # -----------------------------
        # Update metrics
        # -----------------------------
        self._update_metrics()

        return completed_request, completed_rt

    # -----------------------------
    # Update metrics
    # -----------------------------
    def _update_metrics(self):

        queue_length = len(self.queue)

        # 🔥 Realistic metrics that don't saturate instantly
        self.cpu_utilization = min(100, 10 + queue_length * 2.0)
        self.memory_utilization = min(8000, 500 + queue_length * 50)
        self.bandwidth_utilization = min(1000, 100 + queue_length * 10)

        self.active_users = queue_length * 1 + (1 if self.current_request else 0)

    # -----------------------------
    # State for NN
    # -----------------------------
    def get_state(self):

        queue_length = len(self.queue)

        # 🔥 FIX: Use the actual remaining backlog as expected waiting time
        waiting_time = self.remaining_time
        if self.queue:
            # Expected execution time per item is roughly 500 / service_rate
            expected_exec_time = 500 / self.service_rate
            waiting_time += queue_length * expected_exec_time

        waiting_time = min(waiting_time, 5000)
        users = min(self.active_users, 2000)

        return [
            self.cpu_utilization,
            self.memory_utilization,
            self.bandwidth_utilization,
            waiting_time,   # 🔥 NOW REALISTIC
            users
        ]