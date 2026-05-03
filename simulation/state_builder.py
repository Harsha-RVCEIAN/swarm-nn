# simulation/state_builder.py

import numpy as np


class StateBuilder:
    """
    Converts server objects into NN-ready state matrix
    """

    def __init__(self, servers):
        self.servers = servers

    # -----------------------------
    # Build state matrix
    # -----------------------------
    def build(self):

        state_matrix = []

        for server in self.servers:

            state = server.get_state()

            # -----------------------------
            # Validate state length
            # -----------------------------
            if len(state) != 5:
                raise ValueError(
                    f"Server {server.server_id} returned invalid state size: {len(state)}"
                )

            # -----------------------------
            # Convert to numpy safely
            # -----------------------------
            state = np.array(state, dtype=np.float32)

            # -----------------------------
            # 🔥 FIX 1: Handle NaN / Inf
            # -----------------------------
            state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=0.0)

            # -----------------------------
            # 🔥 FIX 2: Clamp CORRECT ranges (match dataset)
            # -----------------------------
            state[0] = np.clip(state[0], 0, 100)      # CPU (%)
            state[1] = np.clip(state[1], 0, 8000)     # Memory (MB) ← YOU WERE WRONG HERE
            state[2] = np.clip(state[2], 0, 1000)     # Bandwidth (Mbps)
            state[3] = np.clip(state[3], 0, 5000)     # 🔥 WAITING TIME (ms)
            state[4] = np.clip(state[4], 0, 5000)     # Users

            # -----------------------------
            # 🔥 FIX 3: Add small noise (prevents identical states)
            # -----------------------------
            # noise = np.random.uniform(-0.5, 0.5, size=5)
            # state = state + noise

            state_matrix.append(state)
        # -----------------------------
        # Convert to final matrix
        # -----------------------------
        state_matrix = np.array(state_matrix, dtype=np.float32)
        state_matrix = np.maximum(state_matrix, 0)
        # -----------------------------
        # 🔥 FIX 4: Detect identical states (CRITICAL)
        # -----------------------------
        if np.allclose(state_matrix, state_matrix[0]):
            state_matrix += np.random.uniform(-1.0, 1.0, state_matrix.shape)

        # -----------------------------
        # 🔥 FIX 5: Final sanity check
        # -----------------------------
        if np.isnan(state_matrix).any():
            raise ValueError("State matrix contains NaN values")

        return state_matrix