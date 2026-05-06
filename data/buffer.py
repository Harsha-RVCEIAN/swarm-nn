# data/buffer.py

import numpy as np
from collections import deque


class Buffer:
    """
    Stores recent (state, response_time) pairs
    Used for incremental training
    """

    def __init__(self, max_size=1000):
        """
        max_size: maximum number of samples to store
        """
        self.buffer = deque(maxlen=max_size)

    # -----------------------------
    # Add new experience
    # -----------------------------
    def add(self, state, response_time):
        """
        state: array-like (5 features)
        response_time: float
        """

        state = np.array(state, dtype=np.float32)
        response_time = float(response_time)

        self.buffer.append((state, response_time))

    # -----------------------------
    # Get buffer size
    # -----------------------------
    def size(self):
        return len(self.buffer)

    # -----------------------------
    # Get recent data (for training)
    # -----------------------------
    def get_recent_data(self):
        """
        Returns:
            list of (state, response_time)
        """
        return list(self.buffer)

    # -----------------------------
    # Clear buffer (optional)
    # -----------------------------
    def clear(self):
        self.buffer.clear() 