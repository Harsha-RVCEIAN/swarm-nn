# nn/model.py

import torch
import torch.nn as nn


class ResponseTimeNN(nn.Module):
    """
    Neural Network for predicting response time
    Input: 5 features
    Output: 1 value (response time)
    """

    def __init__(self, input_dim=5):
        super(ResponseTimeNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)  # Output layer (no activation)
        )

    def forward(self, x):
        return self.model(x)