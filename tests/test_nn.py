# tests/test_nn.py

import numpy as np
from nn.inference import Predictor

def test_nn_behavior():
    model = Predictor(input_dim=5)

    low_load = np.array([[10, 10, 100, 5, 20]])
    high_load = np.array([[90, 90, 900, 40, 180]])

    pred_low = model.predict(low_load)[0]
    pred_high = model.predict(high_load)[0]

    print("Low load RT:", pred_low)
    print("High load RT:", pred_high)

    assert pred_high > pred_low, "NN is not learning load behavior"

if __name__ == "__main__":
    test_nn_behavior()