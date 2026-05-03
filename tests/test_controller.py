# tests/test_controller.py

import numpy as np
from backend.controller import process_request, process_feedback

def test_controller():

    state = np.array([
        [30, 40, 200, 10, 50],
        [70, 80, 800, 30, 150],
        [50, 60, 400, 20, 100],
        [20, 30, 150, 5, 30]
    ])

    server, predicted_rt, probs = process_request(state)

    print("Selected server:", server)
    print("Predicted RT:", predicted_rt)
    print("Probabilities:", probs)

    assert len(predicted_rt) == 4
    assert np.isclose(np.sum(probs), 1.0)

    # simulate feedback
    actual_rt = predicted_rt[server] + 10
    process_feedback(state, server, actual_rt)

    print("Feedback processed successfully")

if __name__ == "__main__":
    test_controller()