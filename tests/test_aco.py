# tests/test_aco.py

import numpy as np
from aco.aco import ACO

def test_aco_behavior():
    aco = ACO(num_servers=4)

    predicted_rt = np.array([100, 50, 80, 120])

    probs = aco.compute_probabilities(predicted_rt)

    print("Probabilities:", probs)
    print("Sum:", np.sum(probs))

    assert np.isclose(np.sum(probs), 1.0), "Probabilities do not sum to 1"

    # simulate updates
    for _ in range(10):
        server = aco.select_server(probs)
        aco.update(server, actual_rt=predicted_rt[server])

    print("Updated pheromone:", aco.get_pheromone())

if __name__ == "__main__":
    test_aco_behavior()