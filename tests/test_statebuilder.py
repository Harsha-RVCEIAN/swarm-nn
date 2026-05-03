from simulation.server import Server
from simulation.state_builder import StateBuilder


def test_state_builder():

    servers = [
        Server(0, 1.0),
        Server(1, 1.5),
        Server(2, 0.8)
    ]

    # 🔥 Add activity
    for t in range(5):
        for server in servers:
            server.add_request({"arrival_time": t})
            server.step(t)

    builder = StateBuilder(servers)

    state = builder.build()

    print("State matrix:")
    print(state)
    print("Shape:", state.shape)


if __name__ == "__main__":
    test_state_builder()