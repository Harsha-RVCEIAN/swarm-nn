# tests/test_server.py

from simulation.server import Server

def test_server():

    server = Server(0, service_rate=2.0)

    for t in range(30):

        # 🔥 FIX HERE (not in server.py)
        if t % 2 == 0:
            server.add_request({"arrival_time": t})

        rt = server.step(t)
        print(f"Time {t} → Response Time:", rt)

if __name__ == "__main__":
    test_server()