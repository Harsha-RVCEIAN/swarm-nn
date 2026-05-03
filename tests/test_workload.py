from simulation.workload import WorkloadGenerator

def test_workload():

    gen = WorkloadGenerator(arrival_rate=1.0)

    for t in range(20):
        reqs = gen.generate(t)
        print(f"Time {t} → Requests:", len(reqs))

if __name__ == "__main__":
    test_workload()