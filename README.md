# 🐜 Neural Swarm Load Balancer (ACO + ML)

An adaptive load balancing system that combines **Ant Colony Optimization (ACO)** with **Neural Network-based latency estimation** to distribute requests across servers under dynamic and adversarial conditions.

---

## 🚀 Overview

This project implements a **feedback-driven load balancing system** where:

- ACO provides **adaptive routing decisions** based on historical performance.
- A Neural Network provides **latency estimates** based on system state.
- A **cooldown-based isolation mechanism** prevents cascading failures.
- The system is evaluated under:
  - Normal load
  - Burst traffic
  - Adversarial degradation

The goal is to build a **stable and explainable adaptive system**, not just an optimized one.

---

## 🧠 Key Idea

Instead of relying on static rules (e.g., Round Robin), the system:

1. Learns from past routing decisions (ACO pheromones)
2. Estimates future latency (Neural Network)
3. Reacts to failures using **temporary isolation (cooldown)**
4. Maintains balance using **probabilistic routing**

---

## 🏗️ Architecture


Request → Controller → ACO Decision Engine → Server Selection
↑
Neural Network (Latency Prediction)
↑
System State (Queue, Wait Time, etc.)
↓
Feedback (Actual RT, Queue Update)


---

## ⚙️ Components

### 1. ACO (Adaptive Routing)
- Maintains pheromone values per server
- Updates based on reward (response time)
- Balances exploration vs exploitation

### 2. Neural Network (Prediction)
- Predicts expected response time
- Uses normalized system features
- Guides ACO heuristic (soft influence)

### 3. Cooldown Mechanism (Critical Feature)
- Detects extreme latency spikes
- Temporarily removes bad servers from routing
- Prevents cascading overload

### 4. Simulation Engine
- Models server queues and workloads
- Supports multiple traffic patterns
- Generates evaluation metrics

---

## 📊 Evaluation Metrics

The system is evaluated using:

- **Response Time (RT)**
- **Load Variance**
- **Entropy (Decision randomness)**
- **Regret**  
  *(Difference between chosen server and best possible server at that time)*

---

## 🧪 Scenarios Tested

### 1. Normal Load
- Stable traffic
- Verifies steady-state behavior

### 2. Burst Load
- Sudden spike in requests
- Tests system stability under pressure

### 3. Adversarial Scenario
- One server is degraded intentionally
- Tests adaptation and recovery

---

## 📈 Results Summary

- System maintains **stable load distribution**
- Avoids **single-server overload**
- Recovers from **adversarial failures using cooldown**
- Regret stabilizes after initial disturbance

---

## ⚠️ Limitations

- Neural Network influence is **limited** (small prediction variance)
- System is **reactive**, not fully predictive
- No explicit modeling of **server capacity (μ vs λ dynamics)**

---

## ▶️ How to Run

### 1. Setup Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
2. Run Simulation
python main.py
3. View Outputs

Generated plots:

simulation_metrics_normal.png
simulation_metrics_burst.png
simulation_metrics_adversarial.png
📂 Project Structure
swarm+nn/
├── aco/              # ACO logic
├── nn/               # Neural network
├── backend/          # Controller + API
├── simulation/       # Environment
├── evaluation/       # Metrics
├── visualization/    # Plots
├── data/             # Dataset
├── logs/             # Logs
├── tests/            # Tests
├── main.py
├── config.py
└── requirements.txt
🧠 Key Takeaway

The most important insight from this project:

Hard isolation (cooldown) is more effective than gradual penalties in preventing cascading failures in dynamic load systems.

📌 Future Improvements
Add capacity-aware modeling (service rate vs arrival rate)
Improve NN prediction robustness
Replace reactive cooldown with predictive failure detection
Extend to real distributed systems (e.g., Kubernetes)
👤 Author

Harsha
Engineering Student – Systems, ML, and Distributed Computing
