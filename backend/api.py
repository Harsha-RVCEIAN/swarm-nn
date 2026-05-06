# backend/api.py

import logging
import math
import random
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from backend.controller import process_request, process_feedback
from aco.aco import ACO

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NN-ACO")

app = FastAPI(
    title="NN-ACO Load Balancer API",
    description="Neural Network + Ant Colony Optimization Load Balancer",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Service rate pool — extended to support up to 12 servers
# Pattern: fast / slow / very-fast / slow / medium / fast / slow / medium ...
# ─────────────────────────────────────────────────────────────
BASE_SERVICE_RATES = [1.8, 1.2, 2.2, 0.9, 1.5, 2.0, 1.0, 1.7, 1.3, 2.1, 0.8, 1.6]

def get_service_rate(idx: int) -> float:
    """Return a deterministic service rate for server index idx."""
    return BASE_SERVICE_RATES[idx % len(BASE_SERVICE_RATES)]


# ─────────────────────────────────────────────────────────────
# Backend Server Model
# ─────────────────────────────────────────────────────────────

class SimServer:
    def __init__(self, idx: int, service_rate: float):
        self.id             = idx
        self.service_rate   = service_rate
        self.queue          = []
        self.total_rts      = []
        self.last_rt        = None
        self.degraded       = False
        self.total_requests = 0

    def to_state_row(self) -> list:
        """
        5-feature row for the NN: [CPU, Memory, Bandwidth, Queue_pressure, Active_users]
        All derived from real queue state — no random numbers.
        """
        q     = len(self.queue)
        cpu   = min(100.0, (q / 10.0) * 100.0)
        mem   = min(100.0, 40.0 + q * 3.0)
        bw    = min(1000.0, 100.0 + q * 30.0)
        wait  = min(5000.0, q * 120.0)
        users = float(q * 25)
        return [cpu, mem, bw, wait, users]

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "service_rate": round(self.service_rate, 4),
            "queue_len":    len(self.queue),
            "last_rt":      round(self.last_rt, 2) if self.last_rt is not None else None,
            "total_reqs":   self.total_requests,
            "degraded":     self.degraded,
        }


# ─────────────────────────────────────────────────────────────
# Backend Simulation  — fully dynamic server count
# ─────────────────────────────────────────────────────────────

class BackendSimulation:
    """
    Server count is set at reset time by the frontend.
    ACO, pheromone vector, and state matrix all resize accordingly.
    """

    def __init__(self):
        self.num_servers = 4        # default, overridden by /reset
        self.reset(num_servers=4)

    # ── Reset with arbitrary server count ─────────────────
    def reset(self, scenario: str = "normal", num_servers: int = 4):
        # Clamp to sane range
        num_servers = max(2, min(num_servers, 12))
        self.num_servers = num_servers

        # Build server pool dynamically
        self.servers = [
            SimServer(i, get_service_rate(i))
            for i in range(num_servers)
        ]

        # ACO resized to match server count
        self.aco = ACO(num_servers=num_servers)

        self.step           = 0
        self.arrival_rate   = 1.0
        self.scenario       = scenario
        self.cum_regret     = 0.0
        self.completed_rts  = []
        self.reg_history    = []
        self.last_selected  = -1
        self.last_probs     = [1.0 / num_servers] * num_servers
        self.last_preds     = [500.0] * num_servers
        self.last_pheromones = [1.0] * num_servers

        logger.info(
            "Simulation reset — scenario=%s  num_servers=%d  rates=%s",
            scenario, num_servers,
            [round(s.service_rate, 2) for s in self.servers]
        )

    # ── Build state matrix ────────────────────────────────
    def _state_matrix(self) -> np.ndarray:
        return np.array(
            [s.to_state_row() for s in self.servers],
            dtype=np.float32
        )

    # ── Poisson arrivals ──────────────────────────────────
    @staticmethod
    def _poisson(lam: float) -> int:
        lam = min(lam, 20.0)
        L = math.exp(-lam)
        k, p = 0, 1.0
        while True:
            k += 1
            p *= random.random()
            if p <= L:
                return max(0, k - 1)

    # ── One simulation step ───────────────────────────────
    def step_once(self) -> dict:

        # ── 1. Scenario events ────────────────────────────
        if self.scenario == "burst":
            self.arrival_rate = 3.0 if (100 <= self.step < 150) else 1.0

        if self.scenario == "adversarial" and self.step == 100:
            # Degrade the fastest server
            target = max(self.servers, key=lambda s: s.service_rate)
            target.service_rate = max(0.05, target.service_rate * 0.1)
            target.degraded = True
            logger.warning(
                "Server %d degraded at step 100 (adversarial)", target.id
            )

        # ── 2. NN + ACO routing decision ──────────────────
        # State matrix is (num_servers × 5) — dynamically sized
        state = self._state_matrix()

        try:
            selected_server, predicted_rt, probabilities = process_request(
                state, self.aco
            )
        except Exception as e:
            logger.error(
                "process_request failed at step %d: %s", self.step, e,
                exc_info=True
            )
            # Safe fallback
            selected_server = np.random.randint(0, self.num_servers)
            predicted_rt    = np.full(self.num_servers, 500.0)
            probabilities   = np.ones(self.num_servers) / self.num_servers

        selected_server   = int(selected_server)
        predicted_rt_list = predicted_rt.tolist()
        probs_list        = probabilities.tolist()
        pheromones_list   = self.aco.get_pheromone().tolist()

        self.last_selected    = selected_server
        self.last_probs       = probs_list
        self.last_preds       = predicted_rt_list
        self.last_pheromones  = pheromones_list

        # ── 3. Entropy ────────────────────────────────────
        p_arr   = np.array(probs_list) + 1e-12
        entropy = float(-np.sum(p_arr * np.log2(p_arr)))

        # ── 4. Arrivals → selected server ─────────────────
        n         = self._poisson(self.arrival_rate)
        best_pred = float(np.min(predicted_rt))
        for _ in range(n):
            self.servers[selected_server].queue.append({
                "arrival":   self.step,
                "best_pred": best_pred,
            })

        # ── 5. Process + feedback for every server ────────
        for i, s in enumerate(self.servers):
            if not s.queue:
                continue

            req      = s.queue.pop(0)
            svc_time = (1.0 / s.service_rate) * (0.5 + random.random())
            actual_rt = (
                svc_time * 400.0
                + len(s.queue) * 10.0
                + random.random() * 50.0
            )
            if s.degraded:
                actual_rt *= 4.0

            actual_rt = float(np.clip(actual_rt, 50.0, 5000.0))
            feedback_rt = min(actual_rt, 2500.0)  # cap what ACO sees — display can show real value

            try:
                process_feedback(state, i, feedback_rt, self.aco)
            except Exception as e:
                logger.error(
                    "process_feedback failed for server %d: %s", i, e,
                    exc_info=True
                )

            s.total_rts.append(actual_rt)
            s.last_rt = actual_rt
            s.total_requests += 1
            self.completed_rts.append(actual_rt)

            regret = max(0.0, actual_rt - req["best_pred"])
            self.cum_regret += regret
            self.reg_history.append(
                self.cum_regret / len(self.completed_rts)
            )

        # ── 6. Metrics ────────────────────────────────────
        avg_rt   = float(np.mean(self.completed_rts)) if self.completed_rts else 0.0
        qs       = [len(s.queue) for s in self.servers]
        variance = float(np.var(qs))
        avg_reg  = self.reg_history[-1] if self.reg_history else 0.0

        self.step += 1

        return {
            "selected_server": selected_server,
            "probabilities":   probs_list,
            "predicted_rt":    predicted_rt_list,
            "pheromones":      pheromones_list,
            "actual_rt":       round(avg_rt, 2),
            "variance":        round(variance, 4),
            "entropy":         round(entropy, 4),
            "regret":          round(avg_reg, 2),
            "arrival_rate":    self.arrival_rate,
            "step":            self.step,
            "num_servers":     self.num_servers,
            "servers":         [s.to_dict() for s in self.servers],
        }


# Single shared instance
sim = BackendSimulation()


# ─────────────────────────────────────────────────────────────
# Request Schemas
# ─────────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    scenario:    Optional[str] = Field("normal", description="normal | burst | adversarial")
    num_servers: Optional[int] = Field(None, ge=2, le=12)


class ResetRequest(BaseModel):
    scenario:    Optional[str] = "normal"
    num_servers: Optional[int] = Field(4, ge=2, le=12,
                                        description="Number of servers (2–12)")


class ServerState(BaseModel):
    CPU_utilization:       float
    Memory_utilization:    float
    Bandwidth_utilization: float
    Queue_pressure:        float
    Active_users:          int


class PredictRequest(BaseModel):
    servers: List[ServerState]


class FeedbackRequest(BaseModel):
    state:           List[ServerState]
    selected_server: int   = Field(..., ge=0)
    actual_rt:       float = Field(..., gt=0)


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status":      "ok",
        "message":     "NN-ACO Load Balancer is running",
        "step":        sim.step,
        "scenario":    sim.scenario,
        "num_servers": sim.num_servers,
    }


@app.post("/step", tags=["Simulation"])
def run_step(body: StepRequest):
    """
    Advance simulation by one step.
    Optionally pass num_servers to resize mid-run (triggers reset).
    """
    try:
        # Resize if frontend changed server count
        if body.num_servers and body.num_servers != sim.num_servers:
            logger.info(
                "Server count changed %d → %d, resetting simulation",
                sim.num_servers, body.num_servers
            )
            sim.reset(
                scenario=body.scenario or sim.scenario,
                num_servers=body.num_servers
            )

        # Update scenario without resetting
        if body.scenario and body.scenario != sim.scenario:
            sim.scenario = body.scenario
            logger.info(
                "Scenario updated to %s at step %d", sim.scenario, sim.step
            )

        result = sim.step_once()
        logger.info(
            "step=%d  servers=%d  selected=%d  probs=%s  avg_rt=%.1f",
            result["step"], sim.num_servers, result["selected_server"],
            [f"{p:.3f}" for p in result["probabilities"]],
            result["actual_rt"],
        )
        return result

    except Exception as exc:
        logger.error(
            "step error at step %d: %s", sim.step, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/reset", tags=["Simulation"])
def reset_sim(body: ResetRequest):
    """
    Reset simulation to step 0.
    Pass num_servers to set the server count for the new run.
    """
    n = body.num_servers if body.num_servers is not None else sim.num_servers
    sim.reset(scenario=body.scenario or "normal", num_servers=n)
    return {
        "status":      "reset",
        "scenario":    sim.scenario,
        "num_servers": sim.num_servers,
        "step":        0,
        "service_rates": [round(s.service_rate, 2) for s in sim.servers],
    }


@app.get("/state", tags=["Simulation"])
def get_state():
    """Return current simulation state without advancing."""
    return {
        "step":          sim.step,
        "scenario":      sim.scenario,
        "arrival_rate":  sim.arrival_rate,
        "num_servers":   sim.num_servers,
        "servers":       [s.to_dict() for s in sim.servers],
        "pheromones":    sim.aco.get_pheromone().tolist(),
        "last_selected": sim.last_selected,
        "last_probs":    sim.last_probs,
        "last_preds":    sim.last_preds,
    }


# ── Legacy endpoints ───────────────────────────────────────────

def _to_matrix(states: List[ServerState]) -> list:
    return [
        [s.CPU_utilization, s.Memory_utilization,
         s.Bandwidth_utilization, s.Queue_pressure,
         float(s.Active_users)]
        for s in states
    ]


@app.post("/predict", tags=["Legacy"])
def predict(body: PredictRequest):
    try:
        # Resize ACO if server count changed
        n = len(body.servers)
        if n != sim.num_servers:
            logger.warning(
                "/predict got %d servers but sim has %d — using sim.aco as-is",
                n, sim.num_servers
            )
        server, predicted_rt, probabilities = process_request(
            _to_matrix(body.servers), sim.aco
        )
        return {
            "selected_server": int(server),
            "predicted_rt":    predicted_rt.tolist(),
            "probabilities":   probabilities.tolist(),
        }
    except Exception as exc:
        logger.error("predict error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/feedback", tags=["Legacy"])
def feedback(body: FeedbackRequest):
    try:
        process_feedback(_to_matrix(body.state), body.selected_server, min(body.actual_rt, 2500.0), sim.aco)
        return {"status": "updated"}
    except Exception as exc:
        logger.error("feedback error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))