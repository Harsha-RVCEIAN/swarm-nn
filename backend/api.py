import asyncio
import logging
import math
import random
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from backend.controller import process_request, process_feedback
from aco.aco import ACO

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NN-ACO")

app = FastAPI(
    title="NN-ACO Load Balancer API",
    description="Neural Network + Ant Colony Optimization Load Balancer",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus ────────────────────────────────────────────────
SERVER_NAMES_POOL = ["Alpha","Beta","Gamma","Delta","Epsilon","Zeta","Eta","Theta","Iota","Kappa","Lambda","Mu"]

def _srv_name(i): return SERVER_NAMES_POOL[i] if i < len(SERVER_NAMES_POOL) else f"srv{i}"

REQUESTS_ROUTED  = Counter("nnaco_requests_routed_total",   "Total requests routed",          ["server_id","server_name"])
COOLDOWN_EVENTS  = Counter("nnaco_cooldown_events_total",   "Times server entered cooldown",  ["server_id"])
DEGRADED_EVENTS  = Counter("nnaco_degraded_events_total",   "Times server was degraded",      ["server_id"])
RETRAIN_COUNTER  = Counter("nnaco_retrain_total",           "NN retraining events")

AVG_RESPONSE_TIME = Gauge("nnaco_avg_response_time_ms",   "Rolling avg response time (ms)")
LOAD_VARIANCE     = Gauge("nnaco_load_variance",           "Variance of queue lengths")
ROUTING_ENTROPY   = Gauge("nnaco_routing_entropy_bits",    "Shannon entropy of routing probs")
AVG_REGRET        = Gauge("nnaco_avg_regret_ms",           "Avg regret vs optimal routing")
ARRIVAL_RATE      = Gauge("nnaco_arrival_rate",            "Poisson arrival rate lambda")
SIM_STEP          = Gauge("nnaco_simulation_step",         "Current simulation step")
QUEUE_LENGTH      = Gauge("nnaco_server_queue_length",     "Queue length per server",         ["server_id","server_name"])
PHEROMONE_LEVEL   = Gauge("nnaco_pheromone_level",         "Pheromone level per server",      ["server_id","server_name"])
ROUTE_PROBABILITY = Gauge("nnaco_route_probability",       "Routing probability per server",  ["server_id","server_name"])
SERVER_STATUS     = Gauge("nnaco_server_status",           "1=online 0=cooldown -1=degraded", ["server_id","server_name"])
TOTAL_REQS_HANDLED= Gauge("nnaco_server_total_requests",   "Total requests handled",          ["server_id","server_name"])
LAST_RT           = Gauge("nnaco_server_last_rt_ms",       "Last response time per server",   ["server_id","server_name"])
SERVICE_RATE      = Gauge("nnaco_server_service_rate",     "Service rate per server",         ["server_id","server_name"])

RESPONSE_TIME_HIST = Histogram(
    "nnaco_response_time_histogram_ms", "Response time distribution",
    buckets=[50,100,200,300,500,750,1000,1500,2000,3000,5000]
)

def _push_metrics(result, pheromones, probs):
    AVG_RESPONSE_TIME.set(result["actual_rt"])
    LOAD_VARIANCE.set(result["variance"])
    ROUTING_ENTROPY.set(result["entropy"])
    AVG_REGRET.set(result["regret"])
    ARRIVAL_RATE.set(result["arrival_rate"])
    SIM_STEP.set(result["step"])
    if result["actual_rt"] > 0:
        RESPONSE_TIME_HIST.observe(result["actual_rt"])
    sel = result["selected_server"]
    REQUESTS_ROUTED.labels(str(sel), _srv_name(sel)).inc()
    for s in result["servers"]:
        i, labels = s["id"], [str(s["id"]), _srv_name(s["id"])]
        is_cool = pheromones[i] <= 0.05 if i < len(pheromones) else False
        QUEUE_LENGTH.labels(*labels).set(s["queue_len"])
        TOTAL_REQS_HANDLED.labels(*labels).set(s["total_reqs"])
        SERVICE_RATE.labels(*labels).set(s["service_rate"])
        if s["last_rt"] is not None:
            LAST_RT.labels(*labels).set(s["last_rt"])
        SERVER_STATUS.labels(*labels).set(-1 if s["degraded"] else (0 if is_cool else 1))
    for i, (ph, pr) in enumerate(zip(pheromones, probs)):
        labels = [str(i), _srv_name(i)]
        PHEROMONE_LEVEL.labels(*labels).set(ph)
        ROUTE_PROBABILITY.labels(*labels).set(pr)

# ── Service rates ─────────────────────────────────────────────
BASE_SERVICE_RATES = [1.8,1.2,2.2,0.9,1.5,2.0,1.0,1.7,1.3,2.1,0.8,1.6]
def get_service_rate(idx): return BASE_SERVICE_RATES[idx % len(BASE_SERVICE_RATES)]

# ── SimServer ─────────────────────────────────────────────────
class SimServer:
    def __init__(self, idx, service_rate):
        self.id, self.service_rate = idx, service_rate
        self.queue, self.total_rts = [], []
        self.last_rt, self.degraded, self.total_requests = None, False, 0

    def to_state_row(self):
        q = len(self.queue)
        return [min(100.0,(q/10.0)*100.0), min(100.0,40.0+q*3.0),
                min(1000.0,100.0+q*30.0), min(5000.0,q*120.0), float(q*25)]

    def to_dict(self):
        return {"id":self.id, "service_rate":round(self.service_rate,4),
                "queue_len":len(self.queue),
                "last_rt":round(self.last_rt,2) if self.last_rt is not None else None,
                "total_reqs":self.total_requests, "degraded":self.degraded}

# ── BackendSimulation ─────────────────────────────────────────
class BackendSimulation:
    def __init__(self):
        self.num_servers = 4
        self.reset(num_servers=4)

    def reset(self, scenario="normal", num_servers=4):
        num_servers = max(2, min(num_servers, 12))
        self.num_servers = num_servers
        self.servers = [SimServer(i, get_service_rate(i)) for i in range(num_servers)]
        self.aco     = ACO(num_servers=num_servers)
        self.step, self.arrival_rate, self.scenario = 0, 1.0, scenario
        self.cum_regret, self.completed_rts, self.reg_history = 0.0, [], []
        self.last_selected   = -1
        self.last_probs      = [1.0/num_servers]*num_servers
        self.last_preds      = [500.0]*num_servers
        self.last_pheromones = [1.0]*num_servers
        logger.info("Reset — scenario=%s  n=%d  rates=%s", scenario, num_servers,
                    [round(s.service_rate,2) for s in self.servers])

    def _state_matrix(self):
        return np.array([s.to_state_row() for s in self.servers], dtype=np.float32)

    @staticmethod
    def _poisson(lam):
        lam = min(lam, 20.0)
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            k += 1; p *= random.random()
            if p <= L: return max(0, k-1)

    def step_once(self):
        if self.scenario == "burst":
            self.arrival_rate = 3.0 if 100 <= self.step < 150 else 1.0
        if self.scenario == "adversarial" and self.step == 100:
            target = max(self.servers, key=lambda s: s.service_rate)
            target.service_rate = max(0.05, target.service_rate*0.1)
            target.degraded = True
            DEGRADED_EVENTS.labels(str(target.id)).inc()
            logger.warning("Server %d degraded (adversarial)", target.id)

        state = self._state_matrix()
        try:
            selected_server, predicted_rt, probabilities = process_request(state, self.aco)
        except Exception as e:
            logger.error("process_request failed: %s", e, exc_info=True)
            selected_server = np.random.randint(0, self.num_servers)
            predicted_rt    = np.full(self.num_servers, 500.0)
            probabilities   = np.ones(self.num_servers)/self.num_servers

        selected_server   = int(selected_server)
        predicted_rt_list = predicted_rt.tolist()
        probs_list        = probabilities.tolist()
        pheromones_list   = self.aco.get_pheromone().tolist()
        self.last_selected, self.last_probs = selected_server, probs_list
        self.last_preds, self.last_pheromones = predicted_rt_list, pheromones_list

        p_arr   = np.array(probs_list) + 1e-12
        entropy = float(-np.sum(p_arr * np.log2(p_arr)))

        n, best_pred = self._poisson(self.arrival_rate), float(np.min(predicted_rt))
        for _ in range(n):
            self.servers[selected_server].queue.append({"arrival":self.step,"best_pred":best_pred})

        prev_ph = self.aco.get_pheromone().copy()
        for i, s in enumerate(self.servers):
            if not s.queue: continue
            req      = s.queue.pop(0)
            svc_time = (1.0/s.service_rate)*(0.5+random.random())
            actual_rt = svc_time*400.0 + len(s.queue)*10.0 + random.random()*50.0
            if s.degraded: actual_rt *= 4.0
            actual_rt   = float(np.clip(actual_rt, 50.0, 5000.0))
            feedback_rt = min(actual_rt, 2500.0)
            try:
                process_feedback(state, i, feedback_rt, self.aco)
            except Exception as e:
                logger.error("process_feedback failed server %d: %s", i, e, exc_info=True)
            new_ph = self.aco.get_pheromone()
            if prev_ph[i] > 0.05 and new_ph[i] <= 0.05:
                COOLDOWN_EVENTS.labels(str(i)).inc()
            prev_ph = new_ph.copy()
            s.total_rts.append(actual_rt); s.last_rt = actual_rt
            s.total_requests += 1; self.completed_rts.append(actual_rt)
            self.cum_regret += max(0.0, actual_rt - req["best_pred"])
            self.reg_history.append(self.cum_regret/len(self.completed_rts))

        avg_rt   = float(np.mean(self.completed_rts)) if self.completed_rts else 0.0
        variance = float(np.var([len(s.queue) for s in self.servers]))
        avg_reg  = self.reg_history[-1] if self.reg_history else 0.0
        self.step += 1

        result = {
            "selected_server": selected_server, "probabilities": probs_list,
            "predicted_rt": predicted_rt_list,  "pheromones": pheromones_list,
            "actual_rt": round(avg_rt,2),        "variance": round(variance,4),
            "entropy": round(entropy,4),          "regret": round(avg_reg,2),
            "arrival_rate": self.arrival_rate,    "step": self.step,
            "num_servers": self.num_servers,      "servers": [s.to_dict() for s in self.servers],
        }
        _push_metrics(result, pheromones_list, probs_list)
        return result

sim = BackendSimulation()

# ── Request Schemas ───────────────────────────────────────────
class StepRequest(BaseModel):
    scenario:    Optional[str] = Field("normal")
    num_servers: Optional[int] = Field(None, ge=2, le=12)

class ResetRequest(BaseModel):
    scenario:    Optional[str] = "normal"
    num_servers: Optional[int] = Field(4, ge=2, le=12)

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

# ── REST Endpoints ────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status":"ok","step":sim.step,"scenario":sim.scenario,"num_servers":sim.num_servers}

@app.post("/step", tags=["Simulation"])
def run_step(body: StepRequest):
    try:
        if body.num_servers and body.num_servers != sim.num_servers:
            sim.reset(scenario=body.scenario or sim.scenario, num_servers=body.num_servers)
        if body.scenario and body.scenario != sim.scenario:
            sim.scenario = body.scenario
        result = sim.step_once()
        logger.info("step=%d  servers=%d  selected=%d  avg_rt=%.1f",
                    result["step"], sim.num_servers, result["selected_server"], result["actual_rt"])
        return result
    except Exception as exc:
        logger.error("step error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/reset", tags=["Simulation"])
def reset_sim(body: ResetRequest):
    n = body.num_servers if body.num_servers is not None else sim.num_servers
    sim.reset(scenario=body.scenario or "normal", num_servers=n)
    return {"status":"reset","scenario":sim.scenario,"num_servers":sim.num_servers,
            "step":0,"service_rates":[round(s.service_rate,2) for s in sim.servers]}

@app.get("/state", tags=["Simulation"])
def get_state():
    return {"step":sim.step,"scenario":sim.scenario,"arrival_rate":sim.arrival_rate,
            "num_servers":sim.num_servers,"servers":[s.to_dict() for s in sim.servers],
            "pheromones":sim.aco.get_pheromone().tolist(),"last_selected":sim.last_selected,
            "last_probs":sim.last_probs,"last_preds":sim.last_preds}

def _to_matrix(states):
    return [[s.CPU_utilization,s.Memory_utilization,s.Bandwidth_utilization,
             s.Queue_pressure,float(s.Active_users)] for s in states]

@app.post("/predict", tags=["Legacy"])
def predict(body: PredictRequest):
    try:
        server, predicted_rt, probabilities = process_request(_to_matrix(body.servers), sim.aco)
        return {"selected_server":int(server),"predicted_rt":predicted_rt.tolist(),
                "probabilities":probabilities.tolist()}
    except Exception as exc:
        logger.error("predict error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/feedback", tags=["Legacy"])
def feedback(body: FeedbackRequest):
    try:
        process_feedback(_to_matrix(body.state), body.selected_server,
                         min(body.actual_rt,2500.0), sim.aco)
        return {"status":"updated"}
    except Exception as exc:
        logger.error("feedback error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

# ── WebSocket ─────────────────────────────────────────────────
SPEED_DELAYS = {1:0.30, 2:0.15, 3:0.08, 4:0.04, 5:0.02, 6:0.008}

@app.websocket("/ws/simulate")
async def ws_simulate(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

    paused    = True
    num_steps = 200
    delay     = SPEED_DELAYS[3]
    sim_task  = None

    async def run_loop():
        nonlocal paused, num_steps
        while True:
            if paused:
                await asyncio.sleep(0.05)
                continue
            if sim.step >= num_steps:
                try:
                    await websocket.send_json({"type":"done","step":sim.step})
                except Exception:
                    return
                paused = True
                continue
            try:
                result = sim.step_once()
            except Exception as exc:
                logger.error("ws step error: %s", exc, exc_info=True)
                try:
                    await websocket.send_json({"type":"error","detail":str(exc)})
                except Exception:
                    return
                paused = True
                continue
            try:
                await websocket.send_json({"type":"step", **result})
            except (WebSocketDisconnect, Exception):
                return   # client gone, exit cleanly
            await asyncio.sleep(delay)

    try:
        sim_task = asyncio.create_task(run_loop())

        while True:
            raw = await websocket.receive_json()
            cmd = raw.get("cmd","")

            if cmd == "start":
                paused = True
                await asyncio.sleep(0.1)
                sim.reset(scenario=raw.get("scenario", sim.scenario),
                          num_servers=raw.get("num_servers", sim.num_servers))
                num_steps = int(raw.get("num_steps", 200))
                delay     = SPEED_DELAYS.get(int(raw.get("speed", 3)), 0.08)
                await websocket.send_json({
                    "type":"reset","status":"reset","scenario":sim.scenario,
                    "num_servers":sim.num_servers,"step":0,
                    "service_rates":[round(s.service_rate,2) for s in sim.servers],
                })
                paused = False

            elif cmd == "pause":
                paused = True
                await websocket.send_json({"type":"paused"})

            elif cmd == "resume":
                if sim.step < num_steps:
                    paused = False
                    await websocket.send_json({"type":"resumed"})

            elif cmd == "reset":
                paused = True
                sim.reset(scenario=raw.get("scenario","normal"),
                          num_servers=raw.get("num_servers", sim.num_servers))
                await websocket.send_json({
                    "type":"reset","status":"reset","scenario":sim.scenario,
                    "num_servers":sim.num_servers,"step":0,
                    "service_rates":[round(s.service_rate,2) for s in sim.servers],
                })

            elif cmd == "speed":
                delay = SPEED_DELAYS.get(int(raw.get("value", 3)), 0.08)

            else:
                logger.warning("Unknown WS command: %s", cmd)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc, exc_info=True)
        try:
            await websocket.send_json({"type":"error","detail":str(exc)})
        except Exception:
            pass
    finally:
        if sim_task and not sim_task.done():
            sim_task.cancel()
        logger.info("WebSocket session ended")

# ── Prometheus /metrics ───────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)