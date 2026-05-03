# backend/api.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from backend.controller import process_request, process_feedback

app = FastAPI(title="NN-ACO Load Balancer API")


# -----------------------------
# Request Schemas
# -----------------------------

class ServerState(BaseModel):
    CPU_utilization: float
    Memory_utilization: float
    Bandwidth_utilization: float
    Queue_pressure: float
    Active_users: int


class RequestInput(BaseModel):
    servers: List[ServerState]


class FeedbackInput(BaseModel):
    state: List[ServerState]
    selected_server: int
    actual_rt: float


# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/")
def root():
    return {"message": "NN-ACO Load Balancer is running"}


# -----------------------------
# Predict + Select Server
# -----------------------------

@app.post("/predict")
def predict(input_data: RequestInput):
    """
    Input:
        List of server states

    Output:
        Selected server
        Predicted RT
        Probabilities
    """

    # Convert to matrix
    state_matrix = [
        [
            s.CPU_utilization,
            s.Memory_utilization,
            s.Bandwidth_utilization,
            s.Queue_pressure,
            s.Active_users
        ]
        for s in input_data.servers
    ]

    server, predicted_rt, probabilities = process_request(state_matrix)

    return {
        "selected_server": int(server),
        "predicted_rt": predicted_rt.tolist(),
        "probabilities": probabilities.tolist()
    }


# -----------------------------
# Feedback endpoint
# -----------------------------

@app.post("/feedback")
def feedback(data: FeedbackInput):
    """
    Input:
        previous state
        selected server
        actual response time
    """

    state_matrix = [
        [
            s.CPU_utilization,
            s.Memory_utilization,
            s.Bandwidth_utilization,
            s.Queue_pressure,
            s.Active_users
        ]
        for s in data.state
    ]

    process_feedback(
        state_matrix,
        data.selected_server,
        data.actual_rt
    )

    return {"status": "updated"}