"""Unit tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "marl" in data["algorithms"]


def test_api_topology():
    res = client.get("/api/topology")
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert len(data["nodes"]) >= 8


def test_api_simulate_step():
    res = client.post("/api/simulate/step", json={"defense_method": "marl"})
    assert res.status_code == 200
    data = res.json()
    assert "step" in data
    assert "actions_executed" in data
    assert "health_ratio" in data


def test_api_benchmark_run():
    res = client.post("/api/benchmark/run", json={"episodes": 2, "steps_per_episode": 10})
    assert res.status_code == 200
    data = res.json()
    assert "marl_system" in data
    assert "supervised_ml_baseline" in data
