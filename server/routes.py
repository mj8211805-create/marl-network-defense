"""REST API endpoints for live cyber defense simulation, training, and benchmarking."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from environment.network_env import MultiAgentNetworkDefenseEnv
from agents.marl.multi_agent_dql import MultiAgentDQNSystem
from agents.single_rl.centralized_dqn import CentralizedSingleAgentDQN
from agents.ml_baseline.supervised_ids import SupervisedMLDefenseAgent
from agents.ml_baseline.anomaly_detector import AnomalyDetectorDefenseAgent
from agents.rule_baseline.snort_rules import RuleBasedIDSAgent
from benchmark.evaluator import CyberMARLEvaluator
from benchmark.train_runner import train_marl_system
from benchmark.metrics import ComparativeBenchmarkReport

router = APIRouter(prefix="/api")

# Persistent instances
live_env = MultiAgentNetworkDefenseEnv(max_steps=500)
live_obs = live_env.reset()

marl_model = MultiAgentDQNSystem()
single_model = CentralizedSingleAgentDQN()
ml_model = SupervisedMLDefenseAgent()
anomaly_model = AnomalyDetectorDefenseAgent()
rule_model = RuleBasedIDSAgent()
evaluator = CyberMARLEvaluator()


class SimulationStepRequest(BaseModel):
    defense_method: str = "marl"  # marl, single_rl, supervised_ml, anomaly_detector, rule_based
    injected_attack: Optional[str] = None  # DoS_SYN_Flood, DDoS_HTTP_Flood, Port_Scan_SYN, Brute_Force_SSH, APT_Lateral_Movement, Benign


class TrainRequest(BaseModel):
    algorithm: str = "marl"
    episodes: int = 80
    steps_per_episode: int = 40


class BenchmarkRequest(BaseModel):
    episodes: int = 10
    steps_per_episode: int = 50
    scenario_name: str = "Enterprise_Threat_Stream"


@router.get("/health")
async def health_check():
    return {
        "status": "online",
        "service": "CyberMARL Defense Engine",
        "version": "1.0.0",
        "algorithms": ["marl", "single_rl", "supervised_ml", "anomaly_detector", "rule_based"]
    }


@router.get("/topology")
async def get_topology():
    return live_env.topology.to_graph_data()


@router.post("/simulate/reset")
async def reset_simulation():
    global live_obs
    live_obs = live_env.reset()
    return {
        "message": "Environment reset successfully",
        "topology": live_env.topology.to_graph_data(),
        "stats": live_env.stats
    }


@router.post("/simulate/step")
async def simulate_step(req: SimulationStepRequest):
    global live_obs

    # 1. Select Model
    if req.defense_method == "marl":
        actions = marl_model.select_actions(live_obs, explore=False)
    elif req.defense_method == "single_rl":
        actions = single_model.select_actions(live_obs, explore=False)
    elif req.defense_method == "supervised_ml":
        actions = ml_model.select_actions(live_obs)
    elif req.defense_method == "anomaly_detector":
        actions = anomaly_model.select_actions(live_obs)
    else:
        actions = rule_model.select_actions(live_obs)

    # 2. Step Environment
    next_obs, rewards, done, info = live_env.step(actions)
    live_obs = next_obs
    if done:
        live_obs = live_env.reset()

    # Friendly action names
    action_names = {
        "perimeter": ["PASS", "RATE_LIMIT", "BLOCK_IP_CIDR", "ENABLE_SYN_COOKIES", "DROP_ALL"][actions["perimeter"]],
        "internal": ["PASS", "ISOLATE_SUBNET", "TERMINATE_LATERAL", "HONEYPOT_REDIRECT"][actions["internal"]],
        "host": ["PASS", "QUARANTINE_HOST", "RESTART_SERVICE", "LOCK_CREDENTIALS"][actions["host"]]
    }

    return {
        "step": info["step"],
        "defense_method": req.defense_method,
        "actions_executed": action_names,
        "raw_actions": actions,
        "active_attack": info["active_attack"],
        "mitigated": info["mitigated"],
        "health_ratio": info["health_ratio"],
        "rewards": rewards,
        "classification": {
            "tp": info["tp"], "fp": info["fp"], "fn": info["fn"], "tn": info["tn"]
        },
        "stats": info["stats"],
        "topology": info["topology"]
    }


@router.post("/train/run")
async def run_training(req: TrainRequest):
    if req.algorithm == "marl":
        trained_marl, history = train_marl_system(episodes=req.episodes, steps_per_episode=req.steps_per_episode)
        global marl_model
        marl_model = trained_marl
        return {
            "algorithm": "marl",
            "episodes": req.episodes,
            "final_reward": history["episode_rewards"][-1] if history["episode_rewards"] else 0.0,
            "final_mitigation_rate": history["mitigation_rates"][-1] if history["mitigation_rates"] else 0.0,
            "history": history
        }
    else:
        raise HTTPException(status_code=400, detail="Only 'marl' currently supported for live training run")


@router.post("/benchmark/run")
async def run_benchmark(req: BenchmarkRequest):
    report = evaluator.run_benchmark(
        episodes=req.episodes,
        steps_per_episode=req.steps_per_episode,
        scenario_name=req.scenario_name
    )
    return report.model_dump(mode="json")
