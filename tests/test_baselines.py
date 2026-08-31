"""Unit tests for Centralized RL, Supervised ML, Anomaly Detection, and Rule IDS."""

import numpy as np
from agents.single_rl.centralized_dqn import CentralizedSingleAgentDQN
from agents.ml_baseline.supervised_ids import SupervisedMLDefenseAgent
from agents.ml_baseline.anomaly_detector import AnomalyDetectorDefenseAgent
from agents.rule_baseline.snort_rules import RuleBasedIDSAgent


def test_centralized_single_agent():
    agent = CentralizedSingleAgentDQN()
    dummy_obs = {
        "perimeter": np.zeros(8, dtype=np.float32),
        "internal": np.zeros(8, dtype=np.float32),
        "host": np.zeros(8, dtype=np.float32),
        "global_state": np.zeros(24, dtype=np.float32)
    }

    actions = agent.select_actions(dummy_obs, explore=False)
    assert 0 <= actions["perimeter"] < 5
    assert 0 <= actions["internal"] < 4
    assert 0 <= actions["host"] < 4


def test_supervised_ml_baseline():
    agent = SupervisedMLDefenseAgent()
    assert agent.is_trained is True

    dummy_obs = {
        "perimeter": np.array([0.9, 0.95, 0.8, 0.6, 0.4, 0.8, 1.0, 1.0], dtype=np.float32),
        "internal": np.zeros(8, dtype=np.float32),
        "host": np.zeros(8, dtype=np.float32)
    }
    actions = agent.select_actions(dummy_obs)
    assert actions["perimeter"] in [2, 3]  # Should trigger SYN cookies or Block on flood


def test_anomaly_detector_and_rule_baseline():
    anomaly_agent = AnomalyDetectorDefenseAgent()
    rule_agent = RuleBasedIDSAgent()

    dummy_obs = {
        "perimeter": np.zeros(8, dtype=np.float32),
        "internal": np.zeros(8, dtype=np.float32),
        "host": np.zeros(8, dtype=np.float32)
    }

    act_anom = anomaly_agent.select_actions(dummy_obs)
    act_rule = rule_agent.select_actions(dummy_obs)

    assert "perimeter" in act_anom
    assert "perimeter" in act_rule
