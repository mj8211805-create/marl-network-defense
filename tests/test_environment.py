"""Unit tests for Network Topology, Attacks, and Simulation Environment."""

import numpy as np
from environment.topology import NetworkTopology, NodeStatus
from environment.attacks import AttackType, ThreatVector
from environment.traffic_generator import TrafficGenerator
from environment.network_env import MultiAgentNetworkDefenseEnv


def test_network_topology_initialization():
    topo = NetworkTopology()
    assert len(topo.nodes) >= 8
    assert "gw-01" in topo.nodes
    assert "dmz-web-01" in topo.nodes
    assert "db-primary-01" in topo.nodes
    assert topo.get_overall_health_ratio() == 1.0


def test_traffic_generator_observations():
    topo = NetworkTopology()
    gen = TrafficGenerator(topo)
    flows, obs = gen.generate_step_traffic(current_attack=None)

    assert len(flows) > 0
    assert "perimeter" in obs
    assert "internal" in obs
    assert "host" in obs
    assert obs["perimeter"].shape == (8,)
    assert obs["internal"].shape == (8,)
    assert obs["host"].shape == (8,)


def test_network_env_step_cycle():
    env = MultiAgentNetworkDefenseEnv(max_steps=20)
    obs = env.reset(seed=42)
    assert "perimeter" in obs

    actions = {"perimeter": 0, "internal": 0, "host": 0}
    next_obs, rewards, done, info = env.step(actions)

    assert "perimeter" in next_obs
    assert "global" in rewards
    assert isinstance(done, bool)
    assert "health_ratio" in info
    assert "active_attack" in info
