"""Unit tests for Vectorized DQN, ReplayBuffer, and MultiAgentDQNSystem."""

import numpy as np
from agents.marl.q_network import VectorizedDQN, ReplayBuffer, DQNAgent
from agents.marl.multi_agent_dql import MultiAgentDQNSystem


def test_vectorized_dqn_forward_and_train():
    net = VectorizedDQN(input_dim=8, output_dim=4, hidden_dim=32, lr=0.01)
    
    # Test Forward Pass
    dummy_obs = np.random.randn(5, 8).astype(np.float32)
    q_vals = net.forward(dummy_obs)
    assert q_vals.shape == (5, 4)

    # Test Train Step
    actions = np.array([0, 1, 2, 3, 0], dtype=np.int64)
    targets = np.array([1.5, 2.0, 0.5, -1.0, 3.0], dtype=np.float32)
    loss = net.train_step(dummy_obs, actions, targets)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_replay_buffer_capacity_and_sampling():
    buf = ReplayBuffer(capacity=50, obs_dim=8)
    for i in range(20):
        obs = np.zeros(8, dtype=np.float32) + i
        buf.store(obs, 1, float(i), obs, False)

    assert buf.size == 20
    b_obs, b_act, b_rew, b_next, b_done = buf.sample_batch(10)
    assert b_obs.shape == (10, 8)
    assert b_act.shape == (10,)
    assert b_rew.shape == (10,)


def test_multi_agent_dqn_system_step():
    marl = MultiAgentDQNSystem()
    dummy_obs = {
        "perimeter": np.zeros(8, dtype=np.float32),
        "internal": np.zeros(8, dtype=np.float32),
        "host": np.zeros(8, dtype=np.float32)
    }

    actions = marl.select_actions(dummy_obs, explore=False)
    assert "perimeter" in actions
    assert "internal" in actions
    assert "host" in actions
    assert 0 <= actions["perimeter"] < 5
    assert 0 <= actions["internal"] < 4
    assert 0 <= actions["host"] < 4
