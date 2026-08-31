"""Centralized Single-Agent Deep Q-Network Baseline."""

import numpy as np
from typing import Dict, Tuple, Optional
from config import config
from agents.marl.q_network import DQNAgent


class CentralizedSingleAgentDQN:
    """Monolithic single-agent RL baseline controlling all 3 tiers with joint action space."""

    def __init__(self, lr: float = None, gamma: float = None):
        self.joint_action_dim = config.PERIMETER_ACTION_DIM * config.INTERNAL_ACTION_DIM * config.HOST_ACTION_DIM  # 5 * 4 * 4 = 80
        self.agent = DQNAgent(
            name="CentralizedRLAgent",
            obs_dim=config.GLOBAL_STATE_DIM,
            action_dim=self.joint_action_dim,
            lr=lr or config.LEARNING_RATE,
            gamma=gamma or config.GAMMA
        )
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = config.EPSILON_DECAY

    def select_actions(self, observations: Dict[str, np.ndarray], explore: bool = True) -> Dict[str, int]:
        """Translates joint action scalar index [0..79] into individual tier actions."""
        global_obs = observations.get("global_state")
        if global_obs is None:
            global_obs = np.concatenate([observations["perimeter"], observations["internal"], observations["host"]])

        eps = self.epsilon if explore else 0.0
        joint_action = self.agent.select_action(global_obs, epsilon=eps)

        # Decompose joint index into [perimeter, internal, host]
        p_act = joint_action // 16
        rem = joint_action % 16
        i_act = rem // 4
        h_act = rem % 4

        return {
            "perimeter": int(p_act),
            "internal": int(i_act),
            "host": int(h_act)
        }

    def remember(
        self,
        obs: Dict[str, np.ndarray],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        done: bool
    ):
        global_obs = obs.get("global_state", np.concatenate([obs["perimeter"], obs["internal"], obs["host"]]))
        global_next_obs = next_obs.get("global_state", np.concatenate([next_obs["perimeter"], next_obs["internal"], next_obs["host"]]))

        # Reconstruct joint action scalar
        joint_action = actions["perimeter"] * 16 + actions["internal"] * 4 + actions["host"]
        self.agent.remember(global_obs, joint_action, rewards["global"], global_next_obs, done)

    def train_step(self, batch_size: int = None) -> Optional[float]:
        loss = self.agent.train_step(batch_size or config.BATCH_SIZE)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss
