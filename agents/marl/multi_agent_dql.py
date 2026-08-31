"""Multi-Agent Deep Q-Learning (MADQN) System with Cooperative Value Sharing."""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from config import config
from agents.marl.q_network import DQNAgent


class MultiAgentDQNSystem:
    """Cooperative Multi-Agent Reinforcement Learning defense coordinator."""

    def __init__(self, lr: float = None, gamma: float = None):
        lr = lr or config.LEARNING_RATE
        gamma = gamma or config.GAMMA

        self.agents: Dict[str, DQNAgent] = {
            "perimeter": DQNAgent("PerimeterAgent", config.PERIMETER_OBS_DIM, config.PERIMETER_ACTION_DIM, lr, gamma),
            "internal": DQNAgent("InternalAgent", config.INTERNAL_OBS_DIM, config.INTERNAL_ACTION_DIM, lr, gamma),
            "host": DQNAgent("HostAgent", config.HOST_OBS_DIM, config.HOST_ACTION_DIM, lr, gamma)
        }

        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = config.EPSILON_DECAY

    def select_actions(self, observations: Dict[str, np.ndarray], explore: bool = True) -> Dict[str, int]:
        """Selects decentralized actions across all three defensive tiers."""
        eps = self.epsilon if explore else 0.0
        return {
            tier: agent.select_action(observations[tier], epsilon=eps)
            for tier, agent in self.agents.items()
        }

    def remember(
        self,
        obs: Dict[str, np.ndarray],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        done: bool
    ):
        """Stores local transitions in each agent's respective replay memory."""
        for tier, agent in self.agents.items():
            # Blend local reward with shared global team reward for cooperation
            team_reward = 0.7 * rewards[tier] + 0.3 * rewards["global"]
            agent.remember(obs[tier], actions[tier], team_reward, next_obs[tier], done)

    def train_step(self, batch_size: int = None) -> Dict[str, Optional[float]]:
        """Executes a training step across all three agent Q-networks."""
        batch_size = batch_size or config.BATCH_SIZE
        losses = {}
        for tier, agent in self.agents.items():
            losses[tier] = agent.train_step(batch_size)

        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return losses

    def save(self, save_dir: str):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        for tier, agent in self.agents.items():
            weights = {
                "W1": agent.eval_net.W1, "b1": agent.eval_net.b1,
                "W2": agent.eval_net.W2, "b2": agent.eval_net.b2,
                "W3": agent.eval_net.W3, "b3": agent.eval_net.b3
            }
            with open(Path(save_dir) / f"{tier}_agent.pkl", "wb") as f:
                pickle.dump(weights, f)

    def load(self, save_dir: str):
        for tier, agent in self.agents.items():
            p = Path(save_dir) / f"{tier}_agent.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    weights = pickle.load(f)
                agent.eval_net.W1 = weights["W1"]
                agent.eval_net.b1 = weights["b1"]
                agent.eval_net.W2 = weights["W2"]
                agent.eval_net.b2 = weights["b2"]
                agent.eval_net.W3 = weights["W3"]
                agent.eval_net.b3 = weights["b3"]
                agent.target_net.copy_weights_from(agent.eval_net)
