"""Training orchestrator for Multi-Agent and Single-Agent Reinforcement Learning."""

import time
import numpy as np
from typing import Dict, List, Any, Tuple
from environment.network_env import MultiAgentNetworkDefenseEnv
from agents.marl.multi_agent_dql import MultiAgentDQNSystem
from agents.single_rl.centralized_dqn import CentralizedSingleAgentDQN


def train_marl_system(
    episodes: int = 150,
    steps_per_episode: int = 60,
    save_weights_path: str = None
) -> Tuple[MultiAgentDQNSystem, Dict[str, List[float]]]:
    """Trains the cooperative Multi-Agent Deep Q-Learning system."""
    env = MultiAgentNetworkDefenseEnv(max_steps=steps_per_episode)
    marl = MultiAgentDQNSystem()

    history = {
        "episode_rewards": [],
        "mitigation_rates": [],
        "availability_scores": [],
        "epsilons": []
    }

    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0.0

        for st in range(steps_per_episode):
            actions = marl.select_actions(obs, explore=True)
            next_obs, rewards, done, info = env.step(actions)

            marl.remember(obs, actions, rewards, next_obs, done)
            marl.train_step()

            ep_reward += rewards["global"]
            obs = next_obs
            if done:
                break

        # Record metrics
        mit_rate = env.stats["total_attacks_mitigated"] / max(1, env.stats["total_attacks_generated"])
        avg_avail = np.mean(env.stats["availability_history"]) if env.stats["availability_history"] else 1.0

        history["episode_rewards"].append(float(ep_reward))
        history["mitigation_rates"].append(float(mit_rate))
        history["availability_scores"].append(float(avg_avail * 100.0))
        history["epsilons"].append(float(marl.epsilon))

    if save_weights_path:
        marl.save(save_weights_path)

    return marl, history


def train_single_agent_system(
    episodes: int = 150,
    steps_per_episode: int = 60
) -> Tuple[CentralizedSingleAgentDQN, Dict[str, List[float]]]:
    """Trains the monolithic Centralized Single-Agent baseline."""
    env = MultiAgentNetworkDefenseEnv(max_steps=steps_per_episode)
    single_agent = CentralizedSingleAgentDQN()

    history = {
        "episode_rewards": [],
        "mitigation_rates": [],
        "availability_scores": []
    }

    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0.0

        for st in range(steps_per_episode):
            actions = single_agent.select_actions(obs, explore=True)
            next_obs, rewards, done, info = env.step(actions)

            single_agent.remember(obs, actions, rewards, next_obs, done)
            single_agent.train_step()

            ep_reward += rewards["global"]
            obs = next_obs
            if done:
                break

        mit_rate = env.stats["total_attacks_mitigated"] / max(1, env.stats["total_attacks_generated"])
        avg_avail = np.mean(env.stats["availability_history"]) if env.stats["availability_history"] else 1.0

        history["episode_rewards"].append(float(ep_reward))
        history["mitigation_rates"].append(float(mit_rate))
        history["availability_scores"].append(float(avg_avail * 100.0))

    return single_agent, history
