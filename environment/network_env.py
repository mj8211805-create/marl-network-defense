"""Gym-compatible Multi-Agent Network Defense Environment (Dec-POMDP)."""

import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from config import config
from environment.topology import NetworkTopology, NodeStatus
from environment.attacks import AttackType, ThreatVector
from environment.traffic_generator import TrafficGenerator, TrafficFlow


class MultiAgentNetworkDefenseEnv:
    """Decentralized Multi-Agent Environment for Autonomous Network Defense."""

    def __init__(self, max_steps: int = None):
        self.max_steps = max_steps or config.MAX_STEPS_PER_EPISODE
        self.topology = NetworkTopology()
        self.traffic_gen = TrafficGenerator(self.topology)
        
        self.current_step = 0
        self.current_attack: Optional[ThreatVector] = None
        
        # Statistics
        self.stats = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0,
            "total_attacks_generated": 0,
            "total_attacks_mitigated": 0,
            "availability_history": [],
            "reward_history": []
        }

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Resets topology, counters, and returns initial observations."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.topology.reset()
        self.current_step = 0
        self.current_attack = None

        self.stats = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0,
            "total_attacks_generated": 0,
            "total_attacks_mitigated": 0,
            "availability_history": [],
            "reward_history": []
        }

        _, obs = self.traffic_gen.generate_step_traffic(current_attack=None)
        return obs

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, Any]]:
        """Executes one simulation step: evaluates agent actions against stochastic threats."""
        self.current_step += 1

        # 1. Decide if attack occurs in this step
        if random.random() < config.ATTACK_OCCURRENCE_PROBABILITY:
            self.current_attack = ThreatVector.generate_random_attack()
            self.stats["total_attacks_generated"] += 1
        else:
            self.current_attack = ThreatVector.create_specific_attack(AttackType.BENIGN)

        # 2. Extract Agent Actions
        p_act = actions.get("perimeter", 0)
        i_act = actions.get("internal", 0)
        h_act = actions.get("host", 0)

        # 3. Evaluate Mitigations & Compute Rewards
        p_rew, i_rew, h_rew, step_info = self._evaluate_defense_actions(p_act, i_act, h_act)
        
        # Calculate shared global reward with QoS bonus
        health_ratio = self.topology.get_overall_health_ratio()
        qos_bonus = health_ratio * config.REWARD_QOS_COEFFICIENT
        self.stats["availability_history"].append(health_ratio)

        global_reward = (p_rew + i_rew + h_rew) / 3.0 + qos_bonus
        self.stats["reward_history"].append(global_reward)

        rewards = {
            "perimeter": p_rew + qos_bonus,
            "internal": i_rew + qos_bonus,
            "host": h_rew + qos_bonus,
            "global": global_reward
        }

        # 4. Generate Next Observations
        _, next_obs = self.traffic_gen.generate_step_traffic(self.current_attack)

        # 5. Check Termination
        done = self.current_step >= self.max_steps or health_ratio <= 0.20

        step_info.update({
            "step": self.current_step,
            "health_ratio": health_ratio,
            "active_attack": self.current_attack.attack_type.value,
            "stats": self.stats.copy(),
            "topology": self.topology.to_graph_data()
        })

        return next_obs, rewards, done, step_info

    def _evaluate_defense_actions(self, p_act: int, i_act: int, h_act: int) -> Tuple[float, float, float, Dict[str, Any]]:
        """Calculates tiered agent rewards based on action correctness."""
        atk_type = self.current_attack.attack_type
        is_attack = atk_type != AttackType.BENIGN

        p_rew, i_rew, h_rew = 0.0, 0.0, 0.0
        mitigated = False
        tp, fp, fn, tn = False, False, False, False

        # --- Tier 1: Perimeter Agent Evaluation ---
        if atk_type in [AttackType.DOS_SYN_FLOOD, AttackType.DDOS_HTTP_FLOOD, AttackType.PORT_SCAN_SYN]:
            if (atk_type == AttackType.DOS_SYN_FLOOD and p_act in [2, 3]) or \
               (atk_type == AttackType.DDOS_HTTP_FLOOD and p_act in [1, 2]) or \
               (atk_type == AttackType.PORT_SCAN_SYN and p_act in [1, 2]):
                p_rew += config.REWARD_MITIGATE_ATTACK
                mitigated = True
                tp = True
            else:
                p_rew += config.PENALTY_FALSE_NEGATIVE
                fn = True
                self._degrade_target_node(self.current_attack.target_node_id)
        elif not is_attack:
            if p_act != 0:  # Blocked benign traffic
                p_rew += config.PENALTY_FALSE_POSITIVE
                fp = True
            else:
                tn = True
        else:
            if p_act != 0:
                p_rew += config.PENALTY_ACTION_COST

        # --- Tier 2: Internal Agent Evaluation ---
        if atk_type == AttackType.APT_LATERAL_MOVEMENT:
            if i_act in [1, 2]:
                i_rew += config.REWARD_MITIGATE_ATTACK
                mitigated = True
                tp = True
            else:
                i_rew += config.PENALTY_FALSE_NEGATIVE
                fn = True
                self._compromise_target_node(self.current_attack.target_node_id)
        elif not is_attack:
            if i_act != 0:
                i_rew += config.PENALTY_FALSE_POSITIVE
                fp = True
            else:
                tn = True
        else:
            if i_act != 0:
                i_rew += config.PENALTY_ACTION_COST

        # --- Tier 3: Host Agent Evaluation ---
        if atk_type == AttackType.BRUTE_FORCE_SSH:
            if h_act in [1, 3]:
                h_rew += config.REWARD_MITIGATE_ATTACK
                mitigated = True
                tp = True
            else:
                h_rew += config.PENALTY_FALSE_NEGATIVE
                fn = True
                self._compromise_target_node(self.current_attack.target_node_id)
        elif not is_attack:
            if h_act != 0:
                h_rew += config.PENALTY_FALSE_POSITIVE
                fp = True
            else:
                tn = True
        else:
            if h_act != 0:
                h_rew += config.PENALTY_ACTION_COST

        # Update running tallies
        if tp: self.stats["true_positives"] += 1
        if fp: self.stats["false_positives"] += 1
        if fn: self.stats["false_negatives"] += 1
        if tn: self.stats["true_negatives"] += 1
        if mitigated: self.stats["total_attacks_mitigated"] += 1

        info = {
            "mitigated": mitigated,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "p_action": p_act,
            "i_action": i_act,
            "h_action": h_act
        }
        return p_rew, i_rew, h_rew, info

    def _degrade_target_node(self, node_id: str):
        node = self.topology.nodes.get(node_id)
        if node:
            node.status = NodeStatus.DEGRADED
            node.cpu_load = min(1.0, node.cpu_load + 0.5)

    def _compromise_target_node(self, node_id: str):
        node = self.topology.nodes.get(node_id)
        if node:
            node.status = NodeStatus.COMPROMISED
            node.compromise_level = 1.0
            node.cpu_load = 0.95
