"""Stochastic Poisson background traffic and attack flow generator."""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from environment.attacks import AttackType, ThreatVector
from environment.topology import NetworkTopology, NodeStatus


@dataclass
class TrafficFlow:
    source_ip: str
    destination_ip: str
    target_node_id: str
    port: int
    protocol: str
    packet_rate: float
    syn_ack_ratio: float
    bytes_transferred: float
    error_rate: float
    auth_failures: int
    payload_entropy: float
    is_attack: bool
    attack_type: AttackType


class TrafficGenerator:
    """Generates continuous stochastic enterprise network traffic."""

    def __init__(self, topology: NetworkTopology):
        self.topology = topology

    def generate_step_traffic(
        self,
        current_attack: Optional[ThreatVector] = None,
        base_rate: float = 100.0
    ) -> Tuple[List[TrafficFlow], Dict[str, np.ndarray]]:
        """Generates network flows and returns structured observations for each tier."""
        flows: List[TrafficFlow] = []

        # 1. Generate Benign Background Flows (Poisson distributed)
        num_benign_flows = np.random.poisson(lam=8)
        for _ in range(num_benign_flows):
            src_node = random.choice(list(self.topology.nodes.values()))
            dst_node = random.choice(list(self.topology.nodes.values()))
            if src_node.node_id == dst_node.node_id:
                continue

            rate = np.random.exponential(scale=base_rate / 8.0)
            flow = TrafficFlow(
                source_ip=src_node.ip_address,
                destination_ip=dst_node.ip_address,
                target_node_id=dst_node.node_id,
                port=random.choice([80, 443, 8080, 445, 53]),
                protocol=random.choice(["TCP", "UDP", "TLS"]),
                packet_rate=rate,
                syn_ack_ratio=random.uniform(0.1, 0.4),
                bytes_transferred=rate * random.uniform(200, 1400),
                error_rate=random.uniform(0.0, 0.05),
                auth_failures=0,
                payload_entropy=random.uniform(4.0, 6.2),
                is_attack=False,
                attack_type=AttackType.BENIGN
            )
            flows.append(flow)

        # 2. Inject Threat Vector if active
        if current_attack and current_attack.attack_type != AttackType.BENIGN:
            target_node = self.topology.nodes.get(current_attack.target_node_id)
            target_ip = target_node.ip_address if target_node else "10.0.1.10"

            attack_flow = TrafficFlow(
                source_ip=current_attack.source_ip,
                destination_ip=target_ip,
                target_node_id=current_attack.target_node_id,
                port=current_attack.target_port,
                protocol="TCP" if "SYN" in current_attack.attack_type.value else "HTTP",
                packet_rate=current_attack.packet_rate,
                syn_ack_ratio=current_attack.syn_ack_ratio,
                bytes_transferred=current_attack.packet_rate * random.uniform(500, 1500),
                error_rate=random.uniform(0.15, 0.8),
                auth_failures=current_attack.failed_auth_count,
                payload_entropy=current_attack.payload_entropy,
                is_attack=True,
                attack_type=current_attack.attack_type
            )
            flows.append(attack_flow)

        # 3. Extract Tier-Specific Feature Vectors
        observations = self._extract_tier_observations(flows, current_attack)
        return flows, observations

    def _extract_tier_observations(
        self,
        flows: List[TrafficFlow],
        current_attack: Optional[ThreatVector]
    ) -> Dict[str, np.ndarray]:
        """Calculates normalized feature vectors (dim=8 each) for Perimeter, Internal, and Host tiers."""
        
        # Aggregate flow metrics
        total_pps = sum(f.packet_rate for f in flows)
        avg_syn_ratio = np.mean([f.syn_ack_ratio for f in flows]) if flows else 0.2
        avg_entropy = np.mean([f.payload_entropy for f in flows]) if flows else 5.0
        total_errors = sum(f.error_rate for f in flows)
        total_auth_fails = sum(f.auth_failures for f in flows)

        is_under_attack = 1.0 if (current_attack and current_attack.attack_type != AttackType.BENIGN) else 0.0

        # Tier 1: Perimeter Gateway Observation (dim=8)
        # Features: [Norm PPS, SYN Ratio, External IP Rate, Error Rate, Entropy/8, Bandwidth Load, Volumetric Anomaly, Threat Flag]
        perimeter_obs = np.array([
            min(1.0, total_pps / 2500.0),
            min(1.0, avg_syn_ratio),
            min(1.0, sum(1 for f in flows if not f.source_ip.startswith("10.")) / 10.0),
            min(1.0, total_errors / 5.0),
            min(1.0, avg_entropy / 8.0),
            min(1.0, self.topology.get_average_cpu_load()),
            1.0 if total_pps > 600.0 else 0.0,
            is_under_attack
        ], dtype=np.float32)

        # Tier 2: Internal Network Observation (dim=8)
        # Features: [East-West PPS, SMB Flow Rate, Subnet Load Delta, Unauth Connections, Internal Entropy, Latency Delta, Compromise Ratio, Threat Flag]
        east_west_flows = [f for f in flows if f.source_ip.startswith("10.") and f.destination_ip.startswith("10.")]
        ew_pps = sum(f.packet_rate for f in east_west_flows)
        smb_count = sum(1 for f in flows if f.port in [445, 139, 3389])

        internal_obs = np.array([
            min(1.0, ew_pps / 1000.0),
            min(1.0, smb_count / 8.0),
            min(1.0, abs(total_pps - 200.0) / 1000.0),
            min(1.0, total_auth_fails / 50.0),
            min(1.0, avg_entropy / 8.0),
            min(1.0, self.topology.get_compromised_count() / len(self.topology.nodes)),
            1.0 if (current_attack and current_attack.attack_type == AttackType.APT_LATERAL_MOVEMENT) else 0.0,
            is_under_attack
        ], dtype=np.float32)

        # Tier 3: Host & Server Observation (dim=8)
        # Features: [Auth Failure Rate, CPU Spike, Process Delta, Port Sweep Count, Memory Load, Server Health Ratio, High Entropy Flag, Threat Flag]
        host_obs = np.array([
            min(1.0, total_auth_fails / 40.0),
            min(1.0, self.topology.get_average_cpu_load()),
            min(1.0, sum(1 for f in flows if f.port == 22 or f.port == 3389) / 5.0),
            1.0 if (current_attack and current_attack.attack_type == AttackType.PORT_SCAN_SYN) else 0.0,
            min(1.0, (1.0 - self.topology.get_overall_health_ratio())),
            1.0 if total_auth_fails > 10 else 0.0,
            1.0 if avg_entropy > 7.0 else 0.0,
            is_under_attack
        ], dtype=np.float32)

        return {
            "perimeter": perimeter_obs,
            "internal": internal_obs,
            "host": host_obs,
            "global_state": np.concatenate([perimeter_obs, internal_obs, host_obs])
        }
