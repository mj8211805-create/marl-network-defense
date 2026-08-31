"""Cyber threat vectors and attack models."""

import random
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class AttackType(str, Enum):
    BENIGN = "Benign"
    DOS_SYN_FLOOD = "DoS_SYN_Flood"
    DDOS_HTTP_FLOOD = "DDoS_HTTP_Flood"
    PORT_SCAN_SYN = "Port_Scan_SYN"
    BRUTE_FORCE_SSH = "Brute_Force_SSH"
    APT_LATERAL_MOVEMENT = "APT_Lateral_Movement"


@dataclass
class ThreatVector:
    attack_type: AttackType
    intensity: float = 1.0  # 0.1 to 1.0
    source_ip: str = "185.220.101.5"
    target_node_id: str = "dmz-web-01"
    target_port: int = 80
    packet_rate: float = 500.0  # packets/step
    syn_ack_ratio: float = 0.95
    failed_auth_count: int = 0
    payload_entropy: float = 6.5
    stealth_level: float = 0.0  # 0.0 (noisy) to 1.0 (ultra-stealth)

    @classmethod
    def generate_random_attack(cls) -> "ThreatVector":
        atk = random.choice([
            AttackType.DOS_SYN_FLOOD,
            AttackType.DDOS_HTTP_FLOOD,
            AttackType.PORT_SCAN_SYN,
            AttackType.BRUTE_FORCE_SSH,
            AttackType.APT_LATERAL_MOVEMENT
        ])
        return cls.create_specific_attack(atk)

    @classmethod
    def create_specific_attack(cls, attack_type: AttackType) -> "ThreatVector":
        if attack_type == AttackType.DOS_SYN_FLOOD:
            return cls(
                attack_type=attack_type,
                intensity=random.uniform(0.7, 1.0),
                source_ip=f"198.51.100.{random.randint(1, 254)}",
                target_node_id="dmz-web-01",
                target_port=80,
                packet_rate=random.uniform(800.0, 1500.0),
                syn_ack_ratio=random.uniform(0.92, 0.99),
                failed_auth_count=0,
                payload_entropy=random.uniform(3.0, 4.5),
                stealth_level=0.1
            )
        elif attack_type == AttackType.DDOS_HTTP_FLOOD:
            return cls(
                attack_type=attack_type,
                intensity=random.uniform(0.8, 1.0),
                source_ip=f"203.0.113.{random.randint(1, 254)}",
                target_node_id="dmz-api-01",
                target_port=443,
                packet_rate=random.uniform(1000.0, 2000.0),
                syn_ack_ratio=random.uniform(0.5, 0.7),
                failed_auth_count=0,
                payload_entropy=random.uniform(5.5, 7.2),
                stealth_level=0.2
            )
        elif attack_type == AttackType.PORT_SCAN_SYN:
            return cls(
                attack_type=attack_type,
                intensity=random.uniform(0.4, 0.7),
                source_ip=f"185.100.50.{random.randint(1, 254)}",
                target_node_id="gw-01",
                target_port=random.randint(20, 8080),
                packet_rate=random.uniform(200.0, 450.0),
                syn_ack_ratio=random.uniform(0.85, 0.98),
                failed_auth_count=0,
                payload_entropy=random.uniform(2.0, 3.5),
                stealth_level=0.6
            )
        elif attack_type == AttackType.BRUTE_FORCE_SSH:
            return cls(
                attack_type=attack_type,
                intensity=random.uniform(0.5, 0.9),
                source_ip=f"45.33.32.{random.randint(1, 254)}",
                target_node_id="srv-app-01",
                target_port=22,
                packet_rate=random.uniform(80.0, 180.0),
                syn_ack_ratio=random.uniform(0.4, 0.6),
                failed_auth_count=random.randint(15, 60),
                payload_entropy=random.uniform(4.0, 5.5),
                stealth_level=0.4
            )
        elif attack_type == AttackType.APT_LATERAL_MOVEMENT:
            return cls(
                attack_type=attack_type,
                intensity=random.uniform(0.6, 0.95),
                source_ip="10.0.4.11",  # Originates from compromised workstation
                target_node_id="dc-primary-01",
                target_port=445,
                packet_rate=random.uniform(60.0, 150.0),
                syn_ack_ratio=random.uniform(0.3, 0.5),
                failed_auth_count=random.randint(5, 25),
                payload_entropy=random.uniform(7.0, 7.9),  # Encrypted C2 payload
                stealth_level=0.85
            )
        else:
            # Benign fallback
            return cls(
                attack_type=AttackType.BENIGN,
                intensity=0.0,
                source_ip="10.0.4.12",
                target_node_id="srv-file-01",
                target_port=445,
                packet_rate=random.uniform(20.0, 60.0),
                syn_ack_ratio=random.uniform(0.1, 0.3),
                failed_auth_count=0,
                payload_entropy=random.uniform(4.0, 6.0),
                stealth_level=1.0
            )
