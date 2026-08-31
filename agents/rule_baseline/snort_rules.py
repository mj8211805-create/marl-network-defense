"""Traditional static threshold and signature rule-based IDS baseline."""

import numpy as np
from typing import Dict, Any


class RuleBasedIDSAgent:
    """Signature and threshold-based rule engine (Snort / Suricata heuristic style)."""

    def select_actions(self, observations: Dict[str, np.ndarray], **kwargs) -> Dict[str, int]:
        p_obs = observations["perimeter"]
        i_obs = observations["internal"]
        h_obs = observations["host"]

        # 1. Perimeter Rules
        # Rule 1: High PPS & High SYN ratio -> SYN Flood Detection -> SYN_COOKIES (3)
        if p_obs[0] > 0.35 and p_obs[1] > 0.80:
            p_act = 3
        # Rule 2: High Volumetric Rate -> BLOCK_IP (2)
        elif p_obs[0] > 0.50:
            p_act = 2
        # Rule 3: High SYN Ratio with Moderate PPS -> Port Scan -> RATE_LIMIT (1)
        elif p_obs[1] > 0.85:
            p_act = 1
        else:
            p_act = 0

        # 2. Internal Network Rules
        # Rule 4: High East-West Traffic & High Entropy -> Lateral APT -> ISOLATE_SUBNET (1)
        if i_obs[0] > 0.30 and i_obs[4] > 0.70:
            i_act = 1
        # Rule 5: High SMB connection anomaly -> TERMINATE_LATERAL (2)
        elif i_obs[1] > 0.40:
            i_act = 2
        else:
            i_act = 0

        # 3. Host Tier Rules
        # Rule 6: Auth failure rate above threshold -> Brute Force -> LOCK_CREDENTIALS (3)
        if h_obs[0] > 0.25:
            h_act = 3
        # Rule 7: Severe CPU Spike & Health degradation -> QUARANTINE_HOST (1)
        elif h_obs[1] > 0.75:
            h_act = 1
        else:
            h_act = 0

        return {
            "perimeter": p_act,
            "internal": i_act,
            "host": h_act
        }
