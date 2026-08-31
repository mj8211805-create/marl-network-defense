"""Supervised Machine Learning Intrusion Detection Baseline (Random Forest / GBDT)."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional
from config import config


class SupervisedMLDefenseAgent:
    """Supervised ML Classifier (Random Forest) trained to map flow features to defense actions."""

    def __init__(self):
        self.p_model = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42)
        self.i_model = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42)
        self.h_model = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42)
        self.is_trained = False
        self._pretrain_synthetic()

    def _pretrain_synthetic(self):
        """Pre-trains on standard labeled feature patterns."""
        # 1. Perimeter Training Data: [PPS, SYN_Ratio, Ext_Rate, Err_Rate, Entropy, CPU, Vol_Flag, Atk_Flag]
        X_p = np.array([
            [0.1, 0.2, 0.2, 0.0, 0.6, 0.1, 0.0, 0.0],  # Benign -> PASS (0)
            [0.2, 0.3, 0.1, 0.0, 0.5, 0.1, 0.0, 0.0],  # Benign -> PASS (0)
            [0.9, 0.95, 0.8, 0.6, 0.4, 0.8, 1.0, 1.0], # SYN Flood -> SYN_COOKIES (3)
            [0.8, 0.9, 0.9, 0.5, 0.5, 0.7, 1.0, 1.0],  # DoS Flood -> BLOCK_IP (2)
            [0.3, 0.9, 0.4, 0.2, 0.3, 0.2, 0.0, 1.0],  # Port Scan -> RATE_LIMIT (1)
            [0.05, 0.1, 0.0, 0.0, 0.5, 0.05, 0.0, 0.0] # Idle -> PASS (0)
        ], dtype=np.float32)
        y_p = np.array([0, 0, 3, 2, 1, 0], dtype=np.int64)
        self.p_model.fit(X_p, y_p)

        # 2. Internal Training Data: [EW_PPS, SMB, Anomaly, AuthFail, Entropy, CompRatio, APT_Flag, Atk_Flag]
        X_i = np.array([
            [0.1, 0.1, 0.1, 0.0, 0.6, 0.0, 0.0, 0.0],  # Benign -> PASS (0)
            [0.8, 0.7, 0.6, 0.4, 0.9, 0.4, 1.0, 1.0],  # APT Lateral -> ISOLATE_SUBNET (1)
            [0.6, 0.5, 0.5, 0.3, 0.8, 0.3, 1.0, 1.0],  # Lateral Conn -> TERMINATE (2)
            [0.05, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]  # Normal -> PASS (0)
        ], dtype=np.float32)
        y_i = np.array([0, 1, 2, 0], dtype=np.int64)
        self.i_model.fit(X_i, y_i)

        # 3. Host Training Data: [AuthFail, CPU, Proc, Sweep, HealthLoss, FailFlag, HighEnt, Atk_Flag]
        X_h = np.array([
            [0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # Normal -> PASS (0)
            [0.8, 0.6, 0.8, 0.0, 0.3, 1.0, 0.0, 1.0],  # Brute Force -> LOCK_CREDENTIALS (3)
            [0.7, 0.8, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0],  # Compromise -> QUARANTINE (1)
            [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # Benign -> PASS (0)
        ], dtype=np.float32)
        y_h = np.array([0, 3, 1, 0], dtype=np.int64)
        self.h_model.fit(X_h, y_h)

        self.is_trained = True

    def select_actions(self, observations: Dict[str, np.ndarray], **kwargs) -> Dict[str, int]:
        """Classifies observations and returns defense action per tier."""
        p_pred = int(self.p_model.predict(observations["perimeter"].reshape(1, -1))[0])
        i_pred = int(self.i_model.predict(observations["internal"].reshape(1, -1))[0])
        h_pred = int(self.h_model.predict(observations["host"].reshape(1, -1))[0])

        return {
            "perimeter": p_pred,
            "internal": i_pred,
            "host": h_pred
        }
