"""Unsupervised Anomaly Detection Baseline (Isolation Forest)."""

import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Any


class AnomalyDetectorDefenseAgent:
    """Unsupervised Anomaly Detection Agent using Isolation Forest."""

    def __init__(self):
        self.iso_forest = IsolationForest(n_estimators=50, contamination=0.15, random_state=42)
        self._fit_baseline_traffic()

    def _fit_baseline_traffic(self):
        """Fits on synthetic normal traffic distributions."""
        # Generate 200 normal benign feature samples
        normal_samples = np.random.normal(loc=0.15, scale=0.08, size=(200, 8))
        normal_samples = np.clip(normal_samples, 0.0, 0.5)
        self.iso_forest.fit(normal_samples)

    def select_actions(self, observations: Dict[str, np.ndarray], **kwargs) -> Dict[str, int]:
        """Calculates outlier score: if anomalous (-1), triggers defensive mitigation."""
        p_score = self.iso_forest.predict(observations["perimeter"].reshape(1, -1))[0]
        i_score = self.iso_forest.predict(observations["internal"].reshape(1, -1))[0]
        h_score = self.iso_forest.predict(observations["host"].reshape(1, -1))[0]

        # Perimeter action: Anomaly -> Rate Limit (1) or Block (2)
        p_act = 2 if p_score == -1 and observations["perimeter"][0] > 0.4 else (1 if p_score == -1 else 0)
        # Internal action: Anomaly -> Isolate (1) or Terminate (2)
        i_act = 1 if i_score == -1 and observations["internal"][0] > 0.3 else 0
        # Host action: Anomaly -> Lock Credentials (3) or Quarantine (1)
        h_act = 3 if h_score == -1 and observations["host"][0] > 0.2 else 0

        return {
            "perimeter": p_act,
            "internal": i_act,
            "host": h_act
        }
