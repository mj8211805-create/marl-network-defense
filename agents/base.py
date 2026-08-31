"""Base abstract class for all defensive agents."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional


class BaseDefenseAgent(ABC):
    """Abstract interface for cyber defense decision-makers."""

    def __init__(self, name: str, obs_dim: int, action_dim: int):
        self.name = name
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    @abstractmethod
    def select_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """Chooses a discrete defensive action given local observations."""
        pass

    def update(self, *args, **kwargs) -> Optional[float]:
        """Optional policy parameter update step."""
        return None

    def save(self, file_path: str):
        """Saves model weights."""
        pass

    def load(self, file_path: str):
        """Loads model weights."""
        pass
