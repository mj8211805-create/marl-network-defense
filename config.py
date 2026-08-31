"""Global configuration and hyperparameters for CyberMARL."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

class CyberConfig(BaseSettings):
    # App Settings
    APP_NAME: str = "CyberMARL Autonomous Network Defense Platform"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Environment Settings
    MAX_STEPS_PER_EPISODE: int = 100
    NUM_WORKSTATIONS: int = 8
    NUM_SERVERS: int = 4
    DEFAULT_TRAFFIC_RATE: float = 100.0  # packets/step
    ATTACK_OCCURRENCE_PROBABILITY: float = 0.45
    
    # Feature Dimensions
    PERIMETER_OBS_DIM: int = 8
    INTERNAL_OBS_DIM: int = 8
    HOST_OBS_DIM: int = 8
    GLOBAL_STATE_DIM: int = 24
    
    # Action Space Dimensions
    PERIMETER_ACTION_DIM: int = 5   # [PASS, RATE_LIMIT, BLOCK_IP, SYN_COOKIES, DROP_ALL]
    INTERNAL_ACTION_DIM: int = 4    # [PASS, ISOLATE_SUBNET, TERMINATE_LATERAL, HONEYPOT_REDIRECT]
    HOST_ACTION_DIM: int = 4        # [PASS, QUARANTINE_HOST, RESTART_SERVICE, LOCK_CREDENTIALS]
    
    # MARL Training Hyperparameters
    LEARNING_RATE: float = 0.001
    GAMMA: float = 0.95
    EPSILON_START: float = 1.0
    EPSILON_MIN: float = 0.05
    EPSILON_DECAY: float = 0.992
    MEMORY_SIZE: int = 20000
    BATCH_SIZE: int = 64
    TARGET_UPDATE_STEPS: int = 20
    
    # Reward Function Weights
    REWARD_MITIGATE_ATTACK: float = 10.0
    PENALTY_FALSE_POSITIVE: float = -6.0
    PENALTY_FALSE_NEGATIVE: float = -15.0
    REWARD_QOS_COEFFICIENT: float = 2.5
    PENALTY_ACTION_COST: float = -0.5

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

config = CyberConfig()
