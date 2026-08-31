"""Evaluation metrics formulas and Pydantic schemas for CyberMARL."""

from pydantic import BaseModel, Field
from typing import Dict, List, Any
from datetime import datetime


class DefenseEvaluationMetrics(BaseModel):
    """Performance metrics for an individual defense method."""
    method_name: str
    total_eval_steps: int = 0
    total_attacks_generated: int = 0
    total_attacks_mitigated: int = 0
    
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    
    network_availability_qos: float = 0.0  # Percentage (0 to 100%)
    mean_time_to_mitigate_steps: float = 0.0
    average_episode_reward: float = 0.0
    avg_inference_latency_ms: float = 0.0


class ComparativeBenchmarkReport(BaseModel):
    """Comparative report across all 5 evaluated defense methods."""
    benchmark_id: str
    scenario_name: str
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    total_episodes: int = 10
    
    marl_system: DefenseEvaluationMetrics
    single_rl_baseline: DefenseEvaluationMetrics
    supervised_ml_baseline: DefenseEvaluationMetrics
    anomaly_detector_baseline: DefenseEvaluationMetrics
    rule_based_ids_baseline: DefenseEvaluationMetrics
    
    summary_analysis: str = ""


def calculate_defense_metrics(
    method_name: str,
    total_steps: int,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    attacks_gen: int,
    attacks_mit: int,
    avail_history: List[float],
    reward_history: List[float],
    latency_ms: float
) -> DefenseEvaluationMetrics:
    """Computes comprehensive classification, availability, and response metrics."""
    total_samples = tp + fp + fn + tn
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    avg_qos = (sum(avail_history) / len(avail_history) * 100.0) if avail_history else 100.0
    avg_reward = (sum(reward_history) / len(reward_history)) if reward_history else 0.0
    mttr = 1.0 + (fn * 0.8) / max(1, attacks_gen)

    return DefenseEvaluationMetrics(
        method_name=method_name,
        total_eval_steps=total_steps,
        total_attacks_generated=attacks_gen,
        total_attacks_mitigated=attacks_mit,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        false_positive_rate=round(fpr * 100.0, 2),
        false_negative_rate=round(fnr * 100.0, 2),
        network_availability_qos=round(avg_qos, 2),
        mean_time_to_mitigate_steps=round(mttr, 2),
        average_episode_reward=round(avg_reward, 2),
        avg_inference_latency_ms=round(latency_ms, 3)
    )
