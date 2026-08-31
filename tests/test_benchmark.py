"""Unit tests for Benchmark Evaluator and metrics."""

from benchmark.evaluator import CyberMARLEvaluator
from benchmark.metrics import calculate_defense_metrics


def test_calculate_defense_metrics_formula():
    metrics = calculate_defense_metrics(
        method_name="TestAgent",
        total_steps=100,
        tp=40,
        fp=5,
        fn=5,
        tn=50,
        attacks_gen=45,
        attacks_mit=40,
        avail_history=[1.0, 0.9, 0.95],
        reward_history=[5.0, 10.0],
        latency_ms=0.1
    )

    assert metrics.precision > 0.8
    assert metrics.recall > 0.8
    assert metrics.f1_score > 0.8
    assert metrics.accuracy == 0.9
    assert metrics.network_availability_qos >= 90.0


def test_evaluator_run():
    evaluator = CyberMARLEvaluator()
    report = evaluator.run_benchmark(episodes=2, steps_per_episode=15)

    assert report.marl_system is not None
    assert report.single_rl_baseline is not None
    assert report.supervised_ml_baseline is not None
    assert report.rule_based_ids_baseline is not None

    md = evaluator.generate_markdown_table(report)
    assert "# CyberMARL Comparative Evaluation" in md
