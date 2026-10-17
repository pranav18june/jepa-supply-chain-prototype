"""Evaluation package for metrics, protocol, and benchmarking."""

from .metrics import EvaluationMetrics, compute_metrics
from .protocol import (
    prepare_tensors,
    train_jepa_world_model,
    train_conventional_dynamics,
    evaluate_model,
    evaluate_heuristic_planner,
)
from .benchmark import BenchmarkConfig, BenchmarkRunner

__all__ = [
    "EvaluationMetrics",
    "compute_metrics",
    "prepare_tensors",
    "train_jepa_world_model",
    "train_conventional_dynamics",
    "evaluate_model",
    "evaluate_heuristic_planner",
    "BenchmarkConfig",
    "BenchmarkRunner",
]
