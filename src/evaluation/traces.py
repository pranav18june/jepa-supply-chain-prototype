"""Trace-level decision and evaluation records ensuring strict separation of inference from evaluation truth."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class InferenceTrace:
    """Contains exclusively decision-time observable inputs and model predictions.
    
    CRITICAL: This object MUST NEVER contain oracle costs, realized future outcomes,
    or ground-truth disruption regime labels.
    """
    run_id: str
    method: str
    world_id: int
    candidate_names: List[str]
    predicted_mean_costs: Dict[str, float]
    predicted_tail_costs: Dict[str, float]
    final_decision_scores: Dict[str, float]
    chosen_plan: str
    chosen_score: float
    second_best_score: float
    score_margin: float
    uncertainty_scores: Dict[str, float] = field(default_factory=dict)
    fallback_triggered: bool = False
    policy_type: str = "learned"  # e.g., 'learned', 'fallback', 'heuristic'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationTruth:
    """Contains evaluation-only counterfactual truth and realized performance.
    
    This object is constructed strictly AFTER action selection is sealed.
    """
    world_id: int
    regime: str
    realized_costs_by_plan: Dict[str, float]
    oracle_plan: str
    oracle_cost: float
    chosen_cost: float
    regret: float
    relative_regret: float
    fill_rate: float
    stockout_periods: int
    terminal_backorders: float
    is_catastrophic_inversion: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorldTrace:
    """Combines inference and evaluation records into a unified audit trail."""
    inference: InferenceTrace
    evaluation: EvaluationTruth

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inference": self.inference.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorldTrace:
        return cls(
            inference=InferenceTrace(**data["inference"]),
            evaluation=EvaluationTruth(**data["evaluation"]),
        )


def save_traces_jsonl(traces: List[WorldTrace], path: Path) -> None:
    """Appends or writes a list of WorldTrace objects to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict()) + "\n")


def load_traces_jsonl(path: Path) -> List[WorldTrace]:
    """Loads a list of WorldTrace objects from a JSONL file."""
    path = Path(path)
    traces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(WorldTrace.from_dict(json.loads(line)))
    return traces
