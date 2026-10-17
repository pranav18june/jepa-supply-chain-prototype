"""Inference-time uncertainty-gated policy and threshold calibration."""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple, Sequence, Any, Optional
import numpy as np
import torch
import torch.nn as nn

from ..env.simulator import SimulatedWorld
from ..models.decision_aligned_jepa import RiskSensitiveJEPAWorldModel
from ..models.uncertainty import EnsembleUncertaintyGate, compute_ensemble_uncertainties
from .traces import InferenceTrace, EvaluationTruth, WorldTrace
from .metrics import compute_metrics, EvaluationMetrics
from .protocol import predict_jepa_candidate_scores


class GatedEnsemblePolicy:
    """Combines an ensemble of learned models with a fallback policy routed by an uncertainty gate.
    
    Guarantees:
    - Never inspects oracle values, future rollouts, or regime labels at decision time.
    - Records complete audit traces for post-hoc regret and reliability analysis.
    """

    def __init__(
        self,
        ensemble_models: List[RiskSensitiveJEPAWorldModel],
        fallback_planner: Any,
        gate: EnsembleUncertaintyGate,
        method_name: str = "gated_ensemble",
        run_id: str = "run_default",
        device: torch.device = torch.device("cpu"),
    ):
        self.ensemble_models = ensemble_models
        self.fallback_planner = fallback_planner
        self.gate = gate
        self.method_name = method_name
        self.run_id = run_id
        self.device = device

    def predict_and_select(self, world: SimulatedWorld) -> Tuple[str, InferenceTrace]:
        """Inference-only decision selection and trace generation."""
        candidate_names = sorted(world.candidate_plans.keys())
        
        # 1. Collect predictions from all ensemble members
        member_mean_costs = []
        member_tail_costs = []
        member_decision_scores = []

        for model in self.ensemble_models:
            means, tails, scores = predict_jepa_candidate_scores(
                model,
                world.public_history,
                world.candidate_plans,
                device=self.device,
            )
            member_mean_costs.append(means)
            member_tail_costs.append(tails)
            member_decision_scores.append(scores)

        # 2. Compute uncertainty metrics across members
        uncertainties = compute_ensemble_uncertainties(member_decision_scores, candidate_names)

        # 3. Aggregate ensemble scores
        agg_mean_costs = {
            p: float(np.mean([m[p] for m in member_mean_costs]))
            for p in candidate_names
        }
        agg_tail_costs = {
            p: float(np.mean([t[p] for t in member_tail_costs]))
            for p in candidate_names
        }
        agg_decision_scores = {
            p: float(np.mean([s[p] for s in member_decision_scores]))
            for p in candidate_names
        }

        sorted_candidates = sorted(candidate_names, key=lambda p: agg_decision_scores[p])
        best_candidate = sorted_candidates[0]
        second_best_candidate = sorted_candidates[1] if len(sorted_candidates) > 1 else best_candidate

        chosen_score = agg_decision_scores[best_candidate]
        second_score = agg_decision_scores[second_best_candidate]
        score_margin = second_score - chosen_score

        # 4. Uncertainty gate decision
        fallback_triggered = self.gate.should_fallback(uncertainties)

        if fallback_triggered:
            chosen_plan = self.fallback_planner.select_plan(world)
            policy_type = "fallback"
        else:
            chosen_plan = best_candidate
            policy_type = "ensemble"

        trace = InferenceTrace(
            run_id=self.run_id,
            method=self.method_name,
            world_id=world.world_id,
            candidate_names=candidate_names,
            predicted_mean_costs=agg_mean_costs,
            predicted_tail_costs=agg_tail_costs,
            final_decision_scores=agg_decision_scores,
            chosen_plan=chosen_plan,
            chosen_score=chosen_score,
            second_best_score=second_score,
            score_margin=score_margin,
            uncertainty_scores=uncertainties,
            fallback_triggered=fallback_triggered,
            policy_type=policy_type,
        )

        return chosen_plan, trace


def calibrate_gate_threshold(
    calibration_traces: List[WorldTrace],
    fallback_planner: Any,
    calibration_worlds: Sequence[SimulatedWorld],
    metric_key: str = "top_action_disagreement",
    reverse_direction: bool = False,
    guardrail_catastrophic_rate: float = 0.05,
) -> float:
    """Calibrate gate threshold strictly on calibration worlds.
    
    Selects the threshold minimizing mean decision regret subject to the catastrophic guardrail.
    """
    assert len(calibration_traces) == len(calibration_worlds)
    world_map = {w.world_id: w for w in calibration_worlds}

    # Extract calibration candidate values
    uncertainty_values = [
        t.inference.uncertainty_scores.get(metric_key, 0.0)
        for t in calibration_traces
    ]
    unique_thresholds = sorted(list(set(uncertainty_values)))
    if not unique_thresholds:
        return 0.5

    best_threshold = unique_thresholds[0]
    best_mean_regret = float("inf")

    # Evaluate each candidate threshold
    for thresh in unique_thresholds:
        test_gate = EnsembleUncertaintyGate(thresh, metric_key=metric_key, reverse_direction=reverse_direction)
        simulated_regrets = []
        catastrophic_flags = []

        for trace in calibration_traces:
            w = world_map[trace.inference.world_id]
            u_score = trace.inference.uncertainty_scores
            should_fb = test_gate.should_fallback(u_score)
            
            if should_fb:
                fb_plan = fallback_planner.select_plan(w)
                chosen_cost = w.expected_cost(fb_plan)
            else:
                ens_plan = min(trace.inference.final_decision_scores.keys(), key=lambda p: trace.inference.final_decision_scores[p])
                chosen_cost = w.expected_cost(ens_plan)

            oracle_cost = w.oracle_cost()
            regret = chosen_cost - oracle_cost
            simulated_regrets.append(regret)
            catastrophic_flags.append(chosen_cost >= (oracle_cost * 1.75 + 5.0))

        cat_rate = float(np.mean(catastrophic_flags))
        mean_reg = float(np.mean(simulated_regrets))

        if cat_rate <= guardrail_catastrophic_rate and mean_reg < best_mean_regret:
            best_mean_regret = mean_reg
            best_threshold = thresh

    return float(best_threshold)


def evaluate_policy_with_traces(
    policy: Any,
    test_worlds: Sequence[SimulatedWorld],
    method_name: str,
    run_id: str = "run_default",
) -> Tuple[List[WorldTrace], EvaluationMetrics]:
    """Execute a policy on test worlds and generate complete WorldTrace objects and metrics."""
    plan_names = sorted(test_worlds[0].candidate_plans.keys())
    n_worlds = len(test_worlds)
    n_candidates = len(plan_names)

    traces: List[WorldTrace] = []
    realized_costs = np.zeros((n_worlds, n_candidates), dtype=np.float32)
    predicted_scores = np.zeros((n_worlds, n_candidates), dtype=np.float32)
    fill_rates: List[float] = []
    stockout_periods: List[int] = []
    fallback_flags: List[bool] = []

    for i, w in enumerate(test_worlds):
        for j, p in enumerate(plan_names):
            realized_costs[i, j] = w.expected_cost(p)

        if hasattr(policy, "predict_and_select"):
            chosen_plan, inf_trace = policy.predict_and_select(w)
            for j, p in enumerate(plan_names):
                predicted_scores[i, j] = inf_trace.final_decision_scores.get(p, 0.0)
            fallback_flags.append(inf_trace.fallback_triggered)
        else:
            # Standalone baseline planner
            chosen_plan = policy.select_plan(w)
            fallback_flags.append(False)
            chosen_idx = plan_names.index(chosen_plan)
            predicted_scores[i, :] = 100.0
            predicted_scores[i, chosen_idx] = 0.0

            inf_trace = InferenceTrace(
                run_id=run_id,
                method=method_name,
                world_id=w.world_id,
                candidate_names=plan_names,
                predicted_mean_costs={p: 0.0 for p in plan_names},
                predicted_tail_costs={p: 0.0 for p in plan_names},
                final_decision_scores={p: float(predicted_scores[i, j]) for j, p in enumerate(plan_names)},
                chosen_plan=chosen_plan,
                chosen_score=0.0,
                second_best_score=100.0,
                score_margin=100.0,
                uncertainty_scores={},
                fallback_triggered=False,
                policy_type="heuristic",
            )

        # Evaluation Truth (constructed strictly post-decision)
        outcomes = w.outcomes_by_plan[chosen_plan]
        chosen_cost = float(w.expected_cost(chosen_plan))
        oracle_plan = w.oracle_plan()
        oracle_cost = float(w.oracle_cost())
        regret = chosen_cost - oracle_cost
        rel_regret = regret / max(oracle_cost, 1e-4)

        w_fill = float(np.mean([o.fill_rate for o in outcomes]))
        w_stockout = int(np.sum([o.stockout_periods for o in outcomes]))
        w_backorder = float(np.mean([o.terminal_backorders for o in outcomes]))
        is_catastrophic = chosen_cost >= (oracle_cost * 1.75 + 5.0)

        fill_rates.append(w_fill)
        stockout_periods.append(w_stockout)

        eval_truth = EvaluationTruth(
            world_id=w.world_id,
            regime=w.regime.value,
            realized_costs_by_plan={p: float(realized_costs[i, j]) for j, p in enumerate(plan_names)},
            oracle_plan=oracle_plan,
            oracle_cost=oracle_cost,
            chosen_cost=chosen_cost,
            regret=regret,
            relative_regret=rel_regret,
            fill_rate=w_fill,
            stockout_periods=w_stockout,
            terminal_backorders=w_backorder,
            is_catastrophic_inversion=is_catastrophic,
        )

        traces.append(WorldTrace(inference=inf_trace, evaluation=eval_truth))

    metrics = compute_metrics(
        realized_costs=realized_costs,
        predicted_scores=predicted_scores,
        fill_rates=fill_rates,
        stockout_periods=stockout_periods,
        fallback_flags=fallback_flags,
    )

    return traces, metrics
