"""Evaluation protocol, dataset batching, and training pipelines with validation checkpoint selection."""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple, Sequence, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ..env.simulator import SimulatedWorld
from ..models.decision_aligned_jepa import RiskSensitiveJEPAWorldModel
from ..baselines.conventional_dynamics import ConventionalDynamicsModel
from ..losses.ranking_losses import RiskSensitiveDecisionAlignmentLoss
from ..losses.risk_weights import RiskWeightingScheme
from .metrics import compute_metrics, EvaluationMetrics
from .traces import InferenceTrace, EvaluationTruth, WorldTrace


def prepare_tensors(worlds: Sequence[SimulatedWorld], device: torch.device) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List[str]]:
    """Convert worlds into PyTorch tensors for model training and evaluation.
    
    Returns:
        histories: [N, seq_len, num_features]
        actions: [N, num_candidates, action_dim]
        horizons: [N, 1]
        expected_costs: [N, num_candidates]
        tail_costs: [N, num_candidates]
        plan_names: List of plan names in order
    """
    sample = worlds[0]
    plan_names = sorted(sample.candidate_plans.keys())

    history_list = []
    action_matrix_list = []
    cost_matrix_list = []
    tail_matrix_list = []

    for w in worlds:
        history_list.append(w.public_history)
        acts = [w.candidate_plans[p].to_feature_vector() for p in plan_names]
        action_matrix_list.append(acts)
        costs = [w.expected_cost(p) for p in plan_names]
        tails = [w.cvar_cost(p, alpha=0.90) for p in plan_names]
        cost_matrix_list.append(costs)
        tail_matrix_list.append(tails)

    histories = torch.tensor(np.asarray(history_list), dtype=torch.float32, device=device)
    actions = torch.tensor(np.asarray(action_matrix_list), dtype=torch.float32, device=device)
    horizons = torch.full((len(worlds), 1), 4.0, dtype=torch.float32, device=device)
    expected_costs = torch.tensor(np.asarray(cost_matrix_list), dtype=torch.float32, device=device)
    tail_costs = torch.tensor(np.asarray(tail_matrix_list), dtype=torch.float32, device=device)

    return histories, actions, horizons, expected_costs, tail_costs, plan_names


def train_jepa_world_model(
    model: RiskSensitiveJEPAWorldModel,
    train_worlds: Sequence[SimulatedWorld],
    val_worlds: Sequence[SimulatedWorld],
    *,
    alignment_weight: float = 1.0,
    weight_mode: str = "tail_cost",
    freeze_encoder: bool = False,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> RiskSensitiveJEPAWorldModel:
    """Train JEPA World Model with validation-selected best checkpoint on decision regret."""
    model = model.to(device)
    
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = model.parameters()

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # Calibrate risk weights strictly on the training partition
    train_costs = np.array([
        [w.expected_cost(p) for p in sorted(w.candidate_plans.keys())]
        for w in train_worlds
    ]).reshape(-1)
    risk_scheme = RiskWeightingScheme.fit_from_training_costs(train_costs)

    criterion = RiskSensitiveDecisionAlignmentLoss(
        alignment_weight=alignment_weight,
        risk_weighting_scheme=risk_scheme,
        weight_mode=weight_mode,
    )

    histories, actions, horizons, exp_costs, tail_costs, plan_names = prepare_tensors(train_worlds, device)
    n_samples = len(train_worlds)

    has_val = len(val_worlds) > 0
    if has_val:
        v_hist, v_act, v_hor, v_exp, _, _ = prepare_tensors(val_worlds, device)

    best_model_state = copy.deepcopy(model.state_dict())
    best_val_regret = float("inf")

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(n_samples, device=device)
        for i in range(0, n_samples, batch_size):
            indices = permutation[i : i + batch_size]
            b_hist = histories[indices]
            b_act = actions[indices]
            b_hor = horizons[indices]
            b_costs = exp_costs[indices]
            b_tail = tail_costs[indices]
            b_costs_centered = b_costs - b_costs.mean(dim=-1, keepdim=True)
            b_tail_centered = b_tail - b_tail.mean(dim=-1, keepdim=True)

            optimizer.zero_grad(set_to_none=True)
            _, pred_costs, pred_tails = model(b_hist, b_act, b_hor)
            loss, _ = criterion(
                pred_costs,
                b_costs_centered,
                pred_tails,
                b_tail_centered,
                raw_costs=b_costs,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        # Validation checkpoint evaluation: select on decision regret
        if has_val and (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                _, v_pred_c, v_pred_t = model(v_hist, v_act, v_hor)
                v_scores = (v_pred_c + model.risk_beta * v_pred_t).cpu().numpy()
                v_real = v_exp.cpu().numpy()
                v_selected = np.argmin(v_scores, axis=1)
                v_oracle = np.argmin(v_real, axis=1)
                v_regret = float(np.mean(v_real[np.arange(len(v_real)), v_selected] - v_real[np.arange(len(v_real)), v_oracle]))
                if v_regret < best_val_regret:
                    best_val_regret = v_regret
                    best_model_state = copy.deepcopy(model.state_dict())
            model.train()

    if has_val:
        model.load_state_dict(best_model_state)

    model.eval()
    return model


def train_conventional_dynamics(
    train_worlds: Sequence[SimulatedWorld],
    val_worlds: Sequence[SimulatedWorld] | None = None,
    *,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 5e-4,
    device: torch.device = torch.device("cpu"),
) -> ConventionalDynamicsModel:
    """Train conventional MLP dynamics baseline with validation selection."""
    model = ConventionalDynamicsModel(raw_feature_dim=16 * 16, action_dim=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    plan_names = sorted(train_worlds[0].candidate_plans.keys())
    raw_hist_list = []
    action_list = []
    target_cost_list = []

    for w in train_worlds:
        flat_hist = w.public_history.reshape(-1)
        w_costs = [w.expected_cost(p) for p in plan_names]
        w_mean = np.mean(w_costs)
        for p, c in zip(plan_names, w_costs):
            raw_hist_list.append(flat_hist)
            action_list.append(w.candidate_plans[p].to_feature_vector())
            target_cost_list.append(c - w_mean)

    raw_hist = torch.tensor(np.asarray(raw_hist_list), dtype=torch.float32, device=device)
    actions = torch.tensor(np.asarray(action_list), dtype=torch.float32, device=device)
    targets = torch.tensor(np.asarray(target_cost_list), dtype=torch.float32, device=device)

    n_samples = len(raw_hist)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(n_samples, device=device)
        for i in range(0, n_samples, batch_size):
            idx = permutation[i : i + batch_size]
            optimizer.zero_grad()
            preds = model(raw_hist[idx], actions[idx])
            loss = F.smooth_l1_loss(preds, targets[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

    model.eval()
    return model


@torch.no_grad()
def predict_jepa_candidate_scores(
    model: RiskSensitiveJEPAWorldModel,
    public_history: np.ndarray,
    candidate_plans: Dict[str, Any],
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Inference-only prediction for candidate plans using ONLY observable inputs.
    
    Returns:
        (predicted_mean_costs, predicted_tail_costs, final_decision_scores)
    """
    model.eval()
    plan_names = sorted(candidate_plans.keys())
    
    hist_t = torch.tensor(public_history[np.newaxis, ...], dtype=torch.float32, device=device)
    act_vectors = [candidate_plans[p].to_feature_vector() for p in plan_names]
    act_t = torch.tensor(np.asarray(act_vectors)[np.newaxis, ...], dtype=torch.float32, device=device)
    hor_t = torch.full((1, 1), 4.0, dtype=torch.float32, device=device)

    _, pred_costs, pred_tails = model(hist_t, act_t, hor_t)
    final_scores = pred_costs + model.risk_beta * pred_tails

    c_vals = pred_costs.squeeze(0).cpu().numpy()
    t_vals = pred_tails.squeeze(0).cpu().numpy()
    f_vals = final_scores.squeeze(0).cpu().numpy()

    mean_dict = {p: float(c_vals[i]) for i, p in enumerate(plan_names)}
    tail_dict = {p: float(t_vals[i]) for i, p in enumerate(plan_names)}
    score_dict = {p: float(f_vals[i]) for i, p in enumerate(plan_names)}

    return mean_dict, tail_dict, score_dict


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_worlds: Sequence[SimulatedWorld],
    model_type: str,
    device: torch.device = torch.device("cpu"),
) -> EvaluationMetrics:
    """Evaluate a trained model or baseline planner on test worlds."""
    model.eval()
    histories, actions, horizons, exp_costs, _, plan_names = prepare_tensors(test_worlds, device)
    n_worlds = len(test_worlds)
    n_candidates = len(plan_names)

    if model_type == "conventional_dynamics":
        predicted_scores = np.zeros((n_worlds, n_candidates), dtype=np.float32)
        for i, w in enumerate(test_worlds):
            flat_hist = torch.tensor(w.public_history.reshape(1, -1), dtype=torch.float32, device=device)
            for j, p in enumerate(plan_names):
                act_vec = torch.tensor(w.candidate_plans[p].to_feature_vector().reshape(1, -1), dtype=torch.float32, device=device)
                pred_c = model(flat_hist, act_vec)
                predicted_scores[i, j] = float(pred_c.item())
    elif isinstance(model, RiskSensitiveJEPAWorldModel):
        _, pred_costs, pred_tails = model(histories, actions, horizons)
        combined_scores = pred_costs + model.risk_beta * pred_tails
        predicted_scores = combined_scores.cpu().numpy()
    else:
        raise ValueError(f"Unknown model type {model_type}")

    realized_costs = exp_costs.cpu().numpy()
    selected_indices = np.argmin(predicted_scores, axis=1)
    fill_rates = []
    stockout_periods = []
    for i, w in enumerate(test_worlds):
        chosen_plan = plan_names[selected_indices[i]]
        outcomes = w.outcomes_by_plan[chosen_plan]
        fill_rates.append(float(np.mean([o.fill_rate for o in outcomes])))
        stockout_periods.append(int(np.sum([o.stockout_periods for o in outcomes])))

    return compute_metrics(
        realized_costs=realized_costs,
        predicted_scores=predicted_scores,
        fill_rates=fill_rates,
        stockout_periods=stockout_periods,
    )


def evaluate_heuristic_planner(
    planner: Any,
    test_worlds: Sequence[SimulatedWorld],
) -> EvaluationMetrics:
    """Evaluate a rule-based or heuristic planner on test worlds."""
    plan_names = sorted(test_worlds[0].candidate_plans.keys())
    plan_to_idx = {name: i for i, name in enumerate(plan_names)}
    n_worlds = len(test_worlds)
    n_candidates = len(plan_names)

    realized_costs = np.zeros((n_worlds, n_candidates), dtype=np.float32)
    predicted_scores = np.zeros((n_worlds, n_candidates), dtype=np.float32)
    fill_rates = []
    stockout_periods = []

    for i, w in enumerate(test_worlds):
        for j, p in enumerate(plan_names):
            realized_costs[i, j] = w.expected_cost(p)

        chosen_plan = planner.select_plan(w)
        chosen_idx = plan_to_idx.get(chosen_plan, 0)
        predicted_scores[i, :] = 100.0
        predicted_scores[i, chosen_idx] = 0.0

        outcomes = w.outcomes_by_plan[chosen_plan]
        fill_rates.append(float(np.mean([o.fill_rate for o in outcomes])))
        stockout_periods.append(int(np.sum([o.stockout_periods for o in outcomes])))

    return compute_metrics(
        realized_costs=realized_costs,
        predicted_scores=predicted_scores,
        fill_rates=fill_rates,
        stockout_periods=stockout_periods,
    )
