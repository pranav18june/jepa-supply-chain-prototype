"""Unit tests for decision traces and schema integrity."""

import tempfile
from pathlib import Path
import pytest
from prototype_new.src.evaluation.traces import (
    InferenceTrace,
    EvaluationTruth,
    WorldTrace,
    save_traces_jsonl,
    load_traces_jsonl,
)


def test_trace_schema_and_serialization():
    inf = InferenceTrace(
        run_id="test_run",
        method="test_method",
        world_id=42,
        candidate_names=["regular", "split", "expedited"],
        predicted_mean_costs={"regular": 10.0, "split": 15.0, "expedited": 20.0},
        predicted_tail_costs={"regular": 20.0, "split": 25.0, "expedited": 30.0},
        final_decision_scores={"regular": 15.0, "split": 20.0, "expedited": 25.0},
        chosen_plan="regular",
        chosen_score=15.0,
        second_best_score=20.0,
        score_margin=5.0,
        uncertainty_scores={"top_action_disagreement": 0.0},
        fallback_triggered=False,
        policy_type="ensemble",
    )

    # Assert inference trace does NOT have oracle costs or regime
    assert not hasattr(inf, "oracle_plan")
    assert not hasattr(inf, "oracle_cost")
    assert not hasattr(inf, "regime")

    ev = EvaluationTruth(
        world_id=42,
        regime="compound_shift",
        realized_costs_by_plan={"regular": 12.0, "split": 11.0, "expedited": 18.0},
        oracle_plan="split",
        oracle_cost=11.0,
        chosen_cost=12.0,
        regret=1.0,
        relative_regret=1.0 / 11.0,
        fill_rate=0.95,
        stockout_periods=1,
        terminal_backorders=2.0,
        is_catastrophic_inversion=False,
    )

    wt = WorldTrace(inference=inf, evaluation=ev)
    d = wt.to_dict()
    assert d["inference"]["chosen_plan"] == "regular"
    assert d["evaluation"]["regret"] == 1.0

    # Test JSONL round-trip
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "traces.jsonl"
        save_traces_jsonl([wt, wt], tmp_file)
        loaded = load_traces_jsonl(tmp_file)
        assert len(loaded) == 2
        assert loaded[0].inference.world_id == 42
        assert loaded[0].evaluation.oracle_plan == "split"
