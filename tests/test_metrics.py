# tests/test_metrics.py
import pytest
from longhorizon_bench.evaluation.scorer import ActionScore, CheckpointScore
from longhorizon_bench.evaluation.consistency import Violation


def test_event_accuracy():
    from longhorizon_bench.evaluation.metrics import compute_metrics

    event_scores = [
        {"event_id": "E01", "is_critical": True, "score": ActionScore(tool_score=1.0, param_scores={"a": 1.0}, total=1.0)},
        {"event_id": "E02", "is_critical": False, "score": ActionScore(tool_score=1.0, param_scores={"a": 0.5}, total=0.7)},
        {"event_id": "E03", "is_critical": True, "score": ActionScore(tool_score=0.0, param_scores={"a": 0.0}, total=0.0)},
    ]
    result = compute_metrics(event_scores=event_scores, checkpoint_scores=[], violations=[])
    assert result.event_accuracy == pytest.approx((1.0 + 0.7 + 0.0) / 3, abs=0.01)
    assert result.critical_event_accuracy == pytest.approx((1.0 + 0.0) / 2, abs=0.01)


def test_chain_score_weights_critical_events():
    from longhorizon_bench.evaluation.metrics import compute_metrics

    event_scores = [
        {"event_id": "E01", "is_critical": True, "score": ActionScore(tool_score=1.0, param_scores={}, total=1.0)},
        {"event_id": "E02", "is_critical": False, "score": ActionScore(tool_score=1.0, param_scores={}, total=0.5)},
    ]
    result = compute_metrics(event_scores=event_scores, checkpoint_scores=[], violations=[])
    # weighted: (1.0*2 + 0.5*1) / (2+1) = 2.5/3 ≈ 0.833
    assert result.chain_score == pytest.approx(2.5 / 3, abs=0.01)


def test_chain_pass_strict():
    from longhorizon_bench.evaluation.metrics import compute_metrics

    event_scores = [
        {"event_id": "E01", "is_critical": False, "score": ActionScore(tool_score=1.0, param_scores={}, total=1.0)},
        {"event_id": "E02", "is_critical": False, "score": ActionScore(tool_score=0.0, param_scores={}, total=0.0)},
    ]
    result = compute_metrics(event_scores=event_scores, checkpoint_scores=[], violations=[])
    assert result.chain_pass is False


def test_chain_pass_true_when_all_tools_correct_no_violations():
    from longhorizon_bench.evaluation.metrics import compute_metrics

    event_scores = [
        {"event_id": "E01", "is_critical": False, "score": ActionScore(tool_score=1.0, param_scores={"a": 0.8}, total=0.88)},
        {"event_id": "E02", "is_critical": False, "score": ActionScore(tool_score=1.0, param_scores={"a": 1.0}, total=1.0)},
    ]
    result = compute_metrics(event_scores=event_scores, checkpoint_scores=[], violations=[])
    assert result.chain_pass is True


def test_chain_pass_false_with_violations():
    from longhorizon_bench.evaluation.metrics import compute_metrics

    event_scores = [
        {"event_id": "E01", "is_critical": False, "score": ActionScore(tool_score=1.0, param_scores={}, total=1.0)},
    ]
    violations = [Violation(rule_id="C01", event_id="E01", description="test")]
    result = compute_metrics(event_scores=event_scores, checkpoint_scores=[], violations=violations)
    assert result.chain_pass is False


def test_checkpoint_dimension_scores():
    from longhorizon_bench.evaluation.metrics import compute_metrics

    checkpoint_scores = [
        {"dimension": "long_term_memory", "score": 0.8},
        {"dimension": "long_term_memory", "score": 0.6},
        {"dimension": "consistency", "score": 1.0},
    ]
    result = compute_metrics(event_scores=[], checkpoint_scores=checkpoint_scores, violations=[])
    assert result.dimension_scores["long_term_memory"] == pytest.approx(0.7, abs=0.01)
    assert result.dimension_scores["consistency"] == pytest.approx(1.0, abs=0.01)


def test_consistency_violation_rate():
    from longhorizon_bench.evaluation.metrics import compute_metrics

    violations = [
        Violation(rule_id="C01", event_id="E01", description="test"),
        Violation(rule_id="C03", event_id="E05", description="test"),
    ]
    result = compute_metrics(event_scores=[], checkpoint_scores=[], violations=violations, applicable_rule_checks=10)
    assert result.consistency_violation_rate == pytest.approx(0.2, abs=0.01)
