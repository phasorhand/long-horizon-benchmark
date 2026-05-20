# longhorizon_bench/evaluation/metrics.py
"""Multi-level metrics aggregation: L1 event, L2 dimension, L3 chain."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from longhorizon_bench.evaluation.scorer import ActionScore, CheckpointScore
from longhorizon_bench.evaluation.consistency import Violation


@dataclass
class EventScoreEntry:
    event_id: str
    is_critical: bool
    score: ActionScore


@dataclass
class CheckpointScoreEntry:
    dimension: str
    score: float


@dataclass
class EvalResult:
    event_accuracy: float
    critical_event_accuracy: float
    chain_score: float
    chain_pass: bool
    dimension_scores: dict[str, float]
    consistency_violation_rate: float
    violations: list[Violation]
    event_scores: list[dict]


def compute_metrics(
    event_scores: list[dict],
    checkpoint_scores: list[dict],
    violations: list[Violation],
    applicable_rule_checks: int = 0,
) -> EvalResult:
    all_totals = [e["score"].total for e in event_scores]
    critical_totals = [e["score"].total for e in event_scores if e["is_critical"]]

    event_accuracy = sum(all_totals) / len(all_totals) if all_totals else 0.0
    critical_event_accuracy = (
        sum(critical_totals) / len(critical_totals) if critical_totals else 0.0
    )

    weighted_sum = 0.0
    weight_total = 0.0
    for e in event_scores:
        w = 2.0 if e["is_critical"] else 1.0
        weighted_sum += e["score"].total * w
        weight_total += w
    chain_score = weighted_sum / weight_total if weight_total else 0.0

    all_tools_correct = all(e["score"].tool_score == 1.0 for e in event_scores)
    chain_pass = all_tools_correct and len(violations) == 0

    dim_groups: dict[str, list[float]] = defaultdict(list)
    for cp in checkpoint_scores:
        dim_groups[cp["dimension"]].append(cp["score"])
    dimension_scores = {
        dim: sum(vals) / len(vals) for dim, vals in dim_groups.items()
    }

    violation_rate = (
        len(violations) / applicable_rule_checks
        if applicable_rule_checks > 0
        else 0.0
    )

    return EvalResult(
        event_accuracy=event_accuracy,
        critical_event_accuracy=critical_event_accuracy,
        chain_score=chain_score,
        chain_pass=chain_pass,
        dimension_scores=dimension_scores,
        consistency_violation_rate=violation_rate,
        violations=violations,
        event_scores=event_scores,
    )
