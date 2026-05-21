import pytest
from unittest.mock import MagicMock


SAMPLE_REVIEW_RESPONSE = """```json
{
  "因果连贯性": 4,
  "证据可追溯": 5,
  "难度梯度": 4,
  "答案区分度": 4,
  "专业准确性": 4
}
```"""


def test_parse_review_scores():
    from longhorizon_bench.pipeline.committee_reviewer import parse_review_scores
    scores = parse_review_scores(SAMPLE_REVIEW_RESPONSE)
    assert scores["因果连贯性"] == 4
    assert scores["证据可追溯"] == 5
    assert len(scores) == 5


def test_check_committee_agreement():
    from longhorizon_bench.pipeline.committee_reviewer import check_committee_agreement
    scores_a = {"因果连贯性": 4, "证据可追溯": 5, "难度梯度": 4, "答案区分度": 4, "专业准确性": 4}
    scores_b = {"因果连贯性": 4, "证据可追溯": 4, "难度梯度": 4, "答案区分度": 4, "专业准确性": 4}
    agreed, disagreed = check_committee_agreement(scores_a, scores_b)
    assert len(agreed) == 5
    assert len(disagreed) == 0


def test_check_committee_disagreement():
    from longhorizon_bench.pipeline.committee_reviewer import check_committee_agreement
    scores_a = {"因果连贯性": 5, "证据可追溯": 2}
    scores_b = {"因果连贯性": 2, "证据可追溯": 5}
    agreed, disagreed = check_committee_agreement(scores_a, scores_b)
    assert len(disagreed) == 2


def test_committee_verdict_pass():
    from longhorizon_bench.pipeline.committee_reviewer import compute_verdict
    scores_a = {"因果连贯性": 4, "证据可追溯": 5, "难度梯度": 4, "答案区分度": 4, "专业准确性": 4}
    scores_b = {"因果连贯性": 4, "证据可追溯": 4, "难度梯度": 4, "答案区分度": 4, "专业准确性": 4}
    verdict = compute_verdict(scores_a, scores_b)
    assert verdict == "PASS"


def test_committee_verdict_fail_low_score():
    from longhorizon_bench.pipeline.committee_reviewer import compute_verdict
    scores_a = {"因果连贯性": 2, "证据可追溯": 2, "难度梯度": 2, "答案区分度": 2, "专业准确性": 2}
    scores_b = {"因果连贯性": 2, "证据可追溯": 2, "难度梯度": 2, "答案区分度": 2, "专业准确性": 2}
    verdict = compute_verdict(scores_a, scores_b)
    assert verdict in ("FAIL", "NEEDS_HUMAN_REVIEW")
