import pytest
from longhorizon_bench.schema import ToolCall


def test_c01_status_regression_detected():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker
    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="update_safety_ledger", kwargs={"item_id": "ITEM-1", "status": "resolved", "remarks": ""})]),
        ("E05", [ToolCall(tool="update_safety_ledger", kwargs={"item_id": "ITEM-1", "status": "open", "remarks": ""})]),
    ]
    violations = checker.check_c01_status_regression(history)
    assert len(violations) == 1
    assert violations[0].rule_id == "C01"
    assert "ITEM-1" in violations[0].description


def test_c01_no_violation_on_forward_progress():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker
    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="update_safety_ledger", kwargs={"item_id": "ITEM-1", "status": "open", "remarks": ""})]),
        ("E05", [ToolCall(tool="update_safety_ledger", kwargs={"item_id": "ITEM-1", "status": "resolved", "remarks": ""})]),
    ]
    violations = checker.check_c01_status_regression(history)
    assert len(violations) == 0


def test_c02_risk_level_decrease_after_incident():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker
    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="submit_inspection_report", kwargs={"target": "A", "findings": "", "risk_level": "high"})]),
        ("E02", [ToolCall(tool="file_incident_report", kwargs={"incident_type": "leak", "severity": "major", "description": ""})]),
        ("E03", [ToolCall(tool="submit_inspection_report", kwargs={"target": "A", "findings": "", "risk_level": "low"})]),
    ]
    violations = checker.check_c02_risk_monotonicity(history)
    assert len(violations) == 1
    assert violations[0].rule_id == "C02"


def test_c02_no_violation_when_risk_stays_or_increases():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker
    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="submit_inspection_report", kwargs={"target": "A", "findings": "", "risk_level": "medium"})]),
        ("E02", [ToolCall(tool="file_incident_report", kwargs={"incident_type": "leak", "severity": "major", "description": ""})]),
        ("E03", [ToolCall(tool="submit_inspection_report", kwargs={"target": "A", "findings": "", "risk_level": "high"})]),
    ]
    violations = checker.check_c02_risk_monotonicity(history)
    assert len(violations) == 0


def test_c03_permit_without_inspection():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker
    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="approve_work_permit", kwargs={"permit_type": "hot_work", "conditions": [], "approved": True})]),
    ]
    violations = checker.check_c03_permit_prerequisite(history)
    assert len(violations) == 1
    assert violations[0].rule_id == "C03"


def test_c03_no_violation_with_prior_inspection():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker
    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="submit_inspection_report", kwargs={"target": "区域A", "findings": "合格", "risk_level": "low"})]),
        ("E02", [ToolCall(tool="approve_work_permit", kwargs={"permit_type": "hot_work", "conditions": [], "approved": True})]),
    ]
    violations = checker.check_c03_permit_prerequisite(history)
    assert len(violations) == 0


def test_c05_assignee_change_without_reason():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker
    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="assign_personnel", kwargs={"task": "巡检", "assignee": "张三", "priority": "high"})]),
        ("E05", [ToolCall(tool="assign_personnel", kwargs={"task": "巡检", "assignee": "李四", "priority": "high"})]),
    ]
    violations = checker.check_c05_assignee_consistency(history)
    assert len(violations) == 1
    assert violations[0].rule_id == "C05"


def test_c06_deadline_before_current_event():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker
    checker = ConsistencyChecker()
    history = [
        ("E10", [ToolCall(tool="issue_rectification_order", kwargs={
            "target_dept": "设备科", "issues": ["泄漏"], "deadline": "2025-01-01",
        })]),
    ]
    event_times = {"E10": "2025-06-01"}
    violations = checker.check_c06_timeline_validity(history, event_times)
    assert len(violations) == 1
    assert violations[0].rule_id == "C06"


def test_check_all_returns_combined():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker
    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="approve_work_permit", kwargs={"permit_type": "hot_work", "conditions": [], "approved": True})]),
    ]
    violations = checker.check_all(history, applicable_rules=["C03"])
    assert len(violations) >= 1
