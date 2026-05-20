"""Consistency rules C01-C06 for cross-event contradiction detection."""
from __future__ import annotations
from dataclasses import dataclass
from longhorizon_bench.schema import ToolCall

STATUS_ORDER = {"open": 0, "in_progress": 1, "resolved": 2, "overdue": 1}
RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

ActionHistory = list[tuple[str, list[ToolCall]]]

@dataclass
class Violation:
    rule_id: str
    event_id: str
    description: str

class ConsistencyChecker:
    def check_c01_status_regression(self, history: ActionHistory) -> list[Violation]:
        latest_status: dict[str, tuple[str, int]] = {}
        violations: list[Violation] = []
        for event_id, calls in history:
            for call in calls:
                if call.tool == "update_safety_ledger":
                    item = call.kwargs.get("item_id", "")
                    status = call.kwargs.get("status", "")
                    order = STATUS_ORDER.get(status, -1)
                    if item in latest_status:
                        prev_event, prev_order = latest_status[item]
                        if order < prev_order:
                            violations.append(Violation(
                                rule_id="C01", event_id=event_id,
                                description=f"Status regression for {item}: was order {prev_order} at {prev_event}, now {order} ({status})",
                            ))
                    latest_status[item] = (event_id, order)
        return violations

    def check_c02_risk_monotonicity(self, history: ActionHistory) -> list[Violation]:
        violations: list[Violation] = []
        incident_seen = False
        pre_incident_risk: dict[str, int] = {}
        for event_id, calls in history:
            for call in calls:
                if call.tool == "file_incident_report":
                    incident_seen = True
                if call.tool == "submit_inspection_report":
                    target = call.kwargs.get("target", "")
                    level = call.kwargs.get("risk_level", "")
                    order = RISK_ORDER.get(level, -1)
                    if not incident_seen:
                        pre_incident_risk[target] = max(pre_incident_risk.get(target, 0), order)
                    elif target in pre_incident_risk and order < pre_incident_risk[target]:
                        violations.append(Violation(
                            rule_id="C02", event_id=event_id,
                            description=f"Risk decreased for '{target}' after incident: was {pre_incident_risk[target]}, now {order} ({level})",
                        ))
        return violations

    def check_c03_permit_prerequisite(self, history: ActionHistory) -> list[Violation]:
        violations: list[Violation] = []
        inspection_done = False
        for event_id, calls in history:
            for call in calls:
                if call.tool == "submit_inspection_report":
                    inspection_done = True
                if call.tool == "approve_work_permit" and not inspection_done:
                    violations.append(Violation(
                        rule_id="C03", event_id=event_id,
                        description="Work permit approved without prior inspection report",
                    ))
        return violations

    def check_c04_rectification_closure(self, history: ActionHistory) -> list[Violation]:
        open_orders: dict[str, str] = {}
        for event_id, calls in history:
            for call in calls:
                if call.tool == "issue_rectification_order":
                    dept = call.kwargs.get("target_dept", "")
                    open_orders[dept] = event_id
                if call.tool == "update_safety_ledger":
                    status = call.kwargs.get("status", "")
                    if status == "resolved":
                        remarks = call.kwargs.get("remarks", "")
                        for dept in list(open_orders):
                            if dept in remarks or not open_orders:
                                del open_orders[dept]
        return [
            Violation(rule_id="C04", event_id=eid, description=f"Rectification order for '{dept}' never closed")
            for dept, eid in open_orders.items()
        ]

    def check_c05_assignee_consistency(self, history: ActionHistory) -> list[Violation]:
        violations: list[Violation] = []
        task_assignees: dict[str, tuple[str, str]] = {}
        for event_id, calls in history:
            for call in calls:
                if call.tool == "assign_personnel":
                    task = call.kwargs.get("task", "")
                    assignee = call.kwargs.get("assignee", "")
                    if task in task_assignees:
                        prev_event, prev_assignee = task_assignees[task]
                        if assignee != prev_assignee:
                            violations.append(Violation(
                                rule_id="C05", event_id=event_id,
                                description=f"Assignee changed for '{task}': '{prev_assignee}' -> '{assignee}' without explanation",
                            ))
                    task_assignees[task] = (event_id, assignee)
        return violations

    def check_c06_timeline_validity(self, history: ActionHistory, event_times: dict[str, str] | None = None) -> list[Violation]:
        if not event_times:
            return []
        violations: list[Violation] = []
        for event_id, calls in history:
            event_time = event_times.get(event_id)
            if not event_time:
                continue
            for call in calls:
                if call.tool == "issue_rectification_order":
                    deadline = call.kwargs.get("deadline", "")
                    if deadline and deadline < event_time:
                        violations.append(Violation(
                            rule_id="C06", event_id=event_id,
                            description=f"Deadline {deadline} is before event time {event_time}",
                        ))
        return violations

    def check_all(self, history: ActionHistory, applicable_rules: list[str] | None = None, event_times: dict[str, str] | None = None) -> list[Violation]:
        all_rules = {
            "C01": lambda: self.check_c01_status_regression(history),
            "C02": lambda: self.check_c02_risk_monotonicity(history),
            "C03": lambda: self.check_c03_permit_prerequisite(history),
            "C04": lambda: self.check_c04_rectification_closure(history),
            "C05": lambda: self.check_c05_assignee_consistency(history),
            "C06": lambda: self.check_c06_timeline_validity(history, event_times),
        }
        rules = applicable_rules or list(all_rules.keys())
        violations: list[Violation] = []
        for rule_id in rules:
            if rule_id in all_rules:
                violations.extend(all_rules[rule_id]())
        return violations
