"""Industrial domain tool definitions."""
from longhorizon_bench.tools.registry import ToolDef

INDUSTRIAL_TOOLS: dict[str, ToolDef] = {
    "submit_inspection_report": ToolDef(params={"target": {"type": "str"}, "findings": {"type": "str"}, "risk_level": {"type": "enum", "values": ["low", "medium", "high", "critical"]}}),
    "update_safety_ledger": ToolDef(params={"item_id": {"type": "str"}, "status": {"type": "enum", "values": ["open", "in_progress", "resolved", "overdue"]}, "remarks": {"type": "str"}}),
    "file_incident_report": ToolDef(params={"incident_type": {"type": "enum", "values": ["injury", "leak", "fire", "equipment_failure", "violation"]}, "severity": {"type": "enum", "values": ["minor", "major", "critical"]}, "description": {"type": "str"}}),
    "issue_rectification_order": ToolDef(params={"target_dept": {"type": "str"}, "issues": {"type": "list[str]"}, "deadline": {"type": "date"}}),
    "request_equipment_shutdown": ToolDef(params={"equipment_id": {"type": "str"}, "reason": {"type": "str"}, "duration": {"type": "str"}}),
    "approve_work_permit": ToolDef(params={"permit_type": {"type": "enum", "values": ["hot_work", "confined_space", "height", "electrical"]}, "conditions": {"type": "list[str]"}, "approved": {"type": "bool"}}),
    "escalate_to_management": ToolDef(params={"issue_summary": {"type": "str"}, "urgency": {"type": "enum", "values": ["routine", "urgent", "emergency"]}, "recommendation": {"type": "str"}}),
    "notify_regulatory_body": ToolDef(params={"authority": {"type": "str"}, "event_type": {"type": "str"}, "details": {"type": "str"}}),
    "schedule_training": ToolDef(params={"topic": {"type": "str"}, "participants": {"type": "list[str]"}, "date": {"type": "date"}}),
    "allocate_budget": ToolDef(params={"item": {"type": "str"}, "amount": {"type": "float"}, "justification": {"type": "str"}}),
    "assign_personnel": ToolDef(params={"task": {"type": "str"}, "assignee": {"type": "str"}, "priority": {"type": "enum", "values": ["low", "medium", "high"]}}),
}
