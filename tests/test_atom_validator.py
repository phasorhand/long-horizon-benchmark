import pytest


def test_valid_atom_passes():
    from longhorizon_bench.pipeline.atom_validator import validate_atom
    atom = {
        "atom_id": "ATOM-001",
        "expected_tool": "submit_inspection_report",
        "params": {
            "target": {"value": "3号车间", "match": "contains"},
            "risk_level": {"value": ["medium", "high"], "match": "enum"},
            "findings": {"keywords": ["压力容器"], "match": "keyword_coverage"},
        },
        "evidence": {
            "required_facts": ["压力容器检验周期为3年"],
            "forbidden_actions": ["approve_work_permit"],
        },
    }
    errors = validate_atom(atom)
    assert errors == []


def test_unknown_tool_fails():
    from longhorizon_bench.pipeline.atom_validator import validate_atom
    atom = {
        "atom_id": "ATOM-002",
        "expected_tool": "nonexistent_tool",
        "params": {},
        "evidence": {"required_facts": ["fact"], "forbidden_actions": []},
    }
    errors = validate_atom(atom)
    assert any("not in ToolRegistry" in e for e in errors)


def test_invalid_match_type_fails():
    from longhorizon_bench.pipeline.atom_validator import validate_atom
    atom = {
        "atom_id": "ATOM-003",
        "expected_tool": "submit_inspection_report",
        "params": {
            "target": {"value": "x", "match": "regex"},
        },
        "evidence": {"required_facts": ["fact"], "forbidden_actions": []},
    }
    errors = validate_atom(atom)
    assert any("match" in e.lower() for e in errors)


def test_missing_required_facts_fails():
    from longhorizon_bench.pipeline.atom_validator import validate_atom
    atom = {
        "atom_id": "ATOM-004",
        "expected_tool": "no_action",
        "params": {"reason": {"value": "test", "match": "contains"}},
        "evidence": {"required_facts": [], "forbidden_actions": []},
    }
    errors = validate_atom(atom)
    assert any("required_facts" in e for e in errors)


def test_unknown_param_key_fails():
    from longhorizon_bench.pipeline.atom_validator import validate_atom
    atom = {
        "atom_id": "ATOM-005",
        "expected_tool": "submit_inspection_report",
        "params": {
            "nonexistent_param": {"value": "x", "match": "exact"},
        },
        "evidence": {"required_facts": ["fact"], "forbidden_actions": []},
    }
    errors = validate_atom(atom)
    assert any("nonexistent_param" in e for e in errors)


def test_validate_batch():
    from longhorizon_bench.pipeline.atom_validator import validate_batch
    atoms = [
        {"atom_id": "A1", "expected_tool": "no_action", "params": {"reason": {"value": "ok", "match": "contains"}}, "evidence": {"required_facts": ["f"], "forbidden_actions": []}},
        {"atom_id": "A2", "expected_tool": "bad_tool", "params": {}, "evidence": {"required_facts": ["f"], "forbidden_actions": []}},
    ]
    passed, failed = validate_batch(atoms)
    assert len(passed) == 1
    assert len(failed) == 1
    assert passed[0]["atom_id"] == "A1"
    assert failed[0]["atom_id"] == "A2"
