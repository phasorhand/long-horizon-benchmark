import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock


def _make_atoms(n: int) -> list[dict]:
    types = ["routine_inspection", "incident", "policy_change", "rectification", "training"]
    dims_pool = ["domain_knowledge", "multi_step_reasoning", "long_term_memory", "consistency"]
    atoms = []
    for i in range(n):
        atoms.append({
            "atom_id": f"ATOM-{i:03d}",
            "type": types[i % len(types)],
            "trigger": f"事件触发描述 {i}",
            "expected_tool": "no_action",
            "params": {"reason": {"value": "ok", "match": "contains"}},
            "evidence": {"required_facts": [f"fact_{i}"], "forbidden_actions": []},
            "dimensions": [dims_pool[i % len(dims_pool)]],
            "is_critical": i % 3 == 0,
        })
    return atoms


def test_validate_chain_dag():
    from longhorizon_bench.pipeline.chain_composer import validate_chain
    chain = {
        "events": [
            {"id": "E01", "depends_on": []},
            {"id": "E02", "depends_on": ["E01"]},
            {"id": "E03", "depends_on": ["E01", "E02"]},
        ],
        "checkpoints": [{"id": "CP01", "after": "E02"}],
    }
    errors = validate_chain(chain)
    assert errors == []


def test_validate_chain_cycle_detected():
    from longhorizon_bench.pipeline.chain_composer import validate_chain
    chain = {
        "events": [
            {"id": "E01", "depends_on": ["E02"]},
            {"id": "E02", "depends_on": ["E01"]},
        ],
        "checkpoints": [],
    }
    errors = validate_chain(chain)
    assert any("cycle" in e.lower() or "DAG" in e for e in errors)


def test_validate_chain_bad_checkpoint_interval():
    from longhorizon_bench.pipeline.chain_composer import validate_chain
    events = [{"id": f"E{i:02d}", "depends_on": [] if i == 1 else [f"E{i-1:02d}"]} for i in range(1, 12)]
    chain = {
        "events": events,
        "checkpoints": [
            {"id": "CP01", "after": "E01"},
            {"id": "CP02", "after": "E02"},
        ],
    }
    errors = validate_chain(chain)
    assert any("interval" in e.lower() or "间隔" in e for e in errors)


def test_compose_chain_from_atoms():
    from longhorizon_bench.pipeline.chain_composer import compose_chain_from_atoms
    atoms = _make_atoms(20)
    mock_client = MagicMock()
    mock_client.chat.return_value = """```yaml
scenario_id: IND-TEST
events:
  - id: E01
    atom_ref: ATOM-000
    depends_on: []
  - id: E02
    atom_ref: ATOM-001
    depends_on: [E01]
  - id: E03
    atom_ref: ATOM-002
    depends_on: [E01]
  - id: E04
    atom_ref: ATOM-003
    depends_on: [E02, E03]
  - id: E05
    atom_ref: ATOM-004
    depends_on: [E04]
  - id: E06
    atom_ref: ATOM-005
    depends_on: [E05]
  - id: E07
    atom_ref: ATOM-006
    depends_on: [E06]
  - id: E08
    atom_ref: ATOM-007
    depends_on: [E07]
  - id: E09
    atom_ref: ATOM-008
    depends_on: [E08]
  - id: E10
    atom_ref: ATOM-009
    depends_on: [E09]
  - id: E11
    atom_ref: ATOM-010
    depends_on: [E10]
  - id: E12
    atom_ref: ATOM-011
    depends_on: [E11]
  - id: E13
    atom_ref: ATOM-012
    depends_on: [E12]
  - id: E14
    atom_ref: ATOM-013
    depends_on: [E13]
  - id: E15
    atom_ref: ATOM-014
    depends_on: [E14]
checkpoints:
  - id: CP01
    after: E05
  - id: CP02
    after: E10
  - id: CP03
    after: E15
```"""
    chain = compose_chain_from_atoms(
        mock_client, atoms, scenario_id="IND-TEST",
        role="测试角色", subdomain="safety_production",
    )
    assert chain["scenario_id"] == "IND-TEST"
    assert len(chain["events"]) == 15
    assert len(chain["checkpoints"]) == 3


def test_save_chain(tmp_path):
    from longhorizon_bench.pipeline.chain_composer import save_chain
    chain = {"scenario_id": "IND-001", "events": []}
    save_chain(chain, tmp_path / "skeletons" / "chains")
    saved = tmp_path / "skeletons" / "chains" / "IND-001.yaml"
    assert saved.exists()
    with open(saved, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded["scenario_id"] == "IND-001"
