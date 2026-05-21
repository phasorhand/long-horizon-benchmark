"""End-to-end pipeline test using mocked LLM calls."""

import json
import yaml
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from click.testing import CliRunner


@pytest.fixture
def pipeline_data_dir(tmp_path):
    """Set up a minimal raw_corpus to test the pipeline stages."""
    petro = tmp_path / "raw_corpus" / "petrochemical"
    petro.mkdir(parents=True)
    for i in range(6):
        doc = {
            "doc_id": f"petro_{i:03d}",
            "source": "IndustryCorpus2_petrochemical",
            "text": f"安全生产管理和压力容器检验规范文档{i}，包括隐患排查和设备检修" * 50,
            "keywords_matched": ["安全生产", "压力容器"],
            "char_count": 1000,
        }
        with open(petro / f"petro_{i:03d}.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)

    regs = tmp_path / "raw_corpus" / "regulations"
    regs.mkdir(parents=True)
    reg = {
        "doc_id": "reg_001",
        "source": "chinese-law-and-regulations",
        "title": "安全生产法",
        "text": "第三十四条 压力容器定期检验周期为3年。" * 20,
    }
    with open(regs / "reg_001.json", "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False)

    return tmp_path


def test_clustering_stage(pipeline_data_dir):
    from longhorizon_bench.pipeline.clustering import load_corpus, cluster_documents, build_topic_packs

    corpus = load_corpus(pipeline_data_dir / "raw_corpus")
    assert len(corpus) >= 6

    clusters = cluster_documents(corpus, n_clusters=2)
    assert len(clusters) == 2

    regs = [d for d in corpus if "reg_" in d.get("doc_id", "")]
    packs = build_topic_packs(clusters, regs)
    assert len(packs) == 2


def test_atom_generation_and_validation(pipeline_data_dir):
    from longhorizon_bench.pipeline.atom_generator import generate_atoms_for_pack
    from longhorizon_bench.pipeline.atom_validator import validate_batch

    mock_client = MagicMock()
    mock_client.chat.return_value = """```yaml
- atom_id: ATOM-e2e-001
  source_cluster: test
  type: routine_inspection
  trigger: "压力容器巡检"
  expected_tool: submit_inspection_report
  params:
    target: {value: "3号车间", match: contains}
    risk_level: {value: [medium, high], match: enum}
    findings: {keywords: [压力容器, 检验周期], match: keyword_coverage}
  evidence:
    required_facts: ["压力容器检验周期为3年"]
    forbidden_actions: [approve_work_permit]
  dimensions: [domain_knowledge]
  is_critical: true
```"""

    pack = {"cluster_id": 0, "docs": [{"text": "test", "doc_id": "t"}], "regulations": []}
    atoms = generate_atoms_for_pack(mock_client, pack)
    assert len(atoms) == 1

    passed, failed = validate_batch(atoms)
    assert len(passed) == 1
    assert len(failed) == 0


def test_assembler_produces_valid_scenario():
    from longhorizon_bench.pipeline.assembler import assemble_scenario
    from longhorizon_bench.schema import Scenario

    chain = {
        "scenario_id": "E2E-001", "domain": "industrial",
        "role": "测试角色", "difficulty": 1,
        "events": [
            {"id": "E01", "atom_ref": "A1", "depends_on": [], "is_critical": True, "dimensions": ["domain_knowledge"]},
        ],
        "checkpoints": [{"id": "CP01", "after": "E01", "queries_target_dimensions": ["long_term_memory"]}],
    }
    atoms = {
        "A1": {
            "type": "test", "expected_tool": "no_action",
            "params": {"reason": {"value": "ok", "match": "contains"}},
            "evidence": {"required_facts": ["f"], "forbidden_actions": [], "acceptable_actions": [{"tool": "no_action"}]},
        },
    }
    inputs = {"E01": "测试事件"}
    cp_queries = {"CP01": [{"query": "q", "expected_keywords": ["k"], "dimension": "long_term_memory", "match": "keyword_coverage"}]}

    result = assemble_scenario(chain, atoms, inputs, cp_queries, "背景" * 100)
    scenario = Scenario.model_validate(result)
    assert scenario.scenario_id == "E2E-001"
