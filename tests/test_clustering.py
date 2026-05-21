import json
import pytest
from pathlib import Path


@pytest.fixture
def corpus_dir(tmp_path):
    petro = tmp_path / "raw_corpus" / "petrochemical"
    petro.mkdir(parents=True)
    docs = [
        {"doc_id": "d1", "text": "压力容器检验周期管理规范，包括定期检验和年度检查"},
        {"doc_id": "d2", "text": "压力容器安全技术监察规程的执行要求"},
        {"doc_id": "d3", "text": "化学品泄漏应急预案编制指南"},
        {"doc_id": "d4", "text": "危险化学品泄漏事故应急处置流程"},
        {"doc_id": "d5", "text": "废气排放标准和环境监测技术要求"},
        {"doc_id": "d6", "text": "工业废气排放超标处理和污染防治措施"},
    ]
    for doc in docs:
        with open(petro / f"{doc['doc_id']}.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
    regs = tmp_path / "raw_corpus" / "regulations"
    regs.mkdir(parents=True)
    reg_doc = {"doc_id": "reg_001", "title": "安全生产法", "text": "安全生产管理的基本法律"}
    with open(regs / "reg_001.json", "w", encoding="utf-8") as f:
        json.dump(reg_doc, f, ensure_ascii=False)
    return tmp_path


def test_load_corpus(corpus_dir):
    from longhorizon_bench.pipeline.clustering import load_corpus
    docs = load_corpus(corpus_dir / "raw_corpus")
    assert len(docs) >= 6


def test_cluster_documents(corpus_dir):
    from longhorizon_bench.pipeline.clustering import load_corpus, cluster_documents
    docs = load_corpus(corpus_dir / "raw_corpus")
    clusters = cluster_documents(docs, n_clusters=3)
    assert len(clusters) == 3
    assert all(isinstance(c, list) for c in clusters.values())
    total_docs = sum(len(c) for c in clusters.values())
    assert total_docs == len(docs)


def test_build_topic_packs(corpus_dir):
    from longhorizon_bench.pipeline.clustering import load_corpus, cluster_documents, build_topic_packs
    docs = load_corpus(corpus_dir / "raw_corpus")
    clusters = cluster_documents(docs, n_clusters=3)
    regs = [d for d in docs if "reg_" in d.get("doc_id", "")]
    packs = build_topic_packs(clusters, regs, top_n=2)
    assert len(packs) == 3
    for pack in packs.values():
        assert "docs" in pack
        assert "regulations" in pack
        assert len(pack["docs"]) <= 2
