import json
import pytest
from pathlib import Path


def test_filter_by_keywords():
    from longhorizon_bench.pipeline.downloader import filter_by_keywords
    docs = [
        {"text": "安全生产管理和压力容器检验是重点工作", "doc_id": "d1"},
        {"text": "今天天气真好", "doc_id": "d2"},
        {"text": "危险化学品存储和应急预案编制", "doc_id": "d3"},
        {"text": "安全生产制度短文", "doc_id": "d4"},
    ]
    result = filter_by_keywords(docs, min_keywords=2)
    assert len(result) == 2
    assert result[0]["doc_id"] == "d1"
    assert result[1]["doc_id"] == "d3"
    assert "keywords_matched" in result[0]
    assert len(result[0]["keywords_matched"]) >= 2


def test_filter_by_length():
    from longhorizon_bench.pipeline.downloader import filter_by_length
    docs = [
        {"text": "短", "doc_id": "d1"},
        {"text": "这是一段足够长的文本" * 100, "doc_id": "d2"},
    ]
    result = filter_by_length(docs, min_chars=500)
    assert len(result) == 1
    assert result[0]["doc_id"] == "d2"


def test_filter_regulations():
    from longhorizon_bench.pipeline.downloader import filter_regulations
    regs = [
        {"title": "安全生产法", "office": "全国人大", "status": "有效", "content": "内容"},
        {"title": "劳动法", "office": "全国人大", "status": "有效", "content": "内容"},
        {"title": "消防法", "office": "全国人大", "status": "已废止", "content": "内容"},
        {"title": "环境保护法", "office": "全国人大", "status": "有效", "content": "内容"},
    ]
    result = filter_regulations(regs)
    assert len(result) == 2
    titles = [r["title"] for r in result]
    assert "安全生产法" in titles
    assert "环境保护法" in titles


def test_dedup_by_simhash():
    from longhorizon_bench.pipeline.downloader import dedup_by_simhash
    docs = [
        {"text": "安全生产管理和压力容器检验是重点工作" * 20, "doc_id": "d1"},
        {"text": "安全生产管理和压力容器检验是重点工作" * 20, "doc_id": "d2"},
        {"text": "环境监测和污染防治是另一个重要领域" * 20, "doc_id": "d3"},
    ]
    result = dedup_by_simhash(docs)
    assert len(result) == 2
    ids = [d["doc_id"] for d in result]
    assert "d1" in ids
    assert "d3" in ids


def test_save_filtered_docs(tmp_path):
    from longhorizon_bench.pipeline.downloader import save_filtered_docs
    docs = [
        {"doc_id": "d1", "source": "test", "text": "内容", "keywords_matched": ["安全"], "char_count": 100},
    ]
    save_filtered_docs(docs, tmp_path / "raw_corpus" / "test")
    saved = list((tmp_path / "raw_corpus" / "test").glob("*.json"))
    assert len(saved) == 1
    with open(saved[0], encoding="utf-8") as f:
        data = json.load(f)
    assert data["doc_id"] == "d1"
