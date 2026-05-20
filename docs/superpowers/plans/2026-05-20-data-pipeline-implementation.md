# Data Construction Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully automated data pipeline that downloads Chinese industrial corpora, extracts event chain skeletons via LLM, generates complete benchmark scenarios conforming to `schema.py`, and validates them through structural checks, LLM committee review, and simulation runs.

**Architecture:** Four-stage CLI pipeline (`download → extract → generate → validate`), each stage a click subcommand producing JSON/YAML intermediates to disk. LLM calls go through a unified `llm_client.py` abstraction supporting Claude (anthropic SDK) and DeepSeek (openai SDK with custom base_url). The pipeline reuses the existing `Scenario` schema, `ToolRegistry`, `EventScorer`, `ConsistencyChecker`, and `LongHorizonEnv` for validation.

**Tech Stack:** Python 3.10+, click, modelscope, datasets (HuggingFace), scikit-learn, simhash, pyyaml, anthropic, openai

---

## File Map

```
longhorizon_bench/pipeline/
├── __init__.py                 # Package marker
├── cli.py                      # click group: download/extract/generate/validate/run-all
├── llm_client.py               # Unified LLM call wrapper (Claude/DeepSeek)
├── downloader.py               # Stage 1: download + filter corpora
├── clustering.py               # Stage 2a: TF-IDF clustering
├── atom_generator.py           # Stage 2b: atomic event generation via LLM
├── atom_validator.py           # Stage 2c: structural validation of atoms
├── chain_composer.py           # Stage 2d: compose atoms into chains
├── bg_generator.py             # Stage 3a: background document generation
├── event_filler.py             # Stage 3b: event detail filling
├── assembler.py                # Stage 3c: assemble into Scenario JSON
├── structural_validator.py     # Stage 4 Layer 1: structural checks
├── committee_reviewer.py       # Stage 4 Layer 2: LLM committee review
└── simulation_validator.py     # Stage 4 Layer 3: PerfectAgent/BadAgent run

tests/
├── test_llm_client.py
├── test_downloader.py
├── test_clustering.py
├── test_atom_generator.py
├── test_atom_validator.py
├── test_chain_composer.py
├── test_bg_generator.py
├── test_event_filler.py
├── test_assembler.py
├── test_structural_validator.py
├── test_committee_reviewer.py
├── test_simulation_validator.py
└── test_pipeline_cli.py
```

---

### Task 1: Project Dependencies and Pipeline Package Scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `longhorizon_bench/pipeline/__init__.py`
- Create: `longhorizon_bench/pipeline/cli.py`
- Create: `tests/test_pipeline_cli.py`

- [ ] **Step 1: Write failing test for CLI entry point**

```python
# tests/test_pipeline_cli.py
from click.testing import CliRunner


def test_cli_group_exists():
    from longhorizon_bench.pipeline.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "download" in result.output
    assert "extract" in result.output
    assert "generate" in result.output
    assert "validate" in result.output
    assert "run-all" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_cli.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Update pyproject.toml with pipeline dependencies**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "longhorizon-bench"
version = "0.1.0"
description = "Chinese long-horizon benchmark for LLM evaluation"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0",
    "click>=8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]
openai = [
    "openai>=1.0",
]
pipeline = [
    "modelscope>=1.0",
    "datasets>=2.0",
    "scikit-learn>=1.0",
    "simhash-pysimhash>=0.1",
    "jieba>=0.42",
    "pyyaml>=6.0",
    "anthropic>=0.30",
    "openai>=1.0",
]

[project.scripts]
lhb-pipeline = "longhorizon_bench.pipeline.cli:cli"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create pipeline package and CLI**

```python
# longhorizon_bench/pipeline/__init__.py
"""Data construction pipeline for LongHorizon-Bench."""
```

```python
# longhorizon_bench/pipeline/cli.py
"""CLI entry point for the data construction pipeline."""

import click


@click.group()
def cli() -> None:
    """LongHorizon-Bench data construction pipeline."""
    pass


@cli.command()
@click.option("--data-dir", default="data", help="Base data directory")
@click.option("--subsets", default="petrochemical,mining,fire_safety_food_safety", help="Comma-separated IndustryCorpus2 subsets")
def download(data_dir: str, subsets: str) -> None:
    """Stage 1: Download and filter corpora."""
    click.echo(f"Downloading to {data_dir}/raw_corpus/ ...")
    from longhorizon_bench.pipeline.downloader import run_download
    run_download(data_dir=data_dir, subsets=subsets.split(","))
    click.echo("Download complete.")


@cli.command()
@click.option("--data-dir", default="data", help="Base data directory")
@click.option("--n-clusters", default=12, help="Number of topic clusters")
def extract(data_dir: str, n_clusters: int) -> None:
    """Stage 2: Extract atomic events and compose chains."""
    click.echo(f"Extracting skeletons from {data_dir}/raw_corpus/ ...")
    from longhorizon_bench.pipeline.chain_composer import run_extract
    run_extract(data_dir=data_dir, n_clusters=n_clusters)
    click.echo("Extraction complete.")


@cli.command()
@click.option("--data-dir", default="data", help="Base data directory")
def generate(data_dir: str) -> None:
    """Stage 3: Generate full scenarios from skeletons."""
    click.echo(f"Generating scenarios from {data_dir}/skeletons/ ...")
    from longhorizon_bench.pipeline.assembler import run_generate
    run_generate(data_dir=data_dir)
    click.echo("Generation complete.")


@cli.command("validate")
@click.option("--data-dir", default="data", help="Base data directory")
def validate_cmd(data_dir: str) -> None:
    """Stage 4: Validate generated scenarios."""
    click.echo(f"Validating scenarios in {data_dir}/scenarios/ ...")
    from longhorizon_bench.pipeline.simulation_validator import run_validate
    run_validate(data_dir=data_dir)
    click.echo("Validation complete.")


@cli.command("run-all")
@click.option("--data-dir", default="data", help="Base data directory")
@click.option("--subsets", default="petrochemical,mining,fire_safety_food_safety")
@click.option("--n-clusters", default=12)
def run_all(data_dir: str, subsets: str, n_clusters: int) -> None:
    """Run the full pipeline: download → extract → generate → validate."""
    from longhorizon_bench.pipeline.downloader import run_download
    from longhorizon_bench.pipeline.chain_composer import run_extract
    from longhorizon_bench.pipeline.assembler import run_generate
    from longhorizon_bench.pipeline.simulation_validator import run_validate

    click.echo("=== Stage 1: Download ===")
    run_download(data_dir=data_dir, subsets=subsets.split(","))
    click.echo("=== Stage 2: Extract ===")
    run_extract(data_dir=data_dir, n_clusters=n_clusters)
    click.echo("=== Stage 3: Generate ===")
    run_generate(data_dir=data_dir)
    click.echo("=== Stage 4: Validate ===")
    run_validate(data_dir=data_dir)
    click.echo("=== Pipeline complete ===")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_pipeline_cli.py -v`
Expected: PASS

- [ ] **Step 6: Run full suite to check no regressions**

Run: `pytest tests/ -v --tb=short`
Expected: 64 tests PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml longhorizon_bench/pipeline/__init__.py longhorizon_bench/pipeline/cli.py tests/test_pipeline_cli.py
git commit -m "feat: add pipeline CLI scaffold with download/extract/generate/validate commands"
```

---

### Task 2: Unified LLM Client

**Files:**
- Create: `longhorizon_bench/pipeline/llm_client.py`
- Create: `tests/test_llm_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm_client.py
import pytest
from unittest.mock import patch, MagicMock


def test_llm_client_claude_provider():
    from longhorizon_bench.pipeline.llm_client import LLMClient

    client = LLMClient(provider="claude", api_key="test-key")
    assert client.provider == "claude"
    assert client.model == "claude-sonnet-4-20250514"


def test_llm_client_deepseek_provider():
    from longhorizon_bench.pipeline.llm_client import LLMClient

    client = LLMClient(provider="deepseek", api_key="test-key")
    assert client.provider == "deepseek"
    assert client.model == "deepseek-chat"


def test_llm_client_custom_model():
    from longhorizon_bench.pipeline.llm_client import LLMClient

    client = LLMClient(provider="claude", api_key="test-key", model="claude-opus-4-20250514")
    assert client.model == "claude-opus-4-20250514"


def test_llm_client_invalid_provider():
    from longhorizon_bench.pipeline.llm_client import LLMClient

    with pytest.raises(ValueError, match="Unknown provider"):
        LLMClient(provider="invalid", api_key="test")


def test_llm_client_chat_claude():
    from longhorizon_bench.pipeline.llm_client import LLMClient

    client = LLMClient(provider="claude", api_key="test-key")
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="response text")]

    with patch.object(client, "_claude_client") as mock_client:
        mock_client.messages.create.return_value = mock_response
        result = client.chat(
            system="You are a helper.",
            messages=[{"role": "user", "content": "hello"}],
        )
    assert result == "response text"


def test_llm_client_chat_deepseek():
    from longhorizon_bench.pipeline.llm_client import LLMClient

    client = LLMClient(provider="deepseek", api_key="test-key")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="ds response"))]

    with patch.object(client, "_openai_client") as mock_client:
        mock_client.chat.completions.create.return_value = mock_response
        result = client.chat(
            system="You are a helper.",
            messages=[{"role": "user", "content": "hello"}],
        )
    assert result == "ds response"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL

- [ ] **Step 3: Implement llm_client.py**

```python
# longhorizon_bench/pipeline/llm_client.py
"""Unified LLM client supporting Claude and DeepSeek."""

from __future__ import annotations

from typing import Any


_DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-20250514",
    "deepseek": "deepseek-chat",
}

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class LLMClient:
    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str | None = None,
        temperature: float = 0.3,
    ) -> None:
        if provider not in _DEFAULT_MODELS:
            raise ValueError(f"Unknown provider: {provider}. Use 'claude' or 'deepseek'.")
        self.provider = provider
        self.model = model or _DEFAULT_MODELS[provider]
        self.temperature = temperature
        self._api_key = api_key
        self._claude_client: Any = None
        self._openai_client: Any = None

        if provider == "claude":
            self._init_claude()
        else:
            self._init_deepseek()

    def _init_claude(self) -> None:
        from anthropic import Anthropic
        self._claude_client = Anthropic(api_key=self._api_key)

    def _init_deepseek(self) -> None:
        from openai import OpenAI
        self._openai_client = OpenAI(
            api_key=self._api_key,
            base_url=_DEEPSEEK_BASE_URL,
        )

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
    ) -> str:
        if self.provider == "claude":
            return self._chat_claude(system, messages, max_tokens)
        return self._chat_deepseek(system, messages, max_tokens)

    def _chat_claude(
        self, system: str, messages: list[dict[str, str]], max_tokens: int
    ) -> str:
        response = self._claude_client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        return response.content[0].text

    def _chat_deepseek(
        self, system: str, messages: list[dict[str, str]], max_tokens: int
    ) -> str:
        all_messages = [{"role": "system", "content": system}] + messages
        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_client.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/pipeline/llm_client.py tests/test_llm_client.py
git commit -m "feat: add unified LLM client supporting Claude and DeepSeek"
```

---

### Task 3: Stage 1 — Data Downloader and Filter

**Files:**
- Create: `longhorizon_bench/pipeline/downloader.py`
- Create: `tests/test_downloader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_downloader.py
import json
import pytest
from pathlib import Path


def test_filter_by_keywords():
    from longhorizon_bench.pipeline.downloader import filter_by_keywords

    docs = [
        {"text": "安全生产管理和压力容器检验是重点工作", "doc_id": "d1"},
        {"text": "今天天气真好", "doc_id": "d2"},
        {"text": "危险化学品存储和应急预案编制", "doc_id": "d3"},
        {"text": "安全生产制度短文", "doc_id": "d4"},  # only 1 keyword, too few
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloader.py -v`
Expected: FAIL

- [ ] **Step 3: Implement downloader.py**

```python
# longhorizon_bench/pipeline/downloader.py
"""Stage 1: Download and filter industrial corpora and regulations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

INDUSTRY_KEYWORDS = [
    "安全生产", "压力容器", "危险化学品", "应急预案", "隐患排查",
    "设备检修", "作业许可", "安全评价", "职业病", "环境监测",
    "污染防治", "排放标准",
]

REGULATION_KEYWORDS = [
    "安全生产", "应急管理", "矿山", "消防", "危险化学品",
    "环境保护", "污染防治",
]


def filter_by_keywords(
    docs: list[dict[str, Any]],
    min_keywords: int = 2,
    keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    kws = keywords or INDUSTRY_KEYWORDS
    result: list[dict[str, Any]] = []
    for doc in docs:
        text = doc.get("text", "")
        matched = [kw for kw in kws if kw in text]
        if len(matched) >= min_keywords:
            doc_copy = dict(doc)
            doc_copy["keywords_matched"] = matched
            doc_copy["char_count"] = len(text)
            result.append(doc_copy)
    return result


def filter_by_length(
    docs: list[dict[str, Any]], min_chars: int = 500
) -> list[dict[str, Any]]:
    return [d for d in docs if len(d.get("text", "")) >= min_chars]


def filter_regulations(
    regs: list[dict[str, Any]],
    keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    kws = keywords or REGULATION_KEYWORDS
    result: list[dict[str, Any]] = []
    for reg in regs:
        title = reg.get("title", "")
        office = reg.get("office", "")
        status = reg.get("status", "")
        if status in ("已废止", "已失效"):
            continue
        if any(kw in title or kw in office for kw in kws):
            result.append(reg)
    return result


def dedup_by_simhash(
    docs: list[dict[str, Any]], threshold: int = 3
) -> list[dict[str, Any]]:
    from simhash import Simhash
    seen: list[tuple[Simhash, dict[str, Any]]] = []
    result: list[dict[str, Any]] = []
    for doc in docs:
        sh = Simhash(doc.get("text", ""))
        is_dup = any(sh.distance(s) <= threshold for s, _ in seen)
        if not is_dup:
            seen.append((sh, doc))
            result.append(doc)
    return result


def save_filtered_docs(
    docs: list[dict[str, Any]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        doc_id = doc.get("doc_id", f"doc_{id(doc)}")
        path = output_dir / f"{doc_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)


def run_download(data_dir: str, subsets: list[str]) -> None:
    base = Path(data_dir)

    # Download and filter IndustryCorpus2 subsets
    for subset in subsets:
        short_name = subset.replace("fire_safety_food_safety", "fire_safety")
        output_path = base / "raw_corpus" / short_name
        if output_path.exists() and any(output_path.glob("*.json")):
            continue
        try:
            from modelscope.msdatasets import MsDataset
            ds = MsDataset.load(
                f"BAAI/IndustryCorpus2_{subset}",
                split="train",
                cache_dir=str(base / ".cache"),
            )
            raw_docs = [{"doc_id": f"{short_name}_{i:06d}", "source": f"IndustryCorpus2_{subset}", "text": row.get("text", "")} for i, row in enumerate(ds)]
        except Exception:
            raw_docs = []

        filtered = filter_by_keywords(raw_docs)
        filtered = filter_by_length(filtered)
        filtered = dedup_by_simhash(filtered)
        save_filtered_docs(filtered, output_path)

    # Download and filter regulations
    reg_path = base / "raw_corpus" / "regulations"
    if not reg_path.exists() or not any(reg_path.glob("*.json")):
        try:
            from datasets import load_dataset
            ds = load_dataset("twang2218/chinese-law-and-regulations", split="train")
            raw_regs = [{"doc_id": f"reg_{i:06d}", "source": "chinese-law-and-regulations", "title": row.get("title", ""), "office": row.get("office", ""), "status": row.get("status", ""), "text": row.get("content", "")} for i, row in enumerate(ds)]
        except Exception:
            raw_regs = []

        filtered_regs = filter_regulations(raw_regs)
        for reg in filtered_regs:
            reg["keywords_matched"] = [kw for kw in REGULATION_KEYWORDS if kw in reg.get("title", "") or kw in reg.get("office", "")]
            reg["char_count"] = len(reg.get("text", ""))
        save_filtered_docs(filtered_regs, reg_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloader.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/pipeline/downloader.py tests/test_downloader.py
git commit -m "feat: add data downloader with keyword/length/regulation filters"
```

---

### Task 4: Stage 2a — TF-IDF Clustering

**Files:**
- Create: `longhorizon_bench/pipeline/clustering.py`
- Create: `tests/test_clustering.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_clustering.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_clustering.py -v`
Expected: FAIL

- [ ] **Step 3: Implement clustering.py**

```python
# longhorizon_bench/pipeline/clustering.py
"""Stage 2a: TF-IDF clustering to group documents by topic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def load_corpus(raw_corpus_dir: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for json_file in sorted(raw_corpus_dir.rglob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            docs.append(json.load(f))
    return docs


def cluster_documents(
    docs: list[dict[str, Any]], n_clusters: int = 12
) -> dict[int, list[dict[str, Any]]]:
    n_clusters = min(n_clusters, len(docs))
    texts = [" ".join(jieba.cut(d.get("text", ""))) for d in docs]

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    clusters: dict[int, list[dict[str, Any]]] = {i: [] for i in range(n_clusters)}
    for doc, label in zip(docs, labels):
        clusters[label].append(doc)
    return clusters


def build_topic_packs(
    clusters: dict[int, list[dict[str, Any]]],
    regulations: list[dict[str, Any]],
    top_n: int = 5,
) -> dict[int, dict[str, Any]]:
    packs: dict[int, dict[str, Any]] = {}
    for cluster_id, docs in clusters.items():
        top_docs = sorted(docs, key=lambda d: d.get("char_count", len(d.get("text", ""))), reverse=True)[:top_n]
        cluster_text = " ".join(d.get("text", "") for d in top_docs)
        matched_regs = []
        for reg in regulations:
            reg_title = reg.get("title", "")
            if any(kw in cluster_text for kw in reg_title[:4].split()):
                matched_regs.append(reg)
        if not matched_regs:
            matched_regs = regulations[:2]
        packs[cluster_id] = {
            "cluster_id": cluster_id,
            "docs": top_docs,
            "regulations": matched_regs[:3],
        }
    return packs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_clustering.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/pipeline/clustering.py tests/test_clustering.py
git commit -m "feat: add TF-IDF document clustering for topic extraction"
```

---

### Task 5: Stage 2b — Atomic Event Generator

**Files:**
- Create: `longhorizon_bench/pipeline/atom_generator.py`
- Create: `tests/test_atom_generator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_atom_generator.py
import pytest
from unittest.mock import MagicMock, patch


SAMPLE_LLM_RESPONSE = """```yaml
- atom_id: ATOM-test-001
  source_cluster: test_cluster
  type: routine_inspection
  trigger: "设备科提交巡检报告，发现压力容器接近检验周期"
  expected_tool: submit_inspection_report
  params:
    target: {value: "3号车间", match: contains}
    risk_level: {value: [medium, high], match: enum}
    findings: {keywords: [压力容器, 检验周期], match: keyword_coverage}
  evidence:
    required_facts: ["压力容器定期检验周期为3年"]
    forbidden_actions: [approve_work_permit]
  dimensions: [domain_knowledge]
  is_critical: true
- atom_id: ATOM-test-002
  source_cluster: test_cluster
  type: incident
  trigger: "车间报告设备异常振动"
  expected_tool: file_incident_report
  params:
    incident_type: {value: equipment_failure, match: exact}
    severity: {value: major, match: exact}
    description: {value: "设备异常振动", match: contains}
  evidence:
    required_facts: ["设备运行记录"]
    forbidden_actions: []
  dimensions: [domain_knowledge, multi_step_reasoning]
  is_critical: false
```"""


def test_parse_atoms_from_llm_response():
    from longhorizon_bench.pipeline.atom_generator import parse_atoms_from_response

    atoms = parse_atoms_from_response(SAMPLE_LLM_RESPONSE)
    assert len(atoms) == 2
    assert atoms[0]["atom_id"] == "ATOM-test-001"
    assert atoms[0]["expected_tool"] == "submit_inspection_report"
    assert atoms[1]["type"] == "incident"


def test_build_atom_prompt():
    from longhorizon_bench.pipeline.atom_generator import build_atom_prompt

    topic_pack = {
        "cluster_id": 0,
        "docs": [{"text": "压力容器检验周期为3年", "doc_id": "d1"}],
        "regulations": [{"title": "安全生产法", "text": "安全生产管理条例"}],
    }
    system, user = build_atom_prompt(topic_pack, n_atoms=3)
    assert "3" in user
    assert "压力容器" in user
    assert "submit_inspection_report" in system or "ToolRegistry" in system


def test_generate_atoms_for_pack():
    from longhorizon_bench.pipeline.atom_generator import generate_atoms_for_pack

    mock_client = MagicMock()
    mock_client.chat.return_value = SAMPLE_LLM_RESPONSE

    topic_pack = {
        "cluster_id": 0,
        "docs": [{"text": "压力容器", "doc_id": "d1"}],
        "regulations": [],
    }
    atoms = generate_atoms_for_pack(mock_client, topic_pack, n_atoms=3)
    assert len(atoms) == 2
    mock_client.chat.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_atom_generator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement atom_generator.py**

```python
# longhorizon_bench/pipeline/atom_generator.py
"""Stage 2b: Generate atomic events from topic packs via LLM."""

from __future__ import annotations

from typing import Any

import yaml

from longhorizon_bench.tools import COMMON_TOOLS, INDUSTRIAL_TOOLS


def _tool_list_text() -> str:
    lines: list[str] = []
    for name, tdef in {**COMMON_TOOLS, **INDUSTRIAL_TOOLS}.items():
        params_desc = ", ".join(
            f"{k}: {v.get('type', 'str')}" + (f" ({v.get('values', '')})" if v.get("values") else "")
            for k, v in tdef.params.items()
        )
        lines.append(f"- {name}({params_desc})")
    return "\n".join(lines)


def build_atom_prompt(
    topic_pack: dict[str, Any], n_atoms: int = 5
) -> tuple[str, str]:
    tool_text = _tool_list_text()

    system = f"""你是一名中国工业安全领域的资深专家，正在为benchmark构造原子事件。

可用工具列表（每个事件必须从中选择一个 expected_tool）：
{tool_text}

输出要求：
- 输出 YAML 列表，每个元素包含：atom_id, source_cluster, type, trigger, expected_tool, params, evidence, dimensions, is_critical
- params 中的 match 类型只能是：exact, contains, enum, keyword_coverage
- evidence 必须包含 required_facts（至少1条）和 forbidden_actions
- dimensions 从以下选择：domain_knowledge, multi_step_reasoning, long_term_memory, consistency, priority_judgment, information_integration
- 用 ```yaml ``` 包裹输出"""

    doc_texts = "\n\n".join(
        f"[{d.get('doc_id', '')}] {d.get('text', '')[:1000]}"
        for d in topic_pack.get("docs", [])
    )
    reg_texts = "\n\n".join(
        f"[{r.get('title', '')}] {r.get('text', '')[:500]}"
        for r in topic_pack.get("regulations", [])
    )

    user = f"""基于以下文档和法规，生成 {n_atoms} 个独立的原子事件：

## 文档素材
{doc_texts}

## 相关法规
{reg_texts}

请生成 {n_atoms} 个原子事件，覆盖不同的事件类型（巡检、事故、政策变更、整改等）。"""

    return system, user


def parse_atoms_from_response(response: str) -> list[dict[str, Any]]:
    start = response.find("```yaml")
    end = response.find("```", start + 7) if start >= 0 else -1
    if start >= 0 and end >= 0:
        yaml_text = response[start + 7:end].strip()
    else:
        yaml_text = response.strip()

    parsed = yaml.safe_load(yaml_text)
    if isinstance(parsed, list):
        return parsed
    return []


def generate_atoms_for_pack(
    llm_client: Any,
    topic_pack: dict[str, Any],
    n_atoms: int = 5,
) -> list[dict[str, Any]]:
    system, user = build_atom_prompt(topic_pack, n_atoms)
    response = llm_client.chat(
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=4096,
    )
    return parse_atoms_from_response(response)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_atom_generator.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/pipeline/atom_generator.py tests/test_atom_generator.py
git commit -m "feat: add atomic event generator with LLM prompt and YAML parsing"
```

---

### Task 6: Stage 2c — Atomic Event Validator

**Files:**
- Create: `longhorizon_bench/pipeline/atom_validator.py`
- Create: `tests/test_atom_validator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_atom_validator.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_atom_validator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement atom_validator.py**

```python
# longhorizon_bench/pipeline/atom_validator.py
"""Stage 2c: Validate atomic events against ToolRegistry and schema rules."""

from __future__ import annotations

from typing import Any

from longhorizon_bench.tools import build_registry

VALID_MATCH_TYPES = {"exact", "contains", "enum", "keyword_coverage"}

_registry = build_registry("industrial")


def validate_atom(atom: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tool_name = atom.get("expected_tool", "")
    if tool_name not in _registry:
        errors.append(f"Tool '{tool_name}' not in ToolRegistry")
    else:
        tool_def = _registry.get(tool_name)
        if tool_def:
            for param_key in atom.get("params", {}):
                if param_key not in tool_def.params:
                    errors.append(f"Param '{param_key}' not defined for tool '{tool_name}'")

    for param_key, param_spec in atom.get("params", {}).items():
        match_type = param_spec.get("match", "")
        if match_type not in VALID_MATCH_TYPES:
            errors.append(f"Invalid match type '{match_type}' for param '{param_key}'. Valid: {VALID_MATCH_TYPES}")

    evidence = atom.get("evidence", {})
    required_facts = evidence.get("required_facts", [])
    if not required_facts:
        errors.append("evidence.required_facts must have at least 1 entry")

    return errors


def validate_batch(
    atoms: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for atom in atoms:
        errors = validate_atom(atom)
        if errors:
            atom["_validation_errors"] = errors
            failed.append(atom)
        else:
            passed.append(atom)
    return passed, failed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_atom_validator.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/pipeline/atom_validator.py tests/test_atom_validator.py
git commit -m "feat: add atomic event validator against ToolRegistry"
```

---

### Task 7: Stage 2d — Chain Composer

**Files:**
- Create: `longhorizon_bench/pipeline/chain_composer.py`
- Create: `tests/test_chain_composer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_chain_composer.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chain_composer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement chain_composer.py**

```python
# longhorizon_bench/pipeline/chain_composer.py
"""Stage 2d: Compose validated atoms into event chains via LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def validate_chain(chain: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    events = chain.get("events", [])
    event_ids = {e["id"] for e in events}

    # Check DAG — no cycles via topological sort attempt
    adj: dict[str, list[str]] = {e["id"]: e.get("depends_on", []) for e in events}
    visited: set[str] = set()
    in_stack: set[str] = set()

    def has_cycle(node: str) -> bool:
        if node in in_stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        in_stack.add(node)
        for dep in adj.get(node, []):
            if dep in event_ids and has_cycle(dep):
                return True
        in_stack.discard(node)
        return False

    for eid in event_ids:
        if has_cycle(eid):
            errors.append(f"DAG cycle detected involving {eid}")
            break

    # Check depends_on references
    for e in events:
        for dep in e.get("depends_on", []):
            if dep not in event_ids:
                errors.append(f"Event {e['id']} depends on unknown {dep}")

    # Check checkpoint intervals
    checkpoints = chain.get("checkpoints", [])
    if checkpoints and events:
        id_to_index = {e["id"]: i for i, e in enumerate(events)}
        cp_positions = sorted(
            id_to_index.get(cp.get("after", ""), 0) for cp in checkpoints
        )
        prev = -1
        for pos in cp_positions:
            interval = pos - prev
            if interval < 3:
                errors.append(f"Checkpoint interval too small ({interval} < 3)")
            prev = pos

    return errors


def compose_chain_from_atoms(
    llm_client: Any,
    atoms: list[dict[str, Any]],
    scenario_id: str,
    role: str,
    subdomain: str,
) -> dict[str, Any]:
    atom_descs = "\n".join(
        f"- {a['atom_id']}: type={a.get('type', '')}, trigger={a.get('trigger', '')}, "
        f"tool={a.get('expected_tool', '')}, critical={a.get('is_critical', False)}, "
        f"dims={a.get('dimensions', [])}"
        for a in atoms
    )

    system = """你是一名benchmark设计专家。将原子事件组合成一条完整的事件链。

要求：
1. 选择 15-25 个原子事件，按因果逻辑排列
2. 设计 depends_on 依赖关系（DAG，无环）
3. 插入 3-5 个检查点，间隔 3-7 个事件
4. 输出 YAML 格式，用 ```yaml ``` 包裹"""

    user = f"""场景ID: {scenario_id}
角色: {role}
子领域: {subdomain}

可用原子事件池：
{atom_descs}

请组合成一条完整的事件链。"""

    response = llm_client.chat(
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=4096,
    )

    start = response.find("```yaml")
    end = response.find("```", start + 7) if start >= 0 else -1
    if start >= 0 and end >= 0:
        yaml_text = response[start + 7:end].strip()
    else:
        yaml_text = response.strip()

    chain = yaml.safe_load(yaml_text)
    if not isinstance(chain, dict):
        chain = {}
    chain.setdefault("scenario_id", scenario_id)
    chain.setdefault("domain", "industrial")
    chain.setdefault("subdomain", subdomain)
    chain.setdefault("role", role)
    return chain


def save_chain(chain: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{chain['scenario_id']}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(chain, f, allow_unicode=True, default_flow_style=False)
    return path


def run_extract(data_dir: str, n_clusters: int = 12) -> None:
    from longhorizon_bench.pipeline.clustering import load_corpus, cluster_documents, build_topic_packs
    from longhorizon_bench.pipeline.atom_generator import generate_atoms_for_pack
    from longhorizon_bench.pipeline.atom_validator import validate_batch
    from longhorizon_bench.pipeline.llm_client import LLMClient
    import os

    base = Path(data_dir)
    corpus = load_corpus(base / "raw_corpus")
    regs = [d for d in corpus if d.get("source", "").startswith("chinese-law")]
    industry_docs = [d for d in corpus if not d.get("source", "").startswith("chinese-law")]

    clusters = cluster_documents(industry_docs, n_clusters=min(n_clusters, len(industry_docs)))
    packs = build_topic_packs(clusters, regs)

    claude = LLMClient(provider="claude", api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    all_atoms: list[dict] = []
    for pack in packs.values():
        atoms = generate_atoms_for_pack(claude, pack)
        passed, _failed = validate_batch(atoms)
        all_atoms.extend(passed)

    atoms_dir = base / "skeletons" / "atoms"
    atoms_dir.mkdir(parents=True, exist_ok=True)
    for atom in all_atoms:
        path = atoms_dir / f"{atom['atom_id']}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(atom, f, allow_unicode=True)

    scenarios = [
        ("IND-001", "化工厂安全生产管理员", "safety_production"),
        ("IND-002", "化工厂安全生产管理员", "safety_production"),
        ("IND-003", "环保合规主管", "environmental_compliance"),
    ]
    chains_dir = base / "skeletons" / "chains"
    for sid, role, subdomain in scenarios:
        chain = compose_chain_from_atoms(claude, all_atoms, sid, role, subdomain)
        errors = validate_chain(chain)
        if not errors:
            save_chain(chain, chains_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chain_composer.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/pipeline/chain_composer.py tests/test_chain_composer.py
git commit -m "feat: add chain composer with DAG validation and LLM-based composition"
```

---

### Task 8: Stage 3a — Background Document Generator

**Files:**
- Create: `longhorizon_bench/pipeline/bg_generator.py`
- Create: `tests/test_bg_generator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_bg_generator.py
import pytest
from unittest.mock import MagicMock
from pathlib import Path


def test_extract_regulation_core():
    from longhorizon_bench.pipeline.bg_generator import extract_regulation_core

    regs = [
        {"title": "安全生产法", "text": "第三十四条 压力容器定期检验周期为3年，检验期间应降负荷运行。第三十五条 危险化学品储存间距不小于15米。"},
        {"title": "消防法", "text": "第二十条 消防通道宽度不小于4米。"},
    ]
    core = extract_regulation_core(regs, max_chars=5000)
    assert "压力容器" in core
    assert "消防" in core
    assert len(core) <= 5000


def test_generate_background():
    from longhorizon_bench.pipeline.bg_generator import generate_background

    mock_client = MagicMock()
    mock_client.chat.return_value = "华安化工有限公司成立于2005年..." + "扩展内容" * 1000

    chain = {
        "scenario_id": "IND-001",
        "role": "化工厂安全生产管理员",
        "events": [
            {"id": "E01", "trigger": "压力容器巡检"},
        ],
    }
    reg_core = "压力容器定期检验周期为3年。"

    bg_text = generate_background(mock_client, chain, reg_core)
    assert len(bg_text) > 0
    mock_client.chat.assert_called_once()


def test_save_background_docs(tmp_path):
    from longhorizon_bench.pipeline.bg_generator import save_background_docs

    save_background_docs("IND-001", "这是背景文档内容", tmp_path)
    doc_path = tmp_path / "background_docs" / "IND-001" / "background.txt"
    assert doc_path.exists()
    assert "背景文档" in doc_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bg_generator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement bg_generator.py**

```python
# longhorizon_bench/pipeline/bg_generator.py
"""Stage 3a: Generate layered background documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def extract_regulation_core(
    regulations: list[dict[str, Any]], max_chars: int = 5000
) -> str:
    parts: list[str] = []
    total = 0
    for reg in regulations:
        title = reg.get("title", "")
        text = reg.get("text", "")
        chunk = f"【{title}】\n{text}"
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)


def generate_background(
    llm_client: Any,
    chain: dict[str, Any],
    regulation_core: str,
) -> str:
    events_desc = "\n".join(
        f"- {e.get('id', '')}: {e.get('trigger', '')}"
        for e in chain.get("events", [])
    )

    system = """你是一名工业安全领域的技术文档撰写专家。基于提供的法规核心内容，扩展生成一份完整的企业背景文档。

要求：
1. 保留法规中的条款编号和核心数据（检验周期、间距标准等）
2. 改写为虚构企业的背景材料（公司名、人名均为虚构）
3. 包括：公司概况、组织架构、设备台账、检验记录、历史事故、整改情况
4. 每个数据点都应当能被事件链中的 evidence 引用
5. 总字数控制在15000-20000字"""

    user = f"""场景角色：{chain.get('role', '')}
场景ID：{chain.get('scenario_id', '')}

## 法规核心内容（底层，必须保留的数据点）
{regulation_core}

## 事件链概览（背景文档需支撑这些事件）
{events_desc}

请生成完整的企业背景文档。"""

    return llm_client.chat(
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=16384,
    )


def save_background_docs(
    scenario_id: str, bg_text: str, data_dir: Path
) -> Path:
    doc_dir = data_dir / "background_docs" / scenario_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    path = doc_dir / "background.txt"
    path.write_text(bg_text, encoding="utf-8")
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bg_generator.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/pipeline/bg_generator.py tests/test_bg_generator.py
git commit -m "feat: add background document generator with regulation core extraction"
```

---

### Task 9: Stage 3b — Event Detail Filler

**Files:**
- Create: `longhorizon_bench/pipeline/event_filler.py`
- Create: `tests/test_event_filler.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_event_filler.py
import pytest
from unittest.mock import MagicMock


def test_fill_action_event():
    from longhorizon_bench.pipeline.event_filler import fill_action_event

    mock_client = MagicMock()
    mock_client.chat.return_value = "收到设备科提交的3号车间季度巡检报告，其中提到1台压力容器接近检验周期，根据《压力容器安全技术监察规程》第34条，应在到期前3个月启动检验申请。"

    atom = {
        "atom_id": "ATOM-001",
        "type": "routine_inspection",
        "trigger": "设备老化报告",
        "expected_tool": "submit_inspection_report",
        "params": {
            "target": {"value": "3号车间", "match": "contains"},
            "findings": {"keywords": ["压力容器", "检验周期"], "match": "keyword_coverage"},
        },
        "evidence": {"required_facts": ["压力容器检验周期为3年"], "forbidden_actions": []},
        "dimensions": ["domain_knowledge"],
        "is_critical": True,
    }
    event_input = fill_action_event(mock_client, atom, "背景文档内容", [])
    assert isinstance(event_input, str)
    assert len(event_input) > 0


def test_fill_checkpoint_queries():
    from longhorizon_bench.pipeline.event_filler import fill_checkpoint_queries

    mock_client = MagicMock()
    mock_client.chat.return_value = """```yaml
- query: "E01中提到的压力容器情况是什么？"
  expected_keywords: ["压力容器", "检验周期"]
  dimension: long_term_memory
  match: keyword_coverage
- query: "前序事件中安全隐患的处理结果如何？"
  expected_keywords: ["整改", "隐患"]
  dimension: consistency
  match: keyword_coverage
```"""

    prior_events = [
        {"event_id": "E01", "input": "压力容器接近检验周期"},
        {"event_id": "E02", "input": "安全隐患排查整改"},
    ]
    queries = fill_checkpoint_queries(
        mock_client, prior_events, target_dimensions=["long_term_memory", "consistency"]
    )
    assert len(queries) == 2
    assert queries[0]["dimension"] == "long_term_memory"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_event_filler.py -v`
Expected: FAIL

- [ ] **Step 3: Implement event_filler.py**

```python
# longhorizon_bench/pipeline/event_filler.py
"""Stage 3b: Fill event details using LLM."""

from __future__ import annotations

from typing import Any

import yaml


def fill_action_event(
    llm_client: Any,
    atom: dict[str, Any],
    background_text: str,
    prior_inputs: list[dict[str, Any]],
) -> str:
    param_hints = []
    for name, spec in atom.get("params", {}).items():
        if spec.get("keywords"):
            param_hints.append(f"必须包含关键词: {spec['keywords']}")
        elif spec.get("value"):
            param_hints.append(f"必须提及: {spec['value']}")

    prior_text = "\n".join(
        f"[{p.get('event_id', '')}] {p.get('input', '')[:200]}"
        for p in prior_inputs[-5:]
    )

    system = """你是一名工业安全事件描述撰写专家。生成一段200-500字的事件描述。

要求：
1. 引用背景文档中的具体条款或数据
2. 包含后续评分所需的关键信息
3. 语言风格为正式工作通知/报告
4. 只输出事件描述文本，不要标题或格式标记"""

    user = f"""事件类型: {atom.get('type', '')}
触发: {atom.get('trigger', '')}

背景文档（节选）:
{background_text[:3000]}

前序事件:
{prior_text}

评分关键信息（必须包含在描述中）:
{chr(10).join(param_hints)}

请生成事件描述文本。"""

    return llm_client.chat(
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=1024,
    )


def fill_checkpoint_queries(
    llm_client: Any,
    prior_events: list[dict[str, Any]],
    target_dimensions: list[str] | None = None,
) -> list[dict[str, Any]]:
    dims = target_dimensions or ["long_term_memory"]
    events_text = "\n".join(
        f"[{e.get('event_id', '')}] {e.get('input', '')[:200]}"
        for e in prior_events
    )

    system = """你是一名benchmark检查点设计专家。设计回溯问题来测试模型的长期记忆和一致性。

输出 YAML 列表，用 ```yaml ``` 包裹。每个问题包含:
- query: 问题文本
- expected_keywords: 正确答案应包含的关键词列表
- dimension: 评测维度
- match: keyword_coverage"""

    user = f"""基于以下前序事件，设计 {len(dims)} 个检查点问题。

前序事件:
{events_text}

要求覆盖的维度: {dims}
问题的答案必须依赖3个以上事件之前的信息。"""

    response = llm_client.chat(
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=2048,
    )

    start = response.find("```yaml")
    end = response.find("```", start + 7) if start >= 0 else -1
    if start >= 0 and end >= 0:
        yaml_text = response[start + 7:end].strip()
    else:
        yaml_text = response.strip()

    parsed = yaml.safe_load(yaml_text)
    return parsed if isinstance(parsed, list) else []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_event_filler.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/pipeline/event_filler.py tests/test_event_filler.py
git commit -m "feat: add event detail filler with action and checkpoint generation"
```

---

### Task 10: Stage 3c — Scenario Assembler

**Files:**
- Create: `longhorizon_bench/pipeline/assembler.py`
- Create: `tests/test_assembler.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_assembler.py
import json
import pytest
from pathlib import Path


def _make_chain() -> dict:
    return {
        "scenario_id": "IND-TEST",
        "domain": "industrial",
        "subdomain": "safety_production",
        "role": "化工厂安全生产管理员",
        "difficulty": 3,
        "events": [
            {"id": "E01", "atom_ref": "ATOM-001", "depends_on": [], "is_critical": True, "dimensions": ["domain_knowledge"]},
            {"id": "E02", "atom_ref": "ATOM-002", "depends_on": ["E01"], "is_critical": False, "dimensions": ["multi_step_reasoning"]},
        ],
        "checkpoints": [
            {"id": "CP01", "after": "E02", "queries_target_dimensions": ["long_term_memory"]},
        ],
    }


def _make_atoms() -> dict:
    return {
        "ATOM-001": {
            "atom_id": "ATOM-001",
            "type": "routine_inspection",
            "expected_tool": "submit_inspection_report",
            "params": {
                "target": {"value": "3号车间", "match": "contains"},
                "risk_level": {"value": ["medium", "high"], "match": "enum"},
            },
            "evidence": {
                "required_facts": ["压力容器检验周期为3年"],
                "forbidden_actions": ["approve_work_permit"],
                "acceptable_actions": [{"tool": "submit_inspection_report"}],
            },
        },
        "ATOM-002": {
            "atom_id": "ATOM-002",
            "type": "routine_report",
            "expected_tool": "no_action",
            "params": {"reason": {"value": "无需", "match": "contains"}},
            "evidence": {
                "required_facts": ["日常报告"],
                "forbidden_actions": [],
                "acceptable_actions": [{"tool": "no_action"}],
            },
        },
    }


def test_assemble_action_event():
    from longhorizon_bench.pipeline.assembler import assemble_action_event

    atom = _make_atoms()["ATOM-001"]
    chain_event = {"id": "E01", "atom_ref": "ATOM-001", "depends_on": [], "is_critical": True, "dimensions": ["domain_knowledge"]}
    event_input = "收到巡检报告，发现压力容器接近检验周期"

    action_event = assemble_action_event(chain_event, atom, event_input)
    assert action_event["event_id"] == "E01"
    assert action_event["node_type"] == "action"
    assert action_event["scoring_rule"]["tool"]["expected"] == "submit_inspection_report"
    assert "target" in action_event["scoring_rule"]["params"]


def test_assemble_checkpoint_event():
    from longhorizon_bench.pipeline.assembler import assemble_checkpoint_event

    queries = [
        {"query": "压力容器情况如何？", "expected_keywords": ["压力容器"], "dimension": "long_term_memory", "match": "keyword_coverage"},
    ]
    cp_event = assemble_checkpoint_event("CP01", queries)
    assert cp_event["event_id"] == "CP01"
    assert cp_event["node_type"] == "checkpoint"
    assert cp_event["is_checkpoint"] is True
    assert len(cp_event["checkpoint_queries"]) == 1


def test_assemble_scenario():
    from longhorizon_bench.pipeline.assembler import assemble_scenario
    from longhorizon_bench.schema import Scenario

    chain = _make_chain()
    atoms = _make_atoms()
    event_inputs = {"E01": "巡检报告内容", "E02": "月报编制"}
    checkpoint_queries = {"CP01": [{"query": "test", "expected_keywords": ["test"], "dimension": "long_term_memory", "match": "keyword_coverage"}]}
    bg_text = "背景文档" * 100

    scenario_dict = assemble_scenario(chain, atoms, event_inputs, checkpoint_queries, bg_text)

    # Validate against Pydantic
    scenario = Scenario.model_validate(scenario_dict)
    assert scenario.scenario_id == "IND-TEST"
    assert len(scenario.action_events) == 2
    assert len(scenario.checkpoint_events) == 1


def test_save_scenario(tmp_path):
    from longhorizon_bench.pipeline.assembler import save_scenario

    scenario_dict = {"scenario_id": "IND-TEST", "domain": "industrial"}
    save_scenario(scenario_dict, tmp_path)
    path = tmp_path / "scenarios" / "industrial" / "IND-TEST.json"
    assert path.exists()
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["scenario_id"] == "IND-TEST"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assembler.py -v`
Expected: FAIL

- [ ] **Step 3: Implement assembler.py**

```python
# longhorizon_bench/pipeline/assembler.py
"""Stage 3c: Assemble chain + atoms + filled content into Scenario JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from longhorizon_bench.schema import Scenario


def assemble_action_event(
    chain_event: dict[str, Any],
    atom: dict[str, Any],
    event_input: str,
) -> dict[str, Any]:
    params_dict: dict[str, Any] = {}
    for name, spec in atom.get("params", {}).items():
        param_rule: dict[str, Any] = {"match": spec.get("match", "exact")}
        if spec.get("keywords"):
            param_rule["required_keywords"] = spec["keywords"]
        if spec.get("value") is not None:
            param_rule["expected"] = spec["value"]
        params_dict[name] = param_rule

    evidence = atom.get("evidence", {})
    if "acceptable_actions" not in evidence:
        evidence["acceptable_actions"] = [{"tool": atom.get("expected_tool", "")}]

    return {
        "event_id": chain_event["id"],
        "type": atom.get("type", "unknown"),
        "input": event_input,
        "depends_on": chain_event.get("depends_on", []),
        "node_type": "action",
        "is_checkpoint": False,
        "is_critical": chain_event.get("is_critical", False),
        "dimensions": chain_event.get("dimensions", []),
        "scoring_rule": {
            "tool": {"expected": atom.get("expected_tool", ""), "match": "exact"},
            "params": params_dict,
        },
        "evidence": evidence,
    }


def assemble_checkpoint_event(
    checkpoint_id: str,
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "event_id": checkpoint_id,
        "type": "checkpoint",
        "input": None,
        "node_type": "checkpoint",
        "is_checkpoint": True,
        "checkpoint_queries": queries,
    }


def assemble_scenario(
    chain: dict[str, Any],
    atoms: dict[str, dict[str, Any]],
    event_inputs: dict[str, str],
    checkpoint_queries: dict[str, list[dict[str, Any]]],
    background_text: str,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    cp_set = {cp["id"]: cp for cp in chain.get("checkpoints", [])}
    cp_after_map: dict[str, str] = {cp.get("after", ""): cp["id"] for cp in chain.get("checkpoints", [])}

    for chain_event in chain.get("events", []):
        atom_ref = chain_event.get("atom_ref", "")
        atom = atoms.get(atom_ref, {})
        event_input = event_inputs.get(chain_event["id"], "")
        events.append(assemble_action_event(chain_event, atom, event_input))

        if chain_event["id"] in cp_after_map:
            cp_id = cp_after_map[chain_event["id"]]
            queries = checkpoint_queries.get(cp_id, [])
            events.append(assemble_checkpoint_event(cp_id, queries))

    action_count = sum(1 for e in events if e["node_type"] == "action")
    cp_count = sum(1 for e in events if e["node_type"] == "checkpoint")

    return {
        "scenario_id": chain["scenario_id"],
        "domain": chain.get("domain", "industrial"),
        "role": chain.get("role", ""),
        "difficulty": chain.get("difficulty", 3),
        "background_docs": ["background.txt"],
        "background_tokens": len(background_text),
        "total_events": action_count,
        "total_checkpoints": cp_count,
        "annotator": "pipeline-v1",
        "generation_model": "claude-sonnet-4-20250514+deepseek-chat",
        "metadata": {"subdomain": chain.get("subdomain", ""), "pipeline_version": "1.0"},
        "events": events,
    }


def save_scenario(scenario_dict: dict[str, Any], data_dir: Path) -> Path:
    domain = scenario_dict.get("domain", "industrial")
    out_dir = data_dir / "scenarios" / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scenario_dict['scenario_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenario_dict, f, ensure_ascii=False, indent=2)
    return path


def run_generate(data_dir: str) -> None:
    from longhorizon_bench.pipeline.bg_generator import (
        extract_regulation_core, generate_background, save_background_docs,
    )
    from longhorizon_bench.pipeline.event_filler import fill_action_event, fill_checkpoint_queries
    from longhorizon_bench.pipeline.llm_client import LLMClient
    import os

    base = Path(data_dir)
    claude = LLMClient(provider="claude", api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    deepseek = LLMClient(provider="deepseek", api_key=os.environ.get("DEEPSEEK_API_KEY", ""))

    # Load regulations for background generation
    reg_dir = base / "raw_corpus" / "regulations"
    regs: list[dict] = []
    if reg_dir.exists():
        for f in sorted(reg_dir.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                regs.append(json.load(fh))

    # Load atoms
    atoms_dir = base / "skeletons" / "atoms"
    atoms: dict[str, dict] = {}
    if atoms_dir.exists():
        for f in sorted(atoms_dir.glob("*.yaml")):
            with open(f, encoding="utf-8") as fh:
                atom = yaml.safe_load(fh)
                atoms[atom["atom_id"]] = atom

    # Process each chain
    chains_dir = base / "skeletons" / "chains"
    if not chains_dir.exists():
        return

    for chain_file in sorted(chains_dir.glob("*.yaml")):
        with open(chain_file, encoding="utf-8") as fh:
            chain = yaml.safe_load(fh)

        # Stage 3a: Background document
        reg_core = extract_regulation_core(regs)
        bg_text = generate_background(claude, chain, reg_core)
        save_background_docs(chain["scenario_id"], bg_text, base)

        # Stage 3b: Fill events
        event_inputs: dict[str, str] = {}
        prior_inputs: list[dict] = []
        for chain_event in chain.get("events", []):
            atom_ref = chain_event.get("atom_ref", "")
            atom = atoms.get(atom_ref, {})
            event_input = fill_action_event(deepseek, atom, bg_text, prior_inputs)
            event_inputs[chain_event["id"]] = event_input
            prior_inputs.append({"event_id": chain_event["id"], "input": event_input})

        # Fill checkpoints
        checkpoint_queries: dict[str, list[dict]] = {}
        for cp in chain.get("checkpoints", []):
            cp_id = cp["id"]
            after_event = cp.get("after", "")
            # Get events before this checkpoint
            prior = []
            for ce in chain.get("events", []):
                prior.append({"event_id": ce["id"], "input": event_inputs.get(ce["id"], "")})
                if ce["id"] == after_event:
                    break
            dims = cp.get("queries_target_dimensions", ["long_term_memory"])
            queries = fill_checkpoint_queries(deepseek, prior, dims)
            checkpoint_queries[cp_id] = queries

        # Stage 3c: Assemble
        scenario_dict = assemble_scenario(chain, atoms, event_inputs, checkpoint_queries, bg_text)
        Scenario.model_validate(scenario_dict)
        save_scenario(scenario_dict, base)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_assembler.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/pipeline/assembler.py tests/test_assembler.py
git commit -m "feat: add scenario assembler with schema validation and Stage 3 orchestration"
```

---

### Task 11: Stage 4 Layer 1 — Structural Validator

**Files:**
- Create: `longhorizon_bench/pipeline/structural_validator.py`
- Create: `tests/test_structural_validator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_structural_validator.py
import json
import pytest
from pathlib import Path


def _make_valid_scenario() -> dict:
    return {
        "scenario_id": "TEST-001",
        "domain": "industrial",
        "role": "测试",
        "difficulty": 1,
        "background_docs": ["background.txt"],
        "background_tokens": 20000,
        "total_events": 2,
        "total_checkpoints": 1,
        "annotator": "test",
        "metadata": {},
        "events": [
            {
                "event_id": "E01", "type": "test", "input": "压力容器检验周期为3年的设备需要安排检查",
                "depends_on": [], "node_type": "action", "is_checkpoint": False, "is_critical": True,
                "dimensions": ["domain_knowledge"],
                "scoring_rule": {"tool": {"expected": "submit_inspection_report", "match": "exact"}, "params": {"target": {"expected": "车间", "match": "contains"}}},
                "evidence": {"required_facts": ["压力容器检验周期为3年"], "forbidden_actions": [], "acceptable_actions": [{"tool": "submit_inspection_report"}]},
            },
            {
                "event_id": "E02", "type": "test", "input": "无需操作的报告",
                "depends_on": ["E01"], "node_type": "action", "is_checkpoint": False, "is_critical": False,
                "dimensions": ["multi_step_reasoning"],
                "scoring_rule": {"tool": {"expected": "no_action", "match": "exact"}, "params": {"reason": {"expected": "无需", "match": "contains"}}},
                "evidence": {"required_facts": ["日常报告"], "forbidden_actions": [], "acceptable_actions": [{"tool": "no_action"}]},
            },
            {
                "event_id": "CP01", "type": "checkpoint", "node_type": "checkpoint", "is_checkpoint": True,
                "checkpoint_queries": [{"query": "test", "expected_keywords": ["压力容器"], "dimension": "long_term_memory", "match": "keyword_coverage"}],
            },
        ],
    }


def test_check_evidence_traceability_pass():
    from longhorizon_bench.pipeline.structural_validator import check_evidence_traceability

    scenario = _make_valid_scenario()
    bg_text = "压力容器检验周期为3年，到期前应安排检查。日常报告管理制度。"
    errors = check_evidence_traceability(scenario, bg_text)
    assert errors == []


def test_check_evidence_traceability_fail():
    from longhorizon_bench.pipeline.structural_validator import check_evidence_traceability

    scenario = _make_valid_scenario()
    bg_text = "完全无关的内容"
    errors = check_evidence_traceability(scenario, bg_text)
    assert len(errors) > 0


def test_check_background_length_pass():
    from longhorizon_bench.pipeline.structural_validator import check_background_length

    errors = check_background_length("内容" * 8000)
    assert errors == []


def test_check_background_length_fail():
    from longhorizon_bench.pipeline.structural_validator import check_background_length

    errors = check_background_length("短")
    assert len(errors) > 0


def test_run_structural_checks():
    from longhorizon_bench.pipeline.structural_validator import run_structural_checks

    scenario = _make_valid_scenario()
    bg_text = "压力容器检验周期为3年。日常报告管理流程。" * 500
    result = run_structural_checks(scenario, bg_text)
    assert "passed" in result
    assert "failed" in result
    assert isinstance(result["passed"], int)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_structural_validator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement structural_validator.py**

```python
# longhorizon_bench/pipeline/structural_validator.py
"""Stage 4 Layer 1: Structural validation of generated scenarios."""

from __future__ import annotations

from typing import Any

from longhorizon_bench.schema import Scenario


def check_background_length(bg_text: str, min_chars: int = 15000) -> list[str]:
    if len(bg_text) < min_chars:
        return [f"Background too short: {len(bg_text)} chars < {min_chars}"]
    return []


def check_evidence_traceability(
    scenario_dict: dict[str, Any], bg_text: str
) -> list[str]:
    errors: list[str] = []
    all_inputs = bg_text
    for event in scenario_dict.get("events", []):
        if event.get("node_type") == "action":
            all_inputs += " " + event.get("input", "")

    for event in scenario_dict.get("events", []):
        if event.get("node_type") != "action":
            continue
        evidence = event.get("evidence", {})
        for fact in evidence.get("required_facts", []):
            keywords = [w for w in fact.split() if len(w) >= 2]
            if keywords:
                hit = any(kw in all_inputs for kw in keywords[:3])
                if not hit:
                    errors.append(f"Event {event['event_id']}: required_fact not traceable: '{fact[:50]}'")
    return errors


def check_checkpoint_intervals(scenario_dict: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    events = scenario_dict.get("events", [])
    action_indices: dict[str, int] = {}
    cp_indices: list[int] = []
    action_count = 0
    for i, e in enumerate(events):
        if e.get("node_type") == "action":
            action_indices[e["event_id"]] = action_count
            action_count += 1
        elif e.get("node_type") == "checkpoint":
            cp_indices.append(action_count)

    prev = 0
    for cp_pos in cp_indices:
        interval = cp_pos - prev
        if interval < 3 or interval > 7:
            errors.append(f"Checkpoint interval {interval} not in [3,7]")
        prev = cp_pos
    return errors


def run_structural_checks(
    scenario_dict: dict[str, Any], bg_text: str
) -> dict[str, Any]:
    all_errors: list[str] = []

    try:
        Scenario.model_validate(scenario_dict)
    except Exception as e:
        all_errors.append(f"Schema validation failed: {e}")

    all_errors.extend(check_background_length(bg_text))
    all_errors.extend(check_evidence_traceability(scenario_dict, bg_text))
    all_errors.extend(check_checkpoint_intervals(scenario_dict))

    total_checks = 4
    failed = len(all_errors)
    return {
        "passed": total_checks - min(failed, total_checks),
        "failed": failed,
        "errors": all_errors,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_structural_validator.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/pipeline/structural_validator.py tests/test_structural_validator.py
git commit -m "feat: add structural validator with evidence traceability and checkpoint interval checks"
```

---

### Task 12: Stage 4 Layer 2 — LLM Committee Reviewer

**Files:**
- Create: `longhorizon_bench/pipeline/committee_reviewer.py`
- Create: `tests/test_committee_reviewer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_committee_reviewer.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_committee_reviewer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement committee_reviewer.py**

```python
# longhorizon_bench/pipeline/committee_reviewer.py
"""Stage 4 Layer 2: LLM committee review with dual-model voting."""

from __future__ import annotations

import json
from typing import Any

REVIEW_DIMENSIONS = ["因果连贯性", "证据可追溯", "难度梯度", "答案区分度", "专业准确性"]


def build_review_prompt(scenario_dict: dict[str, Any]) -> tuple[str, str]:
    system = """你是一名benchmark质量评审专家。评审一个工业安全事件链场景的质量。

对以下5个维度各打1-5分：
1. 因果连贯性：事件间 depends_on 关系是否合理
2. 证据可追溯：required_facts 是否真的能从背景文档找到
3. 难度梯度：事件链是否从简单到复杂递进
4. 答案区分度：scoring_rule 是否能区分好坏agent
5. 专业准确性：法规引用、行业术语是否正确

用 ```json ``` 输出，格式如 {"因果连贯性": 4, ...}"""

    events_summary = "\n".join(
        f"- {e.get('event_id', '')}: {e.get('type', '')} - {str(e.get('input', ''))[:100]}"
        for e in scenario_dict.get("events", [])[:20]
    )

    user = f"""场景ID: {scenario_dict.get('scenario_id', '')}
角色: {scenario_dict.get('role', '')}
事件数: {scenario_dict.get('total_events', 0)} actions + {scenario_dict.get('total_checkpoints', 0)} checkpoints

事件链:
{events_summary}

请评审并打分。"""

    return system, user


def parse_review_scores(response: str) -> dict[str, int]:
    start = response.find("```json")
    end = response.find("```", start + 7) if start >= 0 else -1
    if start >= 0 and end >= 0:
        json_text = response[start + 7:end].strip()
    else:
        json_text = response.strip()
    return json.loads(json_text)


def check_committee_agreement(
    scores_a: dict[str, int], scores_b: dict[str, int], max_diff: int = 1
) -> tuple[list[str], list[str]]:
    agreed: list[str] = []
    disagreed: list[str] = []
    for dim in REVIEW_DIMENSIONS:
        sa = scores_a.get(dim, 0)
        sb = scores_b.get(dim, 0)
        if abs(sa - sb) <= max_diff:
            agreed.append(dim)
        else:
            disagreed.append(dim)
    return agreed, disagreed


def compute_verdict(
    scores_a: dict[str, int],
    scores_b: dict[str, int],
    min_avg: float = 3.5,
) -> str:
    _, disagreed = check_committee_agreement(scores_a, scores_b)
    if disagreed:
        return "NEEDS_HUMAN_REVIEW"

    for dim in REVIEW_DIMENSIONS:
        avg = (scores_a.get(dim, 0) + scores_b.get(dim, 0)) / 2
        if avg < min_avg:
            return "FAIL"

    return "PASS"


def review_scenario(
    claude_client: Any,
    deepseek_client: Any,
    scenario_dict: dict[str, Any],
) -> dict[str, Any]:
    system, user = build_review_prompt(scenario_dict)
    msgs = [{"role": "user", "content": user}]

    response_a = claude_client.chat(system=system, messages=msgs)
    response_b = deepseek_client.chat(system=system, messages=msgs)

    scores_a = parse_review_scores(response_a)
    scores_b = parse_review_scores(response_b)
    verdict = compute_verdict(scores_a, scores_b)

    return {
        "claude": scores_a,
        "deepseek": scores_b,
        "verdict": verdict,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_committee_reviewer.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/pipeline/committee_reviewer.py tests/test_committee_reviewer.py
git commit -m "feat: add LLM committee reviewer with dual-model voting"
```

---

### Task 13: Stage 4 Layer 3 — Simulation Validator

**Files:**
- Create: `longhorizon_bench/pipeline/simulation_validator.py`
- Create: `tests/test_simulation_validator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_simulation_validator.py
import json
import pytest
from pathlib import Path
from longhorizon_bench.schema import AgentAction, CheckpointResponse, ToolCall
from longhorizon_bench.runners.base import BaseRunner


def test_perfect_agent_from_scenario():
    from longhorizon_bench.pipeline.simulation_validator import build_perfect_agent

    scenario_dict = {
        "events": [
            {
                "event_id": "E01", "node_type": "action",
                "scoring_rule": {
                    "tool": {"expected": "no_action", "match": "exact"},
                    "params": {"reason": {"expected": "test", "match": "contains"}},
                },
            },
            {
                "event_id": "CP01", "node_type": "checkpoint",
                "checkpoint_queries": [
                    {"query": "test?", "expected_keywords": ["answer"], "dimension": "long_term_memory", "match": "keyword_coverage"},
                ],
            },
        ]
    }
    agent = build_perfect_agent(scenario_dict)
    obs_action = {"current_event": {"event_id": "E01", "node_type": "action", "input": "test"}}
    result = agent.act(obs_action)
    assert isinstance(result, AgentAction)
    assert result.tool_calls[0].tool == "no_action"

    obs_cp = {"current_event": {"event_id": "CP01", "node_type": "checkpoint", "queries": ["test?"]}}
    result_cp = agent.act(obs_cp)
    assert isinstance(result_cp, CheckpointResponse)
    assert "answer" in result_cp.answers["test?"]


def test_bad_agent():
    from longhorizon_bench.pipeline.simulation_validator import BadSimAgent

    agent = BadSimAgent()
    obs = {"current_event": {"event_id": "E01", "node_type": "action", "input": "test"}}
    result = agent.act(obs)
    assert isinstance(result, AgentAction)
    assert result.tool_calls[0].tool == "approve_work_permit"


def test_run_simulation():
    from longhorizon_bench.pipeline.simulation_validator import run_simulation

    scenario_dict = {
        "scenario_id": "TEST",
        "domain": "industrial",
        "role": "测试",
        "difficulty": 1,
        "background_docs": ["bg.txt"],
        "background_tokens": 100,
        "total_events": 1,
        "total_checkpoints": 1,
        "annotator": "test",
        "metadata": {},
        "events": [
            {
                "event_id": "E01", "type": "test", "input": "测试",
                "depends_on": [], "node_type": "action", "is_checkpoint": False,
                "is_critical": False, "dimensions": ["domain_knowledge"],
                "scoring_rule": {"tool": {"expected": "no_action", "match": "exact"}, "params": {"reason": {"expected": "ok", "match": "contains"}}},
                "evidence": {"required_facts": ["test"], "forbidden_actions": [], "acceptable_actions": []},
            },
            {
                "event_id": "CP01", "type": "checkpoint", "node_type": "checkpoint",
                "is_checkpoint": True,
                "checkpoint_queries": [{"query": "测试?", "expected_keywords": ["测试"], "dimension": "long_term_memory", "match": "keyword_coverage"}],
            },
        ],
    }
    result = run_simulation(scenario_dict, tmp_data_dir=None)
    assert "perfect_agent_score" in result
    assert "bad_agent_score" in result
    assert "delta" in result
    assert result["perfect_agent_score"] > result["bad_agent_score"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_simulation_validator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement simulation_validator.py**

```python
# longhorizon_bench/pipeline/simulation_validator.py
"""Stage 4 Layer 3: PerfectAgent/BadAgent simulation validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from longhorizon_bench.schema import AgentAction, CheckpointResponse, ToolCall
from longhorizon_bench.runners.base import BaseRunner
from longhorizon_bench.env import LongHorizonEnv


class PerfectSimAgent(BaseRunner):
    def __init__(self, scenario_dict: dict[str, Any]) -> None:
        self._answers: dict[str, dict[str, Any]] = {}
        for event in scenario_dict.get("events", []):
            self._answers[event.get("event_id", "")] = event

    def act(self, observation: dict[str, Any]) -> AgentAction | CheckpointResponse:
        event = observation["current_event"]
        event_id = event["event_id"]
        event_data = self._answers.get(event_id, {})

        if event["node_type"] == "checkpoint":
            queries = event.get("queries", [])
            answers: dict[str, str] = {}
            for q in queries:
                cp_queries = event_data.get("checkpoint_queries", [])
                matching = [cq for cq in cp_queries if cq["query"] == q]
                if matching:
                    answers[q] = " ".join(matching[0].get("expected_keywords", []))
                else:
                    answers[q] = q
            return CheckpointResponse(answers=answers)

        scoring_rule = event_data.get("scoring_rule", {})
        tool_name = scoring_rule.get("tool", {}).get("expected", "no_action")
        kwargs: dict[str, Any] = {}
        for param_name, param_spec in scoring_rule.get("params", {}).items():
            expected = param_spec.get("expected")
            keywords = param_spec.get("required_keywords")
            if expected is not None:
                kwargs[param_name] = expected if not isinstance(expected, list) else expected[0]
            elif keywords:
                kwargs[param_name] = " ".join(keywords)
            else:
                kwargs[param_name] = "default"

        return AgentAction(tool_calls=[ToolCall(tool=tool_name, kwargs=kwargs)])


class BadSimAgent(BaseRunner):
    def act(self, observation: dict[str, Any]) -> AgentAction | CheckpointResponse:
        event = observation["current_event"]
        if event["node_type"] == "checkpoint":
            queries = event.get("queries", [])
            return CheckpointResponse(answers={q: "不知道" for q in queries})
        return AgentAction(tool_calls=[
            ToolCall(tool="approve_work_permit", kwargs={
                "permit_type": "hot_work", "conditions": [], "approved": True,
            }),
        ])


def build_perfect_agent(scenario_dict: dict[str, Any]) -> PerfectSimAgent:
    return PerfectSimAgent(scenario_dict)


def run_simulation(
    scenario_dict: dict[str, Any],
    tmp_data_dir: Path | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmp_data_dir) if tmp_data_dir else Path(tmpdir)
        domain = scenario_dict.get("domain", "industrial")
        sid = scenario_dict["scenario_id"]

        scenario_dir = base / "scenarios" / domain
        scenario_dir.mkdir(parents=True, exist_ok=True)
        with open(scenario_dir / f"{sid}.json", "w", encoding="utf-8") as f:
            json.dump(scenario_dict, f, ensure_ascii=False)

        bg_dir = base / "background_docs" / sid
        bg_dir.mkdir(parents=True, exist_ok=True)
        for doc_name in scenario_dict.get("background_docs", []):
            (bg_dir / doc_name).write_text("背景文档占位符", encoding="utf-8")

        perfect = build_perfect_agent(scenario_dict)
        env = LongHorizonEnv(f"{domain}/{sid}", mode="full_context", data_dir=base)
        perfect_results = perfect.run(env)

        bad = BadSimAgent()
        env2 = LongHorizonEnv(f"{domain}/{sid}", mode="full_context", data_dir=base)
        bad_results = bad.run(env2)

    return {
        "perfect_agent_score": perfect_results["chain_score"],
        "bad_agent_score": bad_results["chain_score"],
        "delta": perfect_results["chain_score"] - bad_results["chain_score"],
        "perfect_pass": perfect_results["chain_pass"],
    }


def run_validate(data_dir: str) -> None:
    from longhorizon_bench.pipeline.structural_validator import run_structural_checks
    from longhorizon_bench.pipeline.committee_reviewer import review_scenario
    from longhorizon_bench.pipeline.llm_client import LLMClient
    import os

    base = Path(data_dir)
    scenarios_dir = base / "scenarios"
    validated_dir = base / "validated"
    reports_dir = base / "review_reports"
    validated_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for scenario_file in sorted(scenarios_dir.rglob("*.json")):
        with open(scenario_file, encoding="utf-8") as f:
            scenario_dict = json.load(f)

        sid = scenario_dict["scenario_id"]
        domain = scenario_dict.get("domain", "industrial")

        bg_path = base / "background_docs" / sid / "background.txt"
        bg_text = bg_path.read_text(encoding="utf-8") if bg_path.exists() else ""

        structural = run_structural_checks(scenario_dict, bg_text)
        sim_result = run_simulation(scenario_dict)

        report: dict[str, Any] = {
            "scenario_id": sid,
            "structural_checks": structural,
            "simulation": sim_result,
        }

        api_key_claude = os.environ.get("ANTHROPIC_API_KEY", "")
        api_key_ds = os.environ.get("DEEPSEEK_API_KEY", "")
        if api_key_claude and api_key_ds:
            claude = LLMClient(provider="claude", api_key=api_key_claude)
            deepseek = LLMClient(provider="deepseek", api_key=api_key_ds)
            committee = review_scenario(claude, deepseek, scenario_dict)
            report["committee_scores"] = committee
            verdict = committee["verdict"]
        else:
            verdict = "PASS" if structural["failed"] == 0 and sim_result["delta"] > 0.6 else "FAIL"

        report["verdict"] = verdict

        with open(reports_dir / f"{sid}_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        if verdict == "PASS":
            out_dir = validated_dir / domain
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{sid}.json", "w", encoding="utf-8") as f:
                json.dump(scenario_dict, f, ensure_ascii=False, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_simulation_validator.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS (~100+ tests total)

- [ ] **Step 6: Commit**

```bash
git add longhorizon_bench/pipeline/simulation_validator.py tests/test_simulation_validator.py
git commit -m "feat: add simulation validator with PerfectAgent/BadAgent and Stage 4 orchestration"
```

---

### Task 14: Install Dependencies and End-to-End Smoke Test

**Files:**
- Create: `tests/test_pipeline_e2e.py`

- [ ] **Step 1: Install pipeline dependencies**

Run: `pip install -e ".[pipeline,dev]"`

- [ ] **Step 2: Write end-to-end smoke test (offline, no API calls)**

```python
# tests/test_pipeline_e2e.py
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
```

- [ ] **Step 3: Run end-to-end tests**

Run: `pytest tests/test_pipeline_e2e.py -v`
Expected: All 3 tests PASS

- [ ] **Step 4: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_e2e.py
git commit -m "feat: add end-to-end pipeline smoke tests"
```
