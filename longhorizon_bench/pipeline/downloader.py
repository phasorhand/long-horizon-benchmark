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
            raw_docs = [
                {
                    "doc_id": f"{short_name}_{i:06d}",
                    "source": f"IndustryCorpus2_{subset}",
                    "text": row.get("text", ""),
                }
                for i, row in enumerate(ds)
            ]
        except Exception:
            raw_docs = []
        filtered = filter_by_keywords(raw_docs)
        filtered = filter_by_length(filtered)
        filtered = dedup_by_simhash(filtered)
        save_filtered_docs(filtered, output_path)

    reg_path = base / "raw_corpus" / "regulations"
    if not reg_path.exists() or not any(reg_path.glob("*.json")):
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "twang2218/chinese-law-and-regulations", split="train"
            )
            raw_regs = [
                {
                    "doc_id": f"reg_{i:06d}",
                    "source": "chinese-law-and-regulations",
                    "title": row.get("title", ""),
                    "office": row.get("office", ""),
                    "status": row.get("status", ""),
                    "text": row.get("content", ""),
                }
                for i, row in enumerate(ds)
            ]
        except Exception:
            raw_regs = []
        filtered_regs = filter_regulations(raw_regs)
        for reg in filtered_regs:
            reg["keywords_matched"] = [
                kw
                for kw in REGULATION_KEYWORDS
                if kw in reg.get("title", "") or kw in reg.get("office", "")
            ]
            reg["char_count"] = len(reg.get("text", ""))
        save_filtered_docs(filtered_regs, reg_path)
