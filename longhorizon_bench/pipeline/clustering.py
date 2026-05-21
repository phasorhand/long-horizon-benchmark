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
