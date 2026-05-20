# longhorizon_bench/loader.py
"""Load and validate scenario JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from longhorizon_bench.schema import Scenario

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def load_scenario(
    scenario_path: str, data_dir: Path | None = None
) -> Scenario:
    base = data_dir or _DEFAULT_DATA_DIR
    file_path = base / "scenarios" / f"{scenario_path}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Scenario not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        raw = json.load(f)

    return Scenario.model_validate(raw)


def load_background_docs(
    scenario_id: str,
    doc_filenames: list[str],
    data_dir: Path | None = None,
) -> dict[str, str]:
    base = data_dir or _DEFAULT_DATA_DIR
    docs_dir = base / "background_docs" / scenario_id
    result: dict[str, str] = {}
    for filename in doc_filenames:
        doc_path = docs_dir / filename
        if doc_path.exists():
            result[filename] = doc_path.read_text(encoding="utf-8")
    return result
