"""Stage 2d: Compose validated atoms into event chains via LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def validate_chain(chain: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    events = chain.get("events", [])
    event_ids = {e["id"] for e in events}

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

    for e in events:
        for dep in e.get("depends_on", []):
            if dep not in event_ids:
                errors.append(f"Event {e['id']} depends on unknown {dep}")

    checkpoints = chain.get("checkpoints", [])
    if checkpoints and events:
        id_to_index = {e["id"]: i for i, e in enumerate(events)}
        cp_positions = sorted(
            id_to_index.get(cp.get("after", ""), 0) for cp in checkpoints
        )
        if len(cp_positions) >= 2:
            for i in range(1, len(cp_positions)):
                interval = cp_positions[i] - cp_positions[i - 1]
                if interval < 3:
                    errors.append(f"Checkpoint interval too small ({interval} < 3)")

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
