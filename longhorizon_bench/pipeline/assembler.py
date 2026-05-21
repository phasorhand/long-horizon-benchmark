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

    reg_dir = base / "raw_corpus" / "regulations"
    regs: list[dict] = []
    if reg_dir.exists():
        for f in sorted(reg_dir.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                regs.append(json.load(fh))

    atoms_dir = base / "skeletons" / "atoms"
    atoms: dict[str, dict] = {}
    if atoms_dir.exists():
        for f in sorted(atoms_dir.glob("*.yaml")):
            with open(f, encoding="utf-8") as fh:
                atom = yaml.safe_load(fh)
                atoms[atom["atom_id"]] = atom

    chains_dir = base / "skeletons" / "chains"
    if not chains_dir.exists():
        return

    for chain_file in sorted(chains_dir.glob("*.yaml")):
        with open(chain_file, encoding="utf-8") as fh:
            chain = yaml.safe_load(fh)

        reg_core = extract_regulation_core(regs)
        bg_text = generate_background(claude, chain, reg_core)
        save_background_docs(chain["scenario_id"], bg_text, base)

        event_inputs: dict[str, str] = {}
        prior_inputs: list[dict] = []
        for chain_event in chain.get("events", []):
            atom_ref = chain_event.get("atom_ref", "")
            atom = atoms.get(atom_ref, {})
            event_input = fill_action_event(deepseek, atom, bg_text, prior_inputs)
            event_inputs[chain_event["id"]] = event_input
            prior_inputs.append({"event_id": chain_event["id"], "input": event_input})

        checkpoint_queries: dict[str, list[dict]] = {}
        for cp in chain.get("checkpoints", []):
            cp_id = cp["id"]
            after_event = cp.get("after", "")
            prior = []
            for ce in chain.get("events", []):
                prior.append({"event_id": ce["id"], "input": event_inputs.get(ce["id"], "")})
                if ce["id"] == after_event:
                    break
            dims = cp.get("queries_target_dimensions", ["long_term_memory"])
            queries = fill_checkpoint_queries(deepseek, prior, dims)
            checkpoint_queries[cp_id] = queries

        scenario_dict = assemble_scenario(chain, atoms, event_inputs, checkpoint_queries, bg_text)
        Scenario.model_validate(scenario_dict)
        save_scenario(scenario_dict, base)
