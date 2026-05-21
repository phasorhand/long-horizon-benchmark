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
    cp_indices: list[int] = []
    action_count = 0
    for e in events:
        if e.get("node_type") == "action":
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
