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
