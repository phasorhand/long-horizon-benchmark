# longhorizon_bench/env.py
"""LongHorizonEnv: Gym-style environment for event chain evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from longhorizon_bench.loader import load_scenario, load_background_docs
from longhorizon_bench.schema import (
    ActionEvent,
    AgentAction,
    CheckpointEvent,
    CheckpointResponse,
    Scenario,
)
from longhorizon_bench.evaluation.scorer import EventScorer, ActionScore, CheckpointScore
from longhorizon_bench.evaluation.consistency import ConsistencyChecker
from longhorizon_bench.evaluation.metrics import compute_metrics

Mode = Literal["full_context", "rolling_window", "memory_only"]


class LongHorizonEnv:
    def __init__(
        self,
        scenario_path: str,
        mode: Mode = "rolling_window",
        window_size: int = 5,
        data_dir: Path | None = None,
    ) -> None:
        self._scenario_path = scenario_path
        self._mode = mode
        self._window_size = window_size
        self._data_dir = data_dir
        self._scenario: Scenario | None = None
        self._bg_docs: dict[str, str] = {}
        self._event_index = 0
        self._history: list[dict[str, Any]] = []
        self._action_history: list[tuple[str, list[Any]]] = []
        self._event_scores: list[dict] = []
        self._checkpoint_scores: list[dict] = []
        self._scorer = EventScorer()
        self._consistency_checker = ConsistencyChecker()

    def reset(self) -> dict[str, Any]:
        self._scenario = load_scenario(self._scenario_path, data_dir=self._data_dir)
        self._bg_docs = load_background_docs(
            self._scenario.scenario_id,
            self._scenario.background_docs,
            data_dir=self._data_dir,
        )
        self._event_index = 0
        self._history = []
        self._action_history = []
        self._event_scores = []
        self._checkpoint_scores = []
        return self._current_observation()

    def step(
        self, action: AgentAction | CheckpointResponse
    ) -> tuple[dict[str, Any], float, bool, dict]:
        event = self._scenario.events[self._event_index]
        reward = 0.0

        if isinstance(event, ActionEvent) and isinstance(action, AgentAction):
            score = self._scorer.score_action(event, action)
            reward = score.total
            self._event_scores.append({
                "event_id": event.event_id,
                "is_critical": event.is_critical,
                "score": score,
            })
            self._action_history.append(
                (event.event_id, action.tool_calls)
            )
            self._history.append({
                "event_id": event.event_id,
                "type": event.type,
                "input": event.input,
                "action": [c.model_dump() for c in action.tool_calls],
            })

        elif isinstance(event, CheckpointEvent) and isinstance(action, CheckpointResponse):
            cp_score = self._scorer.score_checkpoint(event, action)
            for i, q in enumerate(event.checkpoint_queries):
                self._checkpoint_scores.append({
                    "dimension": q.dimension,
                    "score": cp_score.query_scores[i] if i < len(cp_score.query_scores) else 0.0,
                })
            reward = cp_score.total
            self._history.append({
                "event_id": event.event_id,
                "type": "checkpoint",
                "input": None,
            })

        self._event_index += 1
        done = self._event_index >= len(self._scenario.events)
        obs = self._current_observation() if not done else {}
        return obs, reward, done, {}

    def evaluate(self) -> dict[str, Any]:
        violations = self._consistency_checker.check_all(self._action_history)
        result = compute_metrics(
            event_scores=self._event_scores,
            checkpoint_scores=self._checkpoint_scores,
            violations=violations,
            applicable_rule_checks=max(len(self._action_history), 1),
        )
        return {
            "chain_score": result.chain_score,
            "chain_pass": result.chain_pass,
            "event_scores": [
                {"event_id": e["event_id"], "total": e["score"].total}
                for e in self._event_scores
            ],
            "dimension_scores": result.dimension_scores,
            "consistency_violations": [
                {"rule_id": v.rule_id, "event_id": v.event_id, "description": v.description}
                for v in result.violations
            ],
            "checkpoint_details": self._checkpoint_scores,
        }

    def _current_observation(self) -> dict[str, Any]:
        if self._scenario is None:
            raise RuntimeError("Call reset() before step()")

        event = self._scenario.events[self._event_index]
        obs: dict[str, Any] = {
            "role": self._scenario.role,
            "current_event": {
                "event_id": event.event_id,
                "type": event.type,
                "node_type": event.node_type,
            },
        }

        if isinstance(event, ActionEvent):
            obs["current_event"]["input"] = event.input
        elif isinstance(event, CheckpointEvent):
            obs["current_event"]["queries"] = [
                q.query for q in event.checkpoint_queries
            ]

        if self._mode == "full_context":
            obs["background_docs"] = self._bg_docs
            obs["history"] = self._history
        elif self._mode == "rolling_window":
            obs["background_docs"] = self._bg_docs
            obs["history"] = self._build_rolling_history()
        elif self._mode == "memory_only":
            full_text = "\n".join(self._bg_docs.values())
            obs["background_summary"] = full_text[:500] + "..." if len(full_text) > 500 else full_text
            obs["history"] = [
                {"event_id": h["event_id"], "type": h["type"]}
                for h in self._history
            ]

        return obs

    def _build_rolling_history(self) -> list[dict]:
        result: list[dict] = []
        for i, h in enumerate(self._history):
            if i >= len(self._history) - self._window_size:
                result.append(h)
            else:
                result.append({
                    "event_id": h["event_id"],
                    "type": h["type"],
                    "summary_only": True,
                })
        return result
