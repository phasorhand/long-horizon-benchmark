# LongHorizon-Bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pip-installable Python benchmark framework that loads Chinese long-horizon scenarios, runs agents through event chains in three context modes, and produces multi-level evaluation scores.

**Architecture:** Pydantic v2 models define the data schema. A Gym-style `LongHorizonEnv` drives event replay in Full/Rolling/Memory modes. A rule-based scorer evaluates each event action against typed `scoring_rule` definitions. Six concrete consistency rules (C01–C06) detect cross-event contradictions. Runners adapt LLM APIs to the environment loop.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest, click (CLI)

---

## File Map

```
long-horizon-bench/
├── pyproject.toml                          # Package metadata + deps
├── DATA_LICENSE                            # CC-BY-SA-4.0 data license
├── data/
│   └── scenarios/
│       └── industrial/
│           └── DEMO-001.json              # Minimal demo scenario for tests
├── longhorizon_bench/
│   ├── __init__.py                        # Public API re-exports
│   ├── schema.py                          # Pydantic models: Scenario, Event, ScoringRule, etc.
│   ├── loader.py                          # load_scenario() + validation
│   ├── env.py                             # LongHorizonEnv: reset/step/evaluate
│   ├── tools/
│   │   ├── __init__.py                    # Re-exports COMMON_TOOLS
│   │   ├── registry.py                    # ToolRegistry: lookup, validate actions
│   │   ├── common.py                      # Common tools shared across domains
│   │   └── industrial.py                  # Industrial domain tools
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── scorer.py                      # EventScorer: score one action against one scoring_rule
│   │   ├── consistency.py                 # ConsistencyChecker: rules C01–C06
│   │   └── metrics.py                     # compute_metrics(): L1/L2/L3 aggregation
│   └── runners/
│       ├── __init__.py
│       ├── base.py                        # BaseRunner ABC
│       └── openai_runner.py               # OpenAI-compatible runner
├── tests/
│   ├── conftest.py                        # Shared fixtures (demo scenario, sample events)
│   ├── test_schema.py
│   ├── test_loader.py
│   ├── test_tools.py
│   ├── test_scorer.py
│   ├── test_consistency.py
│   ├── test_metrics.py
│   ├── test_env.py
│   └── test_runner.py
└── scripts/
    └── validate_scenario.py               # CLI: validate a scenario JSON file
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `DATA_LICENSE`
- Create: `longhorizon_bench/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

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

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create DATA_LICENSE**

```text
DATA LICENSE
============

The code in this repository is licensed under Apache 2.0 (see LICENSE).

The benchmark data (everything under data/) is licensed under
Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0).

Source Attribution
------------------

| Source | License | Usage |
|--------|---------|-------|
| BAAI/IndustryCorpus2 | Apache 2.0 | Filtered and rewritten, no original text |
| ShengbinYue/DISC-Law-SFT | Apache 2.0 | Event patterns extracted, content rewritten |
| twang2218/chinese-law-and-regulations | MIT | Regulations are public domain, cited directly |
| china-ai-law-challenge/cail2018 | CC-BY-4.0 | Structure referenced, attributed |
| fjcanyue/wikipedia-zh-cn | CC-BY-SA-3.0 | Rewritten for background material |

All scenario content is synthetic material derived from the above sources.
No restricted original data is redistributed.
```

- [ ] **Step 3: Create empty __init__.py**

```python
"""LongHorizon-Bench: Chinese long-horizon benchmark for LLM evaluation."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Create tests/conftest.py with empty file**

```python
"""Shared test fixtures for longhorizon-bench."""
```

- [ ] **Step 5: Install the package in editable mode and verify**

Run: `pip install -e ".[dev]"`
Expected: Successfully installed longhorizon-bench-0.1.0

Run: `python -c "import longhorizon_bench; print(longhorizon_bench.__version__)"`
Expected: `0.1.0`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml DATA_LICENSE longhorizon_bench/__init__.py tests/conftest.py
git commit -m "feat: project scaffolding with pyproject.toml and package structure"
```

---

### Task 2: Pydantic Schema Models

**Files:**
- Create: `longhorizon_bench/schema.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Write the failing tests for schema models**

```python
# tests/test_schema.py
import pytest
from datetime import date


def test_match_type_enum_values():
    from longhorizon_bench.schema import MatchType

    assert MatchType.EXACT == "exact"
    assert MatchType.CONTAINS == "contains"
    assert MatchType.ENUM == "enum"
    assert MatchType.KEYWORD_COVERAGE == "keyword_coverage"


def test_param_rule_exact_match():
    from longhorizon_bench.schema import ParamRule

    rule = ParamRule(expected="3号车间", match="exact")
    assert rule.expected == "3号车间"
    assert rule.match == "exact"


def test_param_rule_enum_match_accepts_list():
    from longhorizon_bench.schema import ParamRule

    rule = ParamRule(expected=["medium", "high"], match="enum")
    assert rule.expected == ["medium", "high"]


def test_param_rule_keyword_coverage():
    from longhorizon_bench.schema import ParamRule

    rule = ParamRule(required_keywords=["压力容器", "检验周期"], match="keyword_coverage")
    assert rule.required_keywords == ["压力容器", "检验周期"]


def test_scoring_rule_structure():
    from longhorizon_bench.schema import ScoringRule, ParamRule

    rule = ScoringRule(
        tool=ParamRule(expected="submit_inspection_report", match="exact"),
        params={
            "target": ParamRule(expected="3号车间", match="contains"),
            "risk_level": ParamRule(expected=["medium", "high"], match="enum"),
        },
    )
    assert rule.tool.expected == "submit_inspection_report"
    assert "target" in rule.params


def test_evidence_structure():
    from longhorizon_bench.schema import Evidence

    ev = Evidence(
        required_facts=["E01: 压力容器接近检验周期"],
        forbidden_actions=["approve_work_permit"],
        acceptable_actions=[{"tool": "submit_inspection_report"}],
    )
    assert len(ev.required_facts) == 1
    assert len(ev.forbidden_actions) == 1


def test_action_event_creation():
    from longhorizon_bench.schema import ActionEvent, ScoringRule, ParamRule, Evidence

    event = ActionEvent(
        event_id="E01",
        type="routine_inspection",
        input="收到设备科提交的3号车间季度巡检报告",
        depends_on=[],
        is_critical=True,
        dimensions=["domain_knowledge"],
        scoring_rule=ScoringRule(
            tool=ParamRule(expected="submit_inspection_report", match="exact"),
            params={},
        ),
        evidence=Evidence(
            required_facts=[],
            forbidden_actions=[],
            acceptable_actions=[],
        ),
    )
    assert event.node_type == "action"
    assert event.is_checkpoint is False
    assert event.is_critical is True


def test_checkpoint_event_creation():
    from longhorizon_bench.schema import CheckpointEvent, CheckpointQuery

    event = CheckpointEvent(
        event_id="E05",
        type="checkpoint",
        checkpoint_queries=[
            CheckpointQuery(
                query="E02中发现的隐患当前状态？",
                expected_keywords=["整改中", "未完成"],
                dimension="long_term_memory",
                match="keyword_coverage",
            ),
        ],
    )
    assert event.node_type == "checkpoint"
    assert event.is_checkpoint is True
    assert len(event.checkpoint_queries) == 1


def test_scenario_creation():
    from longhorizon_bench.schema import (
        Scenario, ActionEvent, CheckpointEvent, CheckpointQuery,
        ScoringRule, ParamRule, Evidence,
    )

    scenario = Scenario(
        scenario_id="DEMO-001",
        domain="industrial",
        role="测试角色",
        difficulty=1,
        background_docs=["doc_001.txt"],
        background_tokens=1000,
        total_events=1,
        total_checkpoints=1,
        annotator="test",
        annotator_agreement=0.85,
        generation_model="test-model",
        metadata={"source": "test"},
        events=[
            ActionEvent(
                event_id="E01",
                type="routine_inspection",
                input="测试事件",
                depends_on=[],
                is_critical=False,
                dimensions=["domain_knowledge"],
                scoring_rule=ScoringRule(
                    tool=ParamRule(expected="no_action", match="exact"),
                    params={},
                ),
                evidence=Evidence(
                    required_facts=[],
                    forbidden_actions=[],
                    acceptable_actions=[],
                ),
            ),
            CheckpointEvent(
                event_id="CP01",
                type="checkpoint",
                checkpoint_queries=[
                    CheckpointQuery(
                        query="测试问题",
                        expected_keywords=["测试"],
                        dimension="long_term_memory",
                        match="keyword_coverage",
                    ),
                ],
            ),
        ],
    )
    assert scenario.scenario_id == "DEMO-001"
    assert len(scenario.events) == 2
    assert scenario.action_events[0].event_id == "E01"
    assert scenario.checkpoint_events[0].event_id == "CP01"


def test_agent_action_single_tool():
    from longhorizon_bench.schema import AgentAction, ToolCall

    action = AgentAction(
        tool_calls=[
            ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"}),
        ]
    )
    assert len(action.tool_calls) == 1


def test_agent_action_multiple_tools():
    from longhorizon_bench.schema import AgentAction, ToolCall

    action = AgentAction(
        tool_calls=[
            ToolCall(tool="file_incident_report", kwargs={"severity": "major"}),
            ToolCall(tool="escalate_to_management", kwargs={"urgency": "emergency"}),
        ]
    )
    assert len(action.tool_calls) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'longhorizon_bench.schema'`

- [ ] **Step 3: Implement schema.py**

```python
# longhorizon_bench/schema.py
"""Pydantic models for LongHorizon-Bench data schema."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class MatchType(str, Enum):
    EXACT = "exact"
    CONTAINS = "contains"
    ENUM = "enum"
    KEYWORD_COVERAGE = "keyword_coverage"


class ParamRule(BaseModel):
    expected: Any = None
    required_keywords: list[str] | None = None
    match: MatchType


class ScoringRule(BaseModel):
    tool: ParamRule
    params: dict[str, ParamRule] = Field(default_factory=dict)


class Evidence(BaseModel):
    required_facts: list[str]
    forbidden_actions: list[str]
    acceptable_actions: list[dict[str, str]]


class CheckpointQuery(BaseModel):
    query: str
    expected_keywords: list[str] | None = None
    dimension: str
    match: str | None = None
    consistency_rule: str | None = None


class ActionEvent(BaseModel):
    event_id: str
    type: str
    input: str
    depends_on: list[str]
    node_type: Literal["action"] = "action"
    is_checkpoint: Literal[False] = False
    is_critical: bool = False
    dimensions: list[str]
    scoring_rule: ScoringRule
    evidence: Evidence


class CheckpointEvent(BaseModel):
    event_id: str
    type: str
    input: str | None = None
    node_type: Literal["checkpoint"] = "checkpoint"
    is_checkpoint: Literal[True] = True
    checkpoint_queries: list[CheckpointQuery]


Event = Union[ActionEvent, CheckpointEvent]


class ToolCall(BaseModel):
    tool: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class AgentAction(BaseModel):
    tool_calls: list[ToolCall]


class CheckpointResponse(BaseModel):
    answers: dict[str, str]


class Scenario(BaseModel):
    scenario_id: str
    domain: str
    role: str
    difficulty: int
    background_docs: list[str]
    background_tokens: int
    total_events: int
    total_checkpoints: int
    annotator: str
    annotator_agreement: float | None = None
    generation_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[ActionEvent | CheckpointEvent]

    @property
    def action_events(self) -> list[ActionEvent]:
        return [e for e in self.events if e.node_type == "action"]

    @property
    def checkpoint_events(self) -> list[CheckpointEvent]:
        return [e for e in self.events if e.node_type == "checkpoint"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schema.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/schema.py tests/test_schema.py
git commit -m "feat: add Pydantic schema models for scenarios, events, scoring rules"
```

---

### Task 3: Tool Registry

**Files:**
- Create: `longhorizon_bench/tools/__init__.py`
- Create: `longhorizon_bench/tools/registry.py`
- Create: `longhorizon_bench/tools/common.py`
- Create: `longhorizon_bench/tools/industrial.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests for tool registry**

```python
# tests/test_tools.py
import pytest


def test_common_tools_registered():
    from longhorizon_bench.tools.registry import ToolRegistry
    from longhorizon_bench.tools.common import COMMON_TOOLS

    reg = ToolRegistry(COMMON_TOOLS)
    assert "retrieve_policy" in reg
    assert "inspect_history" in reg
    assert "request_clarification" in reg
    assert "no_action" in reg
    assert "respond_to_query" in reg


def test_industrial_tools_registered():
    from longhorizon_bench.tools.industrial import INDUSTRIAL_TOOLS

    assert "submit_inspection_report" in INDUSTRIAL_TOOLS
    assert "file_incident_report" in INDUSTRIAL_TOOLS
    assert "approve_work_permit" in INDUSTRIAL_TOOLS


def test_registry_merge():
    from longhorizon_bench.tools.registry import ToolRegistry
    from longhorizon_bench.tools.common import COMMON_TOOLS
    from longhorizon_bench.tools.industrial import INDUSTRIAL_TOOLS

    reg = ToolRegistry({**COMMON_TOOLS, **INDUSTRIAL_TOOLS})
    assert "retrieve_policy" in reg
    assert "submit_inspection_report" in reg


def test_validate_action_valid():
    from longhorizon_bench.tools.registry import ToolRegistry, ToolDef
    from longhorizon_bench.schema import ToolCall

    reg = ToolRegistry({
        "no_action": ToolDef(
            params={"reason": {"type": "str"}},
        ),
    })
    errors = reg.validate_tool_call(ToolCall(tool="no_action", kwargs={"reason": "无需操作"}))
    assert errors == []


def test_validate_action_unknown_tool():
    from longhorizon_bench.tools.registry import ToolRegistry
    from longhorizon_bench.schema import ToolCall

    reg = ToolRegistry({})
    errors = reg.validate_tool_call(ToolCall(tool="nonexistent", kwargs={}))
    assert len(errors) == 1
    assert "unknown tool" in errors[0].lower()


def test_validate_action_missing_required_param():
    from longhorizon_bench.tools.registry import ToolRegistry, ToolDef
    from longhorizon_bench.schema import ToolCall

    reg = ToolRegistry({
        "submit_inspection_report": ToolDef(
            params={
                "target": {"type": "str"},
                "findings": {"type": "str"},
                "risk_level": {"type": "enum", "values": ["low", "medium", "high", "critical"]},
            },
        ),
    })
    errors = reg.validate_tool_call(
        ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"})
    )
    assert len(errors) == 2  # missing findings + risk_level


def test_validate_enum_param_invalid_value():
    from longhorizon_bench.tools.registry import ToolRegistry, ToolDef
    from longhorizon_bench.schema import ToolCall

    reg = ToolRegistry({
        "file_incident_report": ToolDef(
            params={
                "incident_type": {"type": "enum", "values": ["injury", "leak", "fire"]},
                "severity": {"type": "enum", "values": ["minor", "major", "critical"]},
                "description": {"type": "str"},
            },
        ),
    })
    errors = reg.validate_tool_call(
        ToolCall(
            tool="file_incident_report",
            kwargs={"incident_type": "explosion", "severity": "major", "description": "test"},
        )
    )
    assert len(errors) == 1
    assert "incident_type" in errors[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL

- [ ] **Step 3: Implement tool registry and definitions**

```python
# longhorizon_bench/tools/registry.py
"""Tool registry: stores definitions, validates agent actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from longhorizon_bench.schema import ToolCall


@dataclass
class ToolDef:
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    note: str = ""


class ToolRegistry:
    def __init__(self, tools: dict[str, ToolDef | dict]) -> None:
        self._tools: dict[str, ToolDef] = {}
        for name, defn in tools.items():
            if isinstance(defn, dict):
                defn = ToolDef(**defn)
            self._tools[name] = defn

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def all_names(self) -> list[str]:
        return list(self._tools.keys())

    def validate_tool_call(self, call: ToolCall) -> list[str]:
        errors: list[str] = []
        defn = self._tools.get(call.tool)
        if defn is None:
            errors.append(f"Unknown tool: {call.tool}")
            return errors

        for param_name, param_spec in defn.params.items():
            if param_name not in call.kwargs:
                errors.append(f"Missing required param: {param_name}")
                continue

            value = call.kwargs[param_name]
            if param_spec.get("type") == "enum":
                allowed = param_spec.get("values", [])
                if value not in allowed:
                    errors.append(
                        f"Invalid value for {param_name}: "
                        f"'{value}' not in {allowed}"
                    )
        return errors
```

```python
# longhorizon_bench/tools/common.py
"""Common tools shared across all domains."""

from longhorizon_bench.tools.registry import ToolDef

COMMON_TOOLS: dict[str, ToolDef] = {
    "retrieve_policy": ToolDef(
        params={"query": {"type": "str"}},
        note="查询背景文档中的相关条款，Mode C 下有调用次数限制",
    ),
    "inspect_history": ToolDef(
        params={"event_id": {"type": "str"}},
        note="查看指定历史事件的详情，Mode C 下有调用次数限制",
    ),
    "request_clarification": ToolDef(
        params={"question": {"type": "str"}},
        note="向上级/同事请求补充信息",
    ),
    "no_action": ToolDef(
        params={"reason": {"type": "str"}},
        note="判断当前事件无需采取行动，需说明理由",
    ),
    "respond_to_query": ToolDef(
        params={"recipient": {"type": "str"}, "content": {"type": "str"}},
    ),
}
```

```python
# longhorizon_bench/tools/industrial.py
"""Industrial domain tool definitions."""

from longhorizon_bench.tools.registry import ToolDef

INDUSTRIAL_TOOLS: dict[str, ToolDef] = {
    "submit_inspection_report": ToolDef(params={
        "target": {"type": "str"},
        "findings": {"type": "str"},
        "risk_level": {"type": "enum", "values": ["low", "medium", "high", "critical"]},
    }),
    "update_safety_ledger": ToolDef(params={
        "item_id": {"type": "str"},
        "status": {"type": "enum", "values": ["open", "in_progress", "resolved", "overdue"]},
        "remarks": {"type": "str"},
    }),
    "file_incident_report": ToolDef(params={
        "incident_type": {"type": "enum", "values": ["injury", "leak", "fire", "equipment_failure", "violation"]},
        "severity": {"type": "enum", "values": ["minor", "major", "critical"]},
        "description": {"type": "str"},
    }),
    "issue_rectification_order": ToolDef(params={
        "target_dept": {"type": "str"},
        "issues": {"type": "list[str]"},
        "deadline": {"type": "date"},
    }),
    "request_equipment_shutdown": ToolDef(params={
        "equipment_id": {"type": "str"},
        "reason": {"type": "str"},
        "duration": {"type": "str"},
    }),
    "approve_work_permit": ToolDef(params={
        "permit_type": {"type": "enum", "values": ["hot_work", "confined_space", "height", "electrical"]},
        "conditions": {"type": "list[str]"},
        "approved": {"type": "bool"},
    }),
    "escalate_to_management": ToolDef(params={
        "issue_summary": {"type": "str"},
        "urgency": {"type": "enum", "values": ["routine", "urgent", "emergency"]},
        "recommendation": {"type": "str"},
    }),
    "notify_regulatory_body": ToolDef(params={
        "authority": {"type": "str"},
        "event_type": {"type": "str"},
        "details": {"type": "str"},
    }),
    "schedule_training": ToolDef(params={
        "topic": {"type": "str"},
        "participants": {"type": "list[str]"},
        "date": {"type": "date"},
    }),
    "allocate_budget": ToolDef(params={
        "item": {"type": "str"},
        "amount": {"type": "float"},
        "justification": {"type": "str"},
    }),
    "assign_personnel": ToolDef(params={
        "task": {"type": "str"},
        "assignee": {"type": "str"},
        "priority": {"type": "enum", "values": ["low", "medium", "high"]},
    }),
}
```

```python
# longhorizon_bench/tools/__init__.py
"""Tool definitions and registry."""

from longhorizon_bench.tools.registry import ToolDef, ToolRegistry
from longhorizon_bench.tools.common import COMMON_TOOLS
from longhorizon_bench.tools.industrial import INDUSTRIAL_TOOLS


def build_registry(domain: str) -> ToolRegistry:
    domain_tools: dict[str, ToolDef] = {
        "industrial": INDUSTRIAL_TOOLS,
    }
    tools = {**COMMON_TOOLS, **(domain_tools.get(domain, {}))}
    return ToolRegistry(tools)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/tools/ tests/test_tools.py
git commit -m "feat: add tool registry with common and industrial domain tools"
```

---

### Task 4: Event Scorer

**Files:**
- Create: `longhorizon_bench/evaluation/__init__.py`
- Create: `longhorizon_bench/evaluation/scorer.py`
- Create: `tests/test_scorer.py`

- [ ] **Step 1: Write failing tests for scorer**

```python
# tests/test_scorer.py
import pytest
from longhorizon_bench.schema import (
    ActionEvent, AgentAction, ToolCall, ScoringRule, ParamRule, Evidence,
    CheckpointEvent, CheckpointQuery, CheckpointResponse,
)


def _make_event(**overrides) -> ActionEvent:
    defaults = dict(
        event_id="E01",
        type="routine_inspection",
        input="测试",
        depends_on=[],
        is_critical=False,
        dimensions=["domain_knowledge"],
        scoring_rule=ScoringRule(
            tool=ParamRule(expected="submit_inspection_report", match="exact"),
            params={
                "target": ParamRule(expected="3号车间", match="contains"),
                "risk_level": ParamRule(expected=["medium", "high"], match="enum"),
                "findings": ParamRule(required_keywords=["压力容器", "检验周期"], match="keyword_coverage"),
            },
        ),
        evidence=Evidence(required_facts=[], forbidden_actions=[], acceptable_actions=[]),
    )
    defaults.update(overrides)
    return ActionEvent(**defaults)


def test_perfect_score():
    from longhorizon_bench.evaluation.scorer import EventScorer

    scorer = EventScorer()
    event = _make_event()
    action = AgentAction(tool_calls=[
        ToolCall(tool="submit_inspection_report", kwargs={
            "target": "3号车间设备",
            "risk_level": "high",
            "findings": "发现压力容器接近检验周期，需要排查",
        }),
    ])
    result = scorer.score_action(event, action)
    assert result.tool_score == 1.0
    assert result.param_scores["target"] == 1.0
    assert result.param_scores["risk_level"] == 1.0
    assert result.param_scores["findings"] == 1.0
    assert result.total == 1.0


def test_wrong_tool():
    from longhorizon_bench.evaluation.scorer import EventScorer

    scorer = EventScorer()
    event = _make_event()
    action = AgentAction(tool_calls=[
        ToolCall(tool="no_action", kwargs={"reason": "无需操作"}),
    ])
    result = scorer.score_action(event, action)
    assert result.tool_score == 0.0
    assert result.total < 0.5


def test_exact_match_wrong_value():
    from longhorizon_bench.evaluation.scorer import EventScorer

    event = _make_event(
        scoring_rule=ScoringRule(
            tool=ParamRule(expected="no_action", match="exact"),
            params={"reason": ParamRule(expected="无需操作", match="exact")},
        ),
    )
    scorer = EventScorer()
    action = AgentAction(tool_calls=[
        ToolCall(tool="no_action", kwargs={"reason": "不需要做什么"}),
    ])
    result = scorer.score_action(event, action)
    assert result.tool_score == 1.0
    assert result.param_scores["reason"] == 0.0


def test_contains_match():
    from longhorizon_bench.evaluation.scorer import EventScorer

    scorer = EventScorer()
    event = _make_event(
        scoring_rule=ScoringRule(
            tool=ParamRule(expected="respond_to_query", match="exact"),
            params={"content": ParamRule(expected="安全隐患", match="contains")},
        ),
    )
    action = AgentAction(tool_calls=[
        ToolCall(tool="respond_to_query", kwargs={"content": "经排查，存在安全隐患需要整改"}),
    ])
    result = scorer.score_action(event, action)
    assert result.param_scores["content"] == 1.0


def test_enum_match_accepts_any_in_list():
    from longhorizon_bench.evaluation.scorer import EventScorer

    scorer = EventScorer()
    event = _make_event()
    action = AgentAction(tool_calls=[
        ToolCall(tool="submit_inspection_report", kwargs={
            "target": "3号车间",
            "risk_level": "medium",
            "findings": "压力容器检验周期到了",
        }),
    ])
    result = scorer.score_action(event, action)
    assert result.param_scores["risk_level"] == 1.0


def test_keyword_coverage_partial():
    from longhorizon_bench.evaluation.scorer import EventScorer

    scorer = EventScorer()
    event = _make_event()
    action = AgentAction(tool_calls=[
        ToolCall(tool="submit_inspection_report", kwargs={
            "target": "3号车间",
            "risk_level": "high",
            "findings": "发现压力容器有问题",  # missing 检验周期
        }),
    ])
    result = scorer.score_action(event, action)
    assert result.param_scores["findings"] == 0.5


def test_missing_param_scores_zero():
    from longhorizon_bench.evaluation.scorer import EventScorer

    scorer = EventScorer()
    event = _make_event()
    action = AgentAction(tool_calls=[
        ToolCall(tool="submit_inspection_report", kwargs={
            "target": "3号车间",
        }),
    ])
    result = scorer.score_action(event, action)
    assert result.param_scores["risk_level"] == 0.0
    assert result.param_scores["findings"] == 0.0


def test_multi_tool_action_scores_first_matching():
    from longhorizon_bench.evaluation.scorer import EventScorer

    scorer = EventScorer()
    event = _make_event()
    action = AgentAction(tool_calls=[
        ToolCall(tool="update_safety_ledger", kwargs={"item_id": "x", "status": "open", "remarks": "y"}),
        ToolCall(tool="submit_inspection_report", kwargs={
            "target": "3号车间",
            "risk_level": "high",
            "findings": "压力容器检验周期到了",
        }),
    ])
    result = scorer.score_action(event, action)
    assert result.tool_score == 1.0


def test_checkpoint_scoring():
    from longhorizon_bench.evaluation.scorer import EventScorer

    scorer = EventScorer()
    event = CheckpointEvent(
        event_id="CP01",
        type="checkpoint",
        checkpoint_queries=[
            CheckpointQuery(
                query="E02中发现的隐患当前状态？",
                expected_keywords=["整改中", "未完成", "3号车间"],
                dimension="long_term_memory",
                match="keyword_coverage",
            ),
        ],
    )
    response = CheckpointResponse(
        answers={"E02中发现的隐患当前状态？": "3号车间的隐患目前整改中，尚未完成"}
    )
    result = scorer.score_checkpoint(event, response)
    assert result.query_scores[0] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scorer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement scorer.py**

```python
# longhorizon_bench/evaluation/__init__.py
"""Evaluation components."""
```

```python
# longhorizon_bench/evaluation/scorer.py
"""Event-level scoring engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from longhorizon_bench.schema import (
    ActionEvent,
    AgentAction,
    CheckpointEvent,
    CheckpointResponse,
    MatchType,
    ParamRule,
)


@dataclass
class ActionScore:
    tool_score: float
    param_scores: dict[str, float]
    total: float


@dataclass
class CheckpointScore:
    query_scores: list[float]
    total: float


class EventScorer:
    def score_action(self, event: ActionEvent, action: AgentAction) -> ActionScore:
        rule = event.scoring_rule

        matching_call = None
        for call in action.tool_calls:
            if call.tool == rule.tool.expected:
                matching_call = call
                break

        if matching_call is None:
            tool_score = 0.0
            param_scores = {name: 0.0 for name in rule.params}
        else:
            tool_score = 1.0
            param_scores = {}
            for name, param_rule in rule.params.items():
                value = matching_call.kwargs.get(name)
                if value is None:
                    param_scores[name] = 0.0
                else:
                    param_scores[name] = self._match_param(param_rule, value)

        if not rule.params:
            total = tool_score
        else:
            param_avg = sum(param_scores.values()) / len(param_scores) if param_scores else 0.0
            total = tool_score * 0.4 + param_avg * 0.6

        return ActionScore(tool_score=tool_score, param_scores=param_scores, total=total)

    def score_checkpoint(
        self, event: CheckpointEvent, response: CheckpointResponse
    ) -> CheckpointScore:
        query_scores: list[float] = []
        for q in event.checkpoint_queries:
            answer = response.answers.get(q.query, "")
            if q.match == "keyword_coverage" and q.expected_keywords:
                hits = sum(1 for kw in q.expected_keywords if kw in answer)
                query_scores.append(hits / len(q.expected_keywords))
            else:
                query_scores.append(0.0)

        total = sum(query_scores) / len(query_scores) if query_scores else 0.0
        return CheckpointScore(query_scores=query_scores, total=total)

    def _match_param(self, rule: ParamRule, value: str) -> float:
        match rule.match:
            case MatchType.EXACT:
                return 1.0 if str(value) == str(rule.expected) else 0.0
            case MatchType.CONTAINS:
                return 1.0 if str(rule.expected) in str(value) else 0.0
            case MatchType.ENUM:
                allowed = rule.expected if isinstance(rule.expected, list) else [rule.expected]
                return 1.0 if value in allowed else 0.0
            case MatchType.KEYWORD_COVERAGE:
                if not rule.required_keywords:
                    return 1.0
                hits = sum(1 for kw in rule.required_keywords if kw in str(value))
                return hits / len(rule.required_keywords)
            case _:
                return 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scorer.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/evaluation/ tests/test_scorer.py
git commit -m "feat: add event-level scorer with multi-match-type param evaluation"
```

---

### Task 5: Consistency Checker

**Files:**
- Create: `longhorizon_bench/evaluation/consistency.py`
- Create: `tests/test_consistency.py`

- [ ] **Step 1: Write failing tests for consistency rules C01–C06**

```python
# tests/test_consistency.py
import pytest
from longhorizon_bench.schema import ToolCall


def test_c01_status_regression_detected():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker

    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="update_safety_ledger", kwargs={"item_id": "ITEM-1", "status": "resolved", "remarks": ""})]),
        ("E05", [ToolCall(tool="update_safety_ledger", kwargs={"item_id": "ITEM-1", "status": "open", "remarks": ""})]),
    ]
    violations = checker.check_c01_status_regression(history)
    assert len(violations) == 1
    assert violations[0].rule_id == "C01"
    assert "ITEM-1" in violations[0].description


def test_c01_no_violation_on_forward_progress():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker

    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="update_safety_ledger", kwargs={"item_id": "ITEM-1", "status": "open", "remarks": ""})]),
        ("E05", [ToolCall(tool="update_safety_ledger", kwargs={"item_id": "ITEM-1", "status": "resolved", "remarks": ""})]),
    ]
    violations = checker.check_c01_status_regression(history)
    assert len(violations) == 0


def test_c02_risk_level_decrease_after_incident():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker

    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="submit_inspection_report", kwargs={"target": "A", "findings": "", "risk_level": "high"})]),
        ("E02", [ToolCall(tool="file_incident_report", kwargs={"incident_type": "leak", "severity": "major", "description": ""})]),
        ("E03", [ToolCall(tool="submit_inspection_report", kwargs={"target": "A", "findings": "", "risk_level": "low"})]),
    ]
    violations = checker.check_c02_risk_monotonicity(history)
    assert len(violations) == 1
    assert violations[0].rule_id == "C02"


def test_c02_no_violation_when_risk_stays_or_increases():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker

    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="submit_inspection_report", kwargs={"target": "A", "findings": "", "risk_level": "medium"})]),
        ("E02", [ToolCall(tool="file_incident_report", kwargs={"incident_type": "leak", "severity": "major", "description": ""})]),
        ("E03", [ToolCall(tool="submit_inspection_report", kwargs={"target": "A", "findings": "", "risk_level": "high"})]),
    ]
    violations = checker.check_c02_risk_monotonicity(history)
    assert len(violations) == 0


def test_c03_permit_without_inspection():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker

    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="approve_work_permit", kwargs={"permit_type": "hot_work", "conditions": [], "approved": True})]),
    ]
    violations = checker.check_c03_permit_prerequisite(history)
    assert len(violations) == 1
    assert violations[0].rule_id == "C03"


def test_c03_no_violation_with_prior_inspection():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker

    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="submit_inspection_report", kwargs={"target": "区域A", "findings": "合格", "risk_level": "low"})]),
        ("E02", [ToolCall(tool="approve_work_permit", kwargs={"permit_type": "hot_work", "conditions": [], "approved": True})]),
    ]
    violations = checker.check_c03_permit_prerequisite(history)
    assert len(violations) == 0


def test_c05_assignee_change_without_reason():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker

    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="assign_personnel", kwargs={"task": "巡检", "assignee": "张三", "priority": "high"})]),
        ("E05", [ToolCall(tool="assign_personnel", kwargs={"task": "巡检", "assignee": "李四", "priority": "high"})]),
    ]
    violations = checker.check_c05_assignee_consistency(history)
    assert len(violations) == 1
    assert violations[0].rule_id == "C05"


def test_c06_deadline_before_current_event():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker

    checker = ConsistencyChecker()
    history = [
        ("E10", [ToolCall(tool="issue_rectification_order", kwargs={
            "target_dept": "设备科", "issues": ["泄漏"], "deadline": "2025-01-01",
        })]),
    ]
    event_times = {"E10": "2025-06-01"}
    violations = checker.check_c06_timeline_validity(history, event_times)
    assert len(violations) == 1
    assert violations[0].rule_id == "C06"


def test_check_all_returns_combined():
    from longhorizon_bench.evaluation.consistency import ConsistencyChecker

    checker = ConsistencyChecker()
    history = [
        ("E01", [ToolCall(tool="approve_work_permit", kwargs={"permit_type": "hot_work", "conditions": [], "approved": True})]),
    ]
    violations = checker.check_all(history, applicable_rules=["C03"])
    assert len(violations) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_consistency.py -v`
Expected: FAIL

- [ ] **Step 3: Implement consistency.py**

```python
# longhorizon_bench/evaluation/consistency.py
"""Consistency rules C01–C06 for cross-event contradiction detection."""

from __future__ import annotations

from dataclasses import dataclass
from longhorizon_bench.schema import ToolCall

STATUS_ORDER = {"open": 0, "in_progress": 1, "resolved": 2, "overdue": 1}
RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

ActionHistory = list[tuple[str, list[ToolCall]]]


@dataclass
class Violation:
    rule_id: str
    event_id: str
    description: str


class ConsistencyChecker:
    def check_c01_status_regression(self, history: ActionHistory) -> list[Violation]:
        latest_status: dict[str, tuple[str, int]] = {}
        violations: list[Violation] = []
        for event_id, calls in history:
            for call in calls:
                if call.tool == "update_safety_ledger":
                    item = call.kwargs.get("item_id", "")
                    status = call.kwargs.get("status", "")
                    order = STATUS_ORDER.get(status, -1)
                    if item in latest_status:
                        prev_event, prev_order = latest_status[item]
                        if order < prev_order:
                            violations.append(Violation(
                                rule_id="C01",
                                event_id=event_id,
                                description=f"Status regression for {item}: "
                                           f"was order {prev_order} at {prev_event}, "
                                           f"now {order} ({status})",
                            ))
                    latest_status[item] = (event_id, order)
        return violations

    def check_c02_risk_monotonicity(self, history: ActionHistory) -> list[Violation]:
        violations: list[Violation] = []
        incident_seen = False
        pre_incident_risk: dict[str, int] = {}

        for event_id, calls in history:
            for call in calls:
                if call.tool == "file_incident_report":
                    incident_seen = True
                if call.tool == "submit_inspection_report":
                    target = call.kwargs.get("target", "")
                    level = call.kwargs.get("risk_level", "")
                    order = RISK_ORDER.get(level, -1)
                    if not incident_seen:
                        pre_incident_risk[target] = max(
                            pre_incident_risk.get(target, 0), order
                        )
                    elif target in pre_incident_risk and order < pre_incident_risk[target]:
                        violations.append(Violation(
                            rule_id="C02",
                            event_id=event_id,
                            description=f"Risk decreased for '{target}' after incident: "
                                       f"was {pre_incident_risk[target]}, now {order} ({level})",
                        ))
        return violations

    def check_c03_permit_prerequisite(self, history: ActionHistory) -> list[Violation]:
        violations: list[Violation] = []
        inspection_done = False
        for event_id, calls in history:
            for call in calls:
                if call.tool == "submit_inspection_report":
                    inspection_done = True
                if call.tool == "approve_work_permit" and not inspection_done:
                    violations.append(Violation(
                        rule_id="C03",
                        event_id=event_id,
                        description="Work permit approved without prior inspection report",
                    ))
        return violations

    def check_c04_rectification_closure(self, history: ActionHistory) -> list[Violation]:
        open_orders: dict[str, str] = {}
        for event_id, calls in history:
            for call in calls:
                if call.tool == "issue_rectification_order":
                    dept = call.kwargs.get("target_dept", "")
                    open_orders[dept] = event_id
                if call.tool == "update_safety_ledger":
                    status = call.kwargs.get("status", "")
                    if status == "resolved":
                        remarks = call.kwargs.get("remarks", "")
                        for dept in list(open_orders):
                            if dept in remarks or not open_orders:
                                del open_orders[dept]

        return [
            Violation(
                rule_id="C04",
                event_id=eid,
                description=f"Rectification order for '{dept}' never closed",
            )
            for dept, eid in open_orders.items()
        ]

    def check_c05_assignee_consistency(self, history: ActionHistory) -> list[Violation]:
        violations: list[Violation] = []
        task_assignees: dict[str, tuple[str, str]] = {}
        for event_id, calls in history:
            for call in calls:
                if call.tool == "assign_personnel":
                    task = call.kwargs.get("task", "")
                    assignee = call.kwargs.get("assignee", "")
                    if task in task_assignees:
                        prev_event, prev_assignee = task_assignees[task]
                        if assignee != prev_assignee:
                            violations.append(Violation(
                                rule_id="C05",
                                event_id=event_id,
                                description=f"Assignee changed for '{task}': "
                                           f"'{prev_assignee}' -> '{assignee}' without explanation",
                            ))
                    task_assignees[task] = (event_id, assignee)
        return violations

    def check_c06_timeline_validity(
        self, history: ActionHistory, event_times: dict[str, str] | None = None
    ) -> list[Violation]:
        if not event_times:
            return []
        violations: list[Violation] = []
        for event_id, calls in history:
            event_time = event_times.get(event_id)
            if not event_time:
                continue
            for call in calls:
                if call.tool == "issue_rectification_order":
                    deadline = call.kwargs.get("deadline", "")
                    if deadline and deadline < event_time:
                        violations.append(Violation(
                            rule_id="C06",
                            event_id=event_id,
                            description=f"Deadline {deadline} is before event time {event_time}",
                        ))
        return violations

    def check_all(
        self,
        history: ActionHistory,
        applicable_rules: list[str] | None = None,
        event_times: dict[str, str] | None = None,
    ) -> list[Violation]:
        all_rules = {
            "C01": lambda: self.check_c01_status_regression(history),
            "C02": lambda: self.check_c02_risk_monotonicity(history),
            "C03": lambda: self.check_c03_permit_prerequisite(history),
            "C04": lambda: self.check_c04_rectification_closure(history),
            "C05": lambda: self.check_c05_assignee_consistency(history),
            "C06": lambda: self.check_c06_timeline_validity(history, event_times),
        }
        rules = applicable_rules or list(all_rules.keys())
        violations: list[Violation] = []
        for rule_id in rules:
            if rule_id in all_rules:
                violations.extend(all_rules[rule_id]())
        return violations
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_consistency.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/evaluation/consistency.py tests/test_consistency.py
git commit -m "feat: add consistency checker with rules C01-C06"
```

---

### Task 6: Metrics Aggregation

**Files:**
- Create: `longhorizon_bench/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests for multi-level metrics**

```python
# tests/test_metrics.py
import pytest
from longhorizon_bench.evaluation.scorer import ActionScore, CheckpointScore
from longhorizon_bench.evaluation.consistency import Violation


def test_event_accuracy():
    from longhorizon_bench.evaluation.metrics import compute_metrics

    event_scores = [
        {"event_id": "E01", "is_critical": True, "score": ActionScore(tool_score=1.0, param_scores={"a": 1.0}, total=1.0)},
        {"event_id": "E02", "is_critical": False, "score": ActionScore(tool_score=1.0, param_scores={"a": 0.5}, total=0.7)},
        {"event_id": "E03", "is_critical": True, "score": ActionScore(tool_score=0.0, param_scores={"a": 0.0}, total=0.0)},
    ]
    result = compute_metrics(event_scores=event_scores, checkpoint_scores=[], violations=[])
    assert result.event_accuracy == pytest.approx((1.0 + 0.7 + 0.0) / 3, abs=0.01)
    assert result.critical_event_accuracy == pytest.approx((1.0 + 0.0) / 2, abs=0.01)


def test_chain_score_weights_critical_events():
    from longhorizon_bench.evaluation.metrics import compute_metrics

    event_scores = [
        {"event_id": "E01", "is_critical": True, "score": ActionScore(tool_score=1.0, param_scores={}, total=1.0)},
        {"event_id": "E02", "is_critical": False, "score": ActionScore(tool_score=1.0, param_scores={}, total=0.5)},
    ]
    result = compute_metrics(event_scores=event_scores, checkpoint_scores=[], violations=[])
    # weighted: (1.0*2 + 0.5*1) / (2+1) = 2.5/3 ≈ 0.833
    assert result.chain_score == pytest.approx(2.5 / 3, abs=0.01)


def test_chain_pass_strict():
    from longhorizon_bench.evaluation.metrics import compute_metrics

    event_scores = [
        {"event_id": "E01", "is_critical": False, "score": ActionScore(tool_score=1.0, param_scores={}, total=1.0)},
        {"event_id": "E02", "is_critical": False, "score": ActionScore(tool_score=0.0, param_scores={}, total=0.0)},
    ]
    result = compute_metrics(event_scores=event_scores, checkpoint_scores=[], violations=[])
    assert result.chain_pass is False


def test_chain_pass_true_when_all_tools_correct_no_violations():
    from longhorizon_bench.evaluation.metrics import compute_metrics

    event_scores = [
        {"event_id": "E01", "is_critical": False, "score": ActionScore(tool_score=1.0, param_scores={"a": 0.8}, total=0.88)},
        {"event_id": "E02", "is_critical": False, "score": ActionScore(tool_score=1.0, param_scores={"a": 1.0}, total=1.0)},
    ]
    result = compute_metrics(event_scores=event_scores, checkpoint_scores=[], violations=[])
    assert result.chain_pass is True


def test_chain_pass_false_with_violations():
    from longhorizon_bench.evaluation.metrics import compute_metrics

    event_scores = [
        {"event_id": "E01", "is_critical": False, "score": ActionScore(tool_score=1.0, param_scores={}, total=1.0)},
    ]
    violations = [Violation(rule_id="C01", event_id="E01", description="test")]
    result = compute_metrics(event_scores=event_scores, checkpoint_scores=[], violations=violations)
    assert result.chain_pass is False


def test_checkpoint_dimension_scores():
    from longhorizon_bench.evaluation.metrics import compute_metrics

    checkpoint_scores = [
        {"dimension": "long_term_memory", "score": 0.8},
        {"dimension": "long_term_memory", "score": 0.6},
        {"dimension": "consistency", "score": 1.0},
    ]
    result = compute_metrics(event_scores=[], checkpoint_scores=checkpoint_scores, violations=[])
    assert result.dimension_scores["long_term_memory"] == pytest.approx(0.7, abs=0.01)
    assert result.dimension_scores["consistency"] == pytest.approx(1.0, abs=0.01)


def test_consistency_violation_rate():
    from longhorizon_bench.evaluation.metrics import compute_metrics

    violations = [
        Violation(rule_id="C01", event_id="E01", description="test"),
        Violation(rule_id="C03", event_id="E05", description="test"),
    ]
    result = compute_metrics(event_scores=[], checkpoint_scores=[], violations=violations, applicable_rule_checks=10)
    assert result.consistency_violation_rate == pytest.approx(0.2, abs=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement metrics.py**

```python
# longhorizon_bench/evaluation/metrics.py
"""Multi-level metrics aggregation: L1 event, L2 dimension, L3 chain."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from longhorizon_bench.evaluation.scorer import ActionScore, CheckpointScore
from longhorizon_bench.evaluation.consistency import Violation


@dataclass
class EventScoreEntry:
    event_id: str
    is_critical: bool
    score: ActionScore


@dataclass
class CheckpointScoreEntry:
    dimension: str
    score: float


@dataclass
class EvalResult:
    event_accuracy: float
    critical_event_accuracy: float
    chain_score: float
    chain_pass: bool
    dimension_scores: dict[str, float]
    consistency_violation_rate: float
    violations: list[Violation]
    event_scores: list[dict]


def compute_metrics(
    event_scores: list[dict],
    checkpoint_scores: list[dict],
    violations: list[Violation],
    applicable_rule_checks: int = 0,
) -> EvalResult:
    all_totals = [e["score"].total for e in event_scores]
    critical_totals = [e["score"].total for e in event_scores if e["is_critical"]]

    event_accuracy = sum(all_totals) / len(all_totals) if all_totals else 0.0
    critical_event_accuracy = (
        sum(critical_totals) / len(critical_totals) if critical_totals else 0.0
    )

    weighted_sum = 0.0
    weight_total = 0.0
    for e in event_scores:
        w = 2.0 if e["is_critical"] else 1.0
        weighted_sum += e["score"].total * w
        weight_total += w
    chain_score = weighted_sum / weight_total if weight_total else 0.0

    all_tools_correct = all(e["score"].tool_score == 1.0 for e in event_scores)
    chain_pass = all_tools_correct and len(violations) == 0

    dim_groups: dict[str, list[float]] = defaultdict(list)
    for cp in checkpoint_scores:
        dim_groups[cp["dimension"]].append(cp["score"])
    dimension_scores = {
        dim: sum(vals) / len(vals) for dim, vals in dim_groups.items()
    }

    violation_rate = (
        len(violations) / applicable_rule_checks
        if applicable_rule_checks > 0
        else 0.0
    )

    return EvalResult(
        event_accuracy=event_accuracy,
        critical_event_accuracy=critical_event_accuracy,
        chain_score=chain_score,
        chain_pass=chain_pass,
        dimension_scores=dimension_scores,
        consistency_violation_rate=violation_rate,
        violations=violations,
        event_scores=event_scores,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add multi-level metrics aggregation (event/dimension/chain)"
```

---

### Task 7: Scenario Loader

**Files:**
- Create: `longhorizon_bench/loader.py`
- Create: `data/scenarios/industrial/DEMO-001.json`
- Create: `data/background_docs/DEMO-001/doc_001.txt`
- Create: `tests/test_loader.py`

- [ ] **Step 1: Write failing tests for loader**

```python
# tests/test_loader.py
import json
import pytest
from pathlib import Path


@pytest.fixture
def demo_scenario_path(tmp_path: Path) -> Path:
    scenario_dir = tmp_path / "scenarios" / "industrial"
    scenario_dir.mkdir(parents=True)
    bg_dir = tmp_path / "background_docs" / "DEMO-001"
    bg_dir.mkdir(parents=True)
    (bg_dir / "doc_001.txt").write_text("这是背景文档内容，压力容器定期检验周期为3年。", encoding="utf-8")

    scenario = {
        "scenario_id": "DEMO-001",
        "domain": "industrial",
        "role": "化工厂安全生产管理员",
        "difficulty": 1,
        "background_docs": ["doc_001.txt"],
        "background_tokens": 50,
        "total_events": 2,
        "total_checkpoints": 1,
        "annotator": "test",
        "annotator_agreement": 0.85,
        "generation_model": "test-model",
        "metadata": {"source": "test"},
        "events": [
            {
                "event_id": "E01",
                "type": "routine_inspection",
                "input": "收到设备科提交的3号车间季度巡检报告，其中提到1台压力容器接近检验周期。",
                "depends_on": [],
                "node_type": "action",
                "is_checkpoint": False,
                "is_critical": True,
                "dimensions": ["domain_knowledge"],
                "scoring_rule": {
                    "tool": {"expected": "submit_inspection_report", "match": "exact"},
                    "params": {
                        "target": {"expected": "3号车间", "match": "contains"},
                        "findings": {"required_keywords": ["压力容器", "检验周期"], "match": "keyword_coverage"},
                        "risk_level": {"expected": ["medium", "high"], "match": "enum"},
                    },
                },
                "evidence": {
                    "required_facts": ["背景文档: 压力容器定期检验周期为3年"],
                    "forbidden_actions": ["approve_work_permit"],
                    "acceptable_actions": [{"tool": "submit_inspection_report"}],
                },
            },
            {
                "event_id": "CP01",
                "type": "checkpoint",
                "input": None,
                "node_type": "checkpoint",
                "is_checkpoint": True,
                "checkpoint_queries": [
                    {
                        "query": "E01中提到的压力容器情况是什么？",
                        "expected_keywords": ["压力容器", "检验周期"],
                        "dimension": "long_term_memory",
                        "match": "keyword_coverage",
                    },
                ],
            },
        ],
    }
    (scenario_dir / "DEMO-001.json").write_text(json.dumps(scenario, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def test_load_scenario_from_path(demo_scenario_path):
    from longhorizon_bench.loader import load_scenario

    scenario = load_scenario("industrial/DEMO-001", data_dir=demo_scenario_path)
    assert scenario.scenario_id == "DEMO-001"
    assert len(scenario.events) == 2


def test_load_background_docs(demo_scenario_path):
    from longhorizon_bench.loader import load_background_docs

    docs = load_background_docs("DEMO-001", ["doc_001.txt"], data_dir=demo_scenario_path)
    assert "压力容器" in docs["doc_001.txt"]


def test_load_scenario_file_not_found():
    from longhorizon_bench.loader import load_scenario

    with pytest.raises(FileNotFoundError):
        load_scenario("industrial/NONEXISTENT", data_dir=Path("/nonexistent"))


def test_load_scenario_validation_error(tmp_path):
    from longhorizon_bench.loader import load_scenario

    scenario_dir = tmp_path / "scenarios" / "industrial"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "BAD-001.json").write_text('{"scenario_id": "BAD"}', encoding="utf-8")

    with pytest.raises(Exception):  # Pydantic ValidationError
        load_scenario("industrial/BAD-001", data_dir=tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Implement loader.py**

```python
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
```

- [ ] **Step 4: Create the demo scenario data files**

Write `data/scenarios/industrial/DEMO-001.json` with the same content as the test fixture above (the complete JSON from the test).

Write `data/background_docs/DEMO-001/doc_001.txt`:
```text
化工安全生产管理背景文件（演示用）

第一章 总则
本文件适用于化工厂日常安全生产管理工作，包括设备巡检、隐患排查、应急处置等。

第二章 设备管理
2.1 压力容器管理
压力容器定期检验周期为3年，到期前3个月应启动检验申请流程。
检验期间设备应降负荷运行或停机。

第三章 安全巡检
3.1 巡检频率
日常巡检每日一次，专项巡检每季度一次。
3.2 隐患等级
隐患分为一般隐患（low）、较大隐患（medium）、重大隐患（high）、特别重大隐患（critical）。

第四章 应急管理
发生安全事故后，应立即启动应急预案，上报管理层，并通知监管部门。
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_loader.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add longhorizon_bench/loader.py tests/test_loader.py data/
git commit -m "feat: add scenario loader with demo data"
```

---

### Task 8: LongHorizonEnv (Core Environment)

**Files:**
- Create: `longhorizon_bench/env.py`
- Create: `tests/test_env.py`

- [ ] **Step 1: Write failing tests for the environment**

```python
# tests/test_env.py
import json
import pytest
from pathlib import Path
from longhorizon_bench.schema import AgentAction, ToolCall, CheckpointResponse


@pytest.fixture
def demo_data_dir(tmp_path: Path) -> Path:
    scenario_dir = tmp_path / "scenarios" / "industrial"
    scenario_dir.mkdir(parents=True)
    bg_dir = tmp_path / "background_docs" / "DEMO-001"
    bg_dir.mkdir(parents=True)
    (bg_dir / "doc_001.txt").write_text("压力容器定期检验周期为3年。", encoding="utf-8")

    scenario = {
        "scenario_id": "DEMO-001",
        "domain": "industrial",
        "role": "化工厂安全生产管理员",
        "difficulty": 1,
        "background_docs": ["doc_001.txt"],
        "background_tokens": 50,
        "total_events": 2,
        "total_checkpoints": 1,
        "annotator": "test",
        "generation_model": "test",
        "metadata": {},
        "events": [
            {
                "event_id": "E01",
                "type": "routine_inspection",
                "input": "收到巡检报告",
                "depends_on": [],
                "node_type": "action",
                "is_checkpoint": False,
                "is_critical": True,
                "dimensions": ["domain_knowledge"],
                "scoring_rule": {
                    "tool": {"expected": "submit_inspection_report", "match": "exact"},
                    "params": {"target": {"expected": "3号车间", "match": "contains"}},
                },
                "evidence": {
                    "required_facts": [],
                    "forbidden_actions": [],
                    "acceptable_actions": [],
                },
            },
            {
                "event_id": "E02",
                "type": "routine_inspection",
                "input": "收到第二份报告",
                "depends_on": ["E01"],
                "node_type": "action",
                "is_checkpoint": False,
                "is_critical": False,
                "dimensions": ["multi_step_reasoning"],
                "scoring_rule": {
                    "tool": {"expected": "no_action", "match": "exact"},
                    "params": {"reason": {"expected": "无需操作", "match": "contains"}},
                },
                "evidence": {
                    "required_facts": [],
                    "forbidden_actions": [],
                    "acceptable_actions": [],
                },
            },
            {
                "event_id": "CP01",
                "type": "checkpoint",
                "input": None,
                "node_type": "checkpoint",
                "is_checkpoint": True,
                "checkpoint_queries": [
                    {
                        "query": "E01的内容是什么？",
                        "expected_keywords": ["巡检"],
                        "dimension": "long_term_memory",
                        "match": "keyword_coverage",
                    },
                ],
            },
        ],
    }
    (scenario_dir / "DEMO-001.json").write_text(
        json.dumps(scenario, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


def test_env_reset_returns_observation(demo_data_dir):
    from longhorizon_bench.env import LongHorizonEnv

    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    obs = env.reset()
    assert "role" in obs
    assert "current_event" in obs
    assert obs["current_event"]["event_id"] == "E01"
    assert "background_docs" in obs


def test_env_step_advances_event(demo_data_dir):
    from longhorizon_bench.env import LongHorizonEnv

    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    env.reset()

    action = AgentAction(tool_calls=[
        ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"}),
    ])
    obs, reward, done, info = env.step(action)
    assert obs["current_event"]["event_id"] == "E02"
    assert done is False


def test_env_checkpoint_step(demo_data_dir):
    from longhorizon_bench.env import LongHorizonEnv

    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    env.reset()

    # Step through E01
    env.step(AgentAction(tool_calls=[ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"})]))
    # Step through E02
    env.step(AgentAction(tool_calls=[ToolCall(tool="no_action", kwargs={"reason": "无需操作"})]))

    # Now at checkpoint — obs should indicate checkpoint
    obs = env._current_observation()
    assert obs["current_event"]["node_type"] == "checkpoint"


def test_env_done_after_last_event(demo_data_dir):
    from longhorizon_bench.env import LongHorizonEnv

    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    env.reset()

    env.step(AgentAction(tool_calls=[ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"})]))
    env.step(AgentAction(tool_calls=[ToolCall(tool="no_action", kwargs={"reason": "无需操作"})]))
    # Checkpoint
    _, _, done, _ = env.step(CheckpointResponse(answers={"E01的内容是什么？": "收到巡检报告"}))
    assert done is True


def test_env_evaluate_returns_results(demo_data_dir):
    from longhorizon_bench.env import LongHorizonEnv

    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    env.reset()

    env.step(AgentAction(tool_calls=[ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"})]))
    env.step(AgentAction(tool_calls=[ToolCall(tool="no_action", kwargs={"reason": "无需操作"})]))
    env.step(CheckpointResponse(answers={"E01的内容是什么？": "收到巡检报告"}))

    results = env.evaluate()
    assert "chain_score" in results
    assert "chain_pass" in results
    assert "event_scores" in results
    assert "dimension_scores" in results
    assert "consistency_violations" in results


def test_env_rolling_window_limits_history(demo_data_dir):
    from longhorizon_bench.env import LongHorizonEnv

    env = LongHorizonEnv("industrial/DEMO-001", mode="rolling_window", window_size=1, data_dir=demo_data_dir)
    env.reset()

    env.step(AgentAction(tool_calls=[ToolCall(tool="submit_inspection_report", kwargs={"target": "3号车间"})]))
    obs, _, _, _ = env.step(AgentAction(tool_calls=[ToolCall(tool="no_action", kwargs={"reason": "无需操作"})]))

    # In rolling window with size 1, only E02 should have full text in history
    full_history = [h for h in obs.get("history", []) if "input" in h and h["input"] is not None]
    summary_history = [h for h in obs.get("history", []) if h.get("summary_only")]
    assert len(full_history) <= 1


def test_env_memory_only_no_background(demo_data_dir):
    from longhorizon_bench.env import LongHorizonEnv

    env = LongHorizonEnv("industrial/DEMO-001", mode="memory_only", data_dir=demo_data_dir)
    obs = env.reset()
    assert obs.get("background_docs") is None or obs.get("background_summary") is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_env.py -v`
Expected: FAIL

- [ ] **Step 3: Implement env.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_env.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/env.py tests/test_env.py
git commit -m "feat: add LongHorizonEnv with three context modes (full/rolling/memory)"
```

---

### Task 9: Base Runner and OpenAI Runner

**Files:**
- Create: `longhorizon_bench/runners/__init__.py`
- Create: `longhorizon_bench/runners/base.py`
- Create: `longhorizon_bench/runners/openai_runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests for runners**

```python
# tests/test_runner.py
import pytest
from unittest.mock import MagicMock
from longhorizon_bench.schema import AgentAction, ToolCall, CheckpointResponse


def test_base_runner_is_abstract():
    from longhorizon_bench.runners.base import BaseRunner

    with pytest.raises(TypeError):
        BaseRunner()


def test_base_runner_subclass_must_implement_act():
    from longhorizon_bench.runners.base import BaseRunner

    class BadRunner(BaseRunner):
        pass

    with pytest.raises(TypeError):
        BadRunner()


def test_base_runner_subclass_works():
    from longhorizon_bench.runners.base import BaseRunner

    class SimpleRunner(BaseRunner):
        def act(self, observation: dict) -> AgentAction | CheckpointResponse:
            if observation["current_event"]["node_type"] == "checkpoint":
                queries = observation["current_event"].get("queries", [])
                return CheckpointResponse(answers={q: "测试回答" for q in queries})
            return AgentAction(tool_calls=[
                ToolCall(tool="no_action", kwargs={"reason": "test"}),
            ])

    runner = SimpleRunner()
    obs = {"current_event": {"event_id": "E01", "node_type": "action", "input": "test"}}
    result = runner.act(obs)
    assert isinstance(result, AgentAction)


def test_base_runner_run_scenario(demo_data_dir):
    from longhorizon_bench.runners.base import BaseRunner
    from longhorizon_bench.env import LongHorizonEnv

    class DummyRunner(BaseRunner):
        def act(self, observation: dict) -> AgentAction | CheckpointResponse:
            if observation["current_event"]["node_type"] == "checkpoint":
                queries = observation["current_event"].get("queries", [])
                return CheckpointResponse(answers={q: "巡检报告" for q in queries})
            return AgentAction(tool_calls=[
                ToolCall(tool="no_action", kwargs={"reason": "test"}),
            ])

    runner = DummyRunner()
    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    results = runner.run(env)
    assert "chain_score" in results


@pytest.fixture
def demo_data_dir(tmp_path):
    import json
    scenario_dir = tmp_path / "scenarios" / "industrial"
    scenario_dir.mkdir(parents=True)
    bg_dir = tmp_path / "background_docs" / "DEMO-001"
    bg_dir.mkdir(parents=True)
    (bg_dir / "doc_001.txt").write_text("背景文档", encoding="utf-8")

    scenario = {
        "scenario_id": "DEMO-001",
        "domain": "industrial",
        "role": "测试角色",
        "difficulty": 1,
        "background_docs": ["doc_001.txt"],
        "background_tokens": 10,
        "total_events": 1,
        "total_checkpoints": 1,
        "annotator": "test",
        "generation_model": "test",
        "metadata": {},
        "events": [
            {
                "event_id": "E01",
                "type": "test",
                "input": "测试",
                "depends_on": [],
                "node_type": "action",
                "is_checkpoint": False,
                "is_critical": False,
                "dimensions": ["domain_knowledge"],
                "scoring_rule": {
                    "tool": {"expected": "no_action", "match": "exact"},
                    "params": {},
                },
                "evidence": {
                    "required_facts": [],
                    "forbidden_actions": [],
                    "acceptable_actions": [],
                },
            },
            {
                "event_id": "CP01",
                "type": "checkpoint",
                "node_type": "checkpoint",
                "is_checkpoint": True,
                "checkpoint_queries": [
                    {
                        "query": "测试",
                        "expected_keywords": ["测试"],
                        "dimension": "long_term_memory",
                        "match": "keyword_coverage",
                    },
                ],
            },
        ],
    }
    (scenario_dir / "DEMO-001.json").write_text(
        json.dumps(scenario, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement runners**

```python
# longhorizon_bench/runners/__init__.py
"""Runner adapters for LLM APIs."""
```

```python
# longhorizon_bench/runners/base.py
"""Abstract base runner for LongHorizon-Bench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from longhorizon_bench.schema import AgentAction, CheckpointResponse
from longhorizon_bench.env import LongHorizonEnv


class BaseRunner(ABC):
    @abstractmethod
    def act(self, observation: dict[str, Any]) -> AgentAction | CheckpointResponse:
        ...

    def run(self, env: LongHorizonEnv) -> dict[str, Any]:
        obs = env.reset()
        done = False
        while not done:
            action = self.act(obs)
            obs, reward, done, info = env.step(action)
        return env.evaluate()
```

```python
# longhorizon_bench/runners/openai_runner.py
"""OpenAI-compatible runner for LongHorizon-Bench."""

from __future__ import annotations

from typing import Any

from longhorizon_bench.runners.base import BaseRunner
from longhorizon_bench.schema import AgentAction, CheckpointResponse, ToolCall


class OpenAIRunner(BaseRunner):
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: "
                    "pip install longhorizon-bench[openai]"
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def act(self, observation: dict[str, Any]) -> AgentAction | CheckpointResponse:
        event = observation["current_event"]

        if event["node_type"] == "checkpoint":
            return self._handle_checkpoint(observation)

        return self._handle_action(observation)

    def _handle_action(self, observation: dict[str, Any]) -> AgentAction:
        client = self._get_client()
        messages = self._build_messages(observation)

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            tools=self._build_tool_defs(),
            tool_choice="required",
        )

        tool_calls: list[ToolCall] = []
        for tc in response.choices[0].message.tool_calls or []:
            import json
            tool_calls.append(ToolCall(
                tool=tc.function.name,
                kwargs=json.loads(tc.function.arguments),
            ))

        return AgentAction(tool_calls=tool_calls) if tool_calls else AgentAction(
            tool_calls=[ToolCall(tool="no_action", kwargs={"reason": "no tool calls returned"})]
        )

    def _handle_checkpoint(self, observation: dict[str, Any]) -> CheckpointResponse:
        client = self._get_client()
        queries = observation["current_event"].get("queries", [])
        answers: dict[str, str] = {}

        for query in queries:
            messages = self._build_messages(observation)
            messages.append({"role": "user", "content": query})

            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )
            answers[query] = response.choices[0].message.content or ""

        return CheckpointResponse(answers=answers)

    def _build_messages(self, observation: dict[str, Any]) -> list[dict]:
        system = f"你是一名{observation['role']}。根据背景文档和历史事件，对当前事件做出决策。"

        bg = observation.get("background_docs") or observation.get("background_summary", "")
        if isinstance(bg, dict):
            bg = "\n\n".join(bg.values())

        history_text = ""
        for h in observation.get("history", []):
            if h.get("summary_only"):
                history_text += f"\n[{h['event_id']}] {h['type']}（摘要）"
            else:
                history_text += f"\n[{h['event_id']}] {h.get('type', '')}: {h.get('input', '')}"

        event = observation["current_event"]
        current = f"当前事件 [{event['event_id']}]: {event.get('input', '')}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"背景文档:\n{bg}\n\n历史事件:{history_text}\n\n{current}"},
        ]

    def _build_tool_defs(self) -> list[dict]:
        from longhorizon_bench.tools import COMMON_TOOLS, INDUSTRIAL_TOOLS

        defs: list[dict] = []
        for name, tool_def in {**COMMON_TOOLS, **INDUSTRIAL_TOOLS}.items():
            properties = {}
            for param_name, param_spec in tool_def.params.items():
                ptype = param_spec.get("type", "str")
                if ptype.startswith("enum"):
                    properties[param_name] = {
                        "type": "string",
                        "enum": param_spec.get("values", []),
                    }
                elif ptype == "bool":
                    properties[param_name] = {"type": "boolean"}
                elif ptype == "float":
                    properties[param_name] = {"type": "number"}
                elif ptype.startswith("list"):
                    properties[param_name] = {"type": "array", "items": {"type": "string"}}
                else:
                    properties[param_name] = {"type": "string"}

            defs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties.keys()),
                    },
                },
            })
        return defs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runner.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/runners/ tests/test_runner.py
git commit -m "feat: add base runner ABC and OpenAI-compatible runner"
```

---

### Task 10: Public API and Validation Script

**Files:**
- Modify: `longhorizon_bench/__init__.py`
- Create: `scripts/validate_scenario.py`

- [ ] **Step 1: Update __init__.py with public API re-exports**

```python
# longhorizon_bench/__init__.py
"""LongHorizon-Bench: Chinese long-horizon benchmark for LLM evaluation."""

__version__ = "0.1.0"

from longhorizon_bench.env import LongHorizonEnv
from longhorizon_bench.loader import load_scenario, load_background_docs
from longhorizon_bench.schema import (
    Scenario,
    ActionEvent,
    CheckpointEvent,
    AgentAction,
    CheckpointResponse,
    ToolCall,
)
```

- [ ] **Step 2: Create validation script**

```python
# scripts/validate_scenario.py
"""Validate a scenario JSON file against the Pydantic schema."""

import json
import sys
from pathlib import Path

import click


@click.command()
@click.argument("scenario_file", type=click.Path(exists=True))
def validate(scenario_file: str) -> None:
    """Validate a scenario JSON file."""
    from longhorizon_bench.schema import Scenario

    path = Path(scenario_file)
    click.echo(f"Validating {path.name}...")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    try:
        scenario = Scenario.model_validate(raw)
    except Exception as e:
        click.echo(f"FAIL: {e}", err=True)
        sys.exit(1)

    action_count = len(scenario.action_events)
    checkpoint_count = len(scenario.checkpoint_events)
    total = len(scenario.events)

    # Check declared counts
    warnings: list[str] = []
    if scenario.total_events != action_count:
        warnings.append(
            f"total_events={scenario.total_events} but found {action_count} action events"
        )
    if scenario.total_checkpoints != checkpoint_count:
        warnings.append(
            f"total_checkpoints={scenario.total_checkpoints} but found {checkpoint_count} checkpoints"
        )

    # Check event ID uniqueness
    ids = [e.event_id for e in scenario.events]
    if len(ids) != len(set(ids)):
        warnings.append("Duplicate event IDs found")

    # Check depends_on references
    id_set = set(ids)
    for event in scenario.action_events:
        for dep in event.depends_on:
            if dep not in id_set:
                warnings.append(f"Event {event.event_id} depends on unknown {dep}")

    if warnings:
        click.echo(f"PASS with {len(warnings)} warning(s):")
        for w in warnings:
            click.echo(f"  - {w}")
    else:
        click.echo(f"PASS: {scenario.scenario_id} ({action_count} actions, {checkpoint_count} checkpoints)")


if __name__ == "__main__":
    validate()
```

- [ ] **Step 3: Test the validation script against the demo scenario**

Run: `python scripts/validate_scenario.py data/scenarios/industrial/DEMO-001.json`
Expected: `PASS: DEMO-001 (...)` or PASS with warnings about counts

- [ ] **Step 4: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS (should be ~42 tests total)

- [ ] **Step 5: Commit**

```bash
git add longhorizon_bench/__init__.py scripts/validate_scenario.py
git commit -m "feat: add public API exports and scenario validation CLI"
```

---

### Task 11: Integration Test — Full End-to-End Run

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write end-to-end integration test**

```python
# tests/test_integration.py
"""End-to-end integration test: load scenario → run agent → evaluate."""

import json
import pytest
from pathlib import Path

from longhorizon_bench import LongHorizonEnv, AgentAction, CheckpointResponse, ToolCall
from longhorizon_bench.runners.base import BaseRunner


class PerfectAgent(BaseRunner):
    """Agent that gives the 'correct' answers for DEMO-001."""

    def act(self, observation):
        event = observation["current_event"]

        if event["node_type"] == "checkpoint":
            queries = event.get("queries", [])
            return CheckpointResponse(answers={
                q: "E01中提到1台压力容器接近检验周期，需要安排检验" for q in queries
            })

        if event["event_id"] == "E01":
            return AgentAction(tool_calls=[
                ToolCall(tool="submit_inspection_report", kwargs={
                    "target": "3号车间设备",
                    "findings": "压力容器接近检验周期，需要安排定期检验",
                    "risk_level": "medium",
                }),
            ])

        return AgentAction(tool_calls=[
            ToolCall(tool="no_action", kwargs={"reason": "无需操作，继续观察"}),
        ])


class BadAgent(BaseRunner):
    """Agent that always picks the wrong tool."""

    def act(self, observation):
        event = observation["current_event"]

        if event["node_type"] == "checkpoint":
            queries = event.get("queries", [])
            return CheckpointResponse(answers={q: "不记得了" for q in queries})

        return AgentAction(tool_calls=[
            ToolCall(tool="approve_work_permit", kwargs={
                "permit_type": "hot_work",
                "conditions": [],
                "approved": True,
            }),
        ])


@pytest.fixture
def demo_data_dir(tmp_path):
    scenario_dir = tmp_path / "scenarios" / "industrial"
    scenario_dir.mkdir(parents=True)
    bg_dir = tmp_path / "background_docs" / "DEMO-001"
    bg_dir.mkdir(parents=True)
    (bg_dir / "doc_001.txt").write_text(
        "压力容器定期检验周期为3年。", encoding="utf-8"
    )

    scenario = {
        "scenario_id": "DEMO-001",
        "domain": "industrial",
        "role": "化工厂安全生产管理员",
        "difficulty": 1,
        "background_docs": ["doc_001.txt"],
        "background_tokens": 50,
        "total_events": 2,
        "total_checkpoints": 1,
        "annotator": "test",
        "generation_model": "test",
        "metadata": {},
        "events": [
            {
                "event_id": "E01",
                "type": "routine_inspection",
                "input": "收到设备科提交的3号车间季度巡检报告，其中提到1台压力容器接近检验周期。",
                "depends_on": [],
                "node_type": "action",
                "is_checkpoint": False,
                "is_critical": True,
                "dimensions": ["domain_knowledge"],
                "scoring_rule": {
                    "tool": {"expected": "submit_inspection_report", "match": "exact"},
                    "params": {
                        "target": {"expected": "3号车间", "match": "contains"},
                        "findings": {"required_keywords": ["压力容器", "检验周期"], "match": "keyword_coverage"},
                        "risk_level": {"expected": ["medium", "high"], "match": "enum"},
                    },
                },
                "evidence": {
                    "required_facts": ["背景文档: 压力容器定期检验周期为3年"],
                    "forbidden_actions": ["approve_work_permit"],
                    "acceptable_actions": [{"tool": "submit_inspection_report"}],
                },
            },
            {
                "event_id": "E02",
                "type": "routine_report",
                "input": "本月安全生产月报需要编制。",
                "depends_on": ["E01"],
                "node_type": "action",
                "is_checkpoint": False,
                "is_critical": False,
                "dimensions": ["multi_step_reasoning"],
                "scoring_rule": {
                    "tool": {"expected": "no_action", "match": "exact"},
                    "params": {"reason": {"expected": "无需", "match": "contains"}},
                },
                "evidence": {
                    "required_facts": [],
                    "forbidden_actions": [],
                    "acceptable_actions": [],
                },
            },
            {
                "event_id": "CP01",
                "type": "checkpoint",
                "node_type": "checkpoint",
                "is_checkpoint": True,
                "checkpoint_queries": [
                    {
                        "query": "E01中提到的压力容器情况是什么？",
                        "expected_keywords": ["压力容器", "检验周期"],
                        "dimension": "long_term_memory",
                        "match": "keyword_coverage",
                    },
                ],
            },
        ],
    }
    (scenario_dir / "DEMO-001.json").write_text(
        json.dumps(scenario, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


def test_perfect_agent_scores_high(demo_data_dir):
    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    agent = PerfectAgent()
    results = agent.run(env)

    assert results["chain_score"] > 0.8
    assert results["chain_pass"] is True
    assert len(results["consistency_violations"]) == 0
    assert results["dimension_scores"]["long_term_memory"] == 1.0


def test_bad_agent_scores_low(demo_data_dir):
    env = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    agent = BadAgent()
    results = agent.run(env)

    assert results["chain_score"] < 0.5
    assert results["chain_pass"] is False


def test_rolling_window_mode_works(demo_data_dir):
    env = LongHorizonEnv("industrial/DEMO-001", mode="rolling_window", data_dir=demo_data_dir)
    agent = PerfectAgent()
    results = agent.run(env)

    assert results["chain_score"] > 0.8


def test_memory_only_mode_works(demo_data_dir):
    env = LongHorizonEnv("industrial/DEMO-001", mode="memory_only", data_dir=demo_data_dir)
    agent = PerfectAgent()
    results = agent.run(env)

    assert "chain_score" in results


def test_mode_comparison_shows_delta(demo_data_dir):
    """The core thesis: Mode A vs Mode B delta measures memory reliance."""
    agent = PerfectAgent()

    env_a = LongHorizonEnv("industrial/DEMO-001", mode="full_context", data_dir=demo_data_dir)
    results_a = agent.run(env_a)

    env_b = LongHorizonEnv("industrial/DEMO-001", mode="rolling_window", data_dir=demo_data_dir)
    results_b = agent.run(env_b)

    # With a perfect agent both should score well (no real memory pressure in demo)
    assert results_a["chain_score"] > 0
    assert results_b["chain_score"] > 0
```

- [ ] **Step 2: Run the integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: All 5 tests PASS

- [ ] **Step 3: Run the full test suite one final time**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS (~47 tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add end-to-end integration tests with perfect and bad agent"
```

---

### Task 12: Final Wiring — README and Package Exports

**Files:**
- Modify: `longhorizon_bench/__init__.py` (verify exports work)

- [ ] **Step 1: Verify the public API works as documented in the spec**

Run in Python:
```python
from longhorizon_bench import LongHorizonEnv, load_scenario
# These should import without error
from longhorizon_bench.schema import Scenario, AgentAction, ToolCall, CheckpointResponse
from longhorizon_bench.runners.base import BaseRunner
from longhorizon_bench.evaluation.scorer import EventScorer
from longhorizon_bench.evaluation.consistency import ConsistencyChecker
from longhorizon_bench.evaluation.metrics import compute_metrics
from longhorizon_bench.tools import build_registry
print("All imports OK")
```

Run: `python -c "from longhorizon_bench import LongHorizonEnv, load_scenario; from longhorizon_bench.tools import build_registry; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 2: Final full test run with coverage**

Run: `pytest tests/ -v --cov=longhorizon_bench --cov-report=term-missing`
Expected: All tests PASS, coverage > 80%

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: final wiring and verification"
```
