# LongHorizon-Bench 设计文档

## 1. 项目定位

**一句话定位**：面向中文的长程岗位模拟 Benchmark，通过事件链驱动，评测大模型在持续工作流中的长程记忆、多步推理与决策一致性。

**差异化**：现有 benchmark 要么偏长上下文理解（LongBench、CLongEval），要么偏 Agent 多步操作（SWE-bench、τ-bench），且几乎无中文行业垂直场景。LongHorizon-Bench 将两者结合——长文档输入 + 多步事件链决策，聚焦中文工业垂直领域。

**目标路径**：先学术发表建立影响力，再工具化为可集成的评测 SDK。

## 2. 核心概念

- **Scenario（场景）**：一个具体的岗位角色 + 行业背景，附带大量背景文档（5-10万字）
- **Event Chain（事件链）**：该角色在数月工作中经历的一系列事件（15-25个），事件驱动推进，非按天模拟
- **Checkpoint（检查点）**：每隔约5个事件设置快照检查点，包含回溯题，专门度量记忆保持和一致性
- **Decision Node（决策节点）**：每个事件是一个决策节点，模型通过 tool-calling 产出决策

## 3. 子领域

三个子领域，工业优先：

| 子领域 | 优先级 | 第一版规模 |
|--------|--------|-----------|
| 工业（安全生产、质量管理、设备运维、环保合规） | P0 | 10条链，~195个事件 |
| 人事（招聘、绩效管理、劳动合规） | P1 | 5条链，~100个事件 |
| 架构（系统设计、技术选型、架构评审） | P2 | 5条链，~100个事件 |

## 4. 评测维度

按优先级排序：

1. **长程记忆**（核心）— 能否在后期事件中准确引用早期信息
2. **多步推理**（核心）— 面对复杂事件能否分步推导出合理决策
3. **一致性**（核心）— 跨事件的决策是否前后连贯
4. **优先级判断**（辅助）— 多事件并发时的轻重缓急
5. **领域知识运用**（辅助）— 行业规范/法规的正确应用
6. **信息整合**（辅助）— 从分散事件中拼出完整图景

## 5. 数据构造流程

三阶段混合构造：真实骨架 → LLM 填充 → 人工校验。

### 5.1 阶段一：骨架提取（真实数据）

数据源按可靠性分级：

**第一优先级：HuggingFace 数据集**

| 领域 | 数据集 | HF 路径 | 用途 |
|------|--------|---------|------|
| 工业 | IndustryCorpus2 (石化/采矿/消防安全/制造业) | `BAAI/IndustryCorpus2` | 筛选中文工业安全文档，提取事件骨架 |
| 人事 | DISC-Law-SFT | `ShengbinYue/DISC-Law-SFT` | 劳动法相关 QA，提取劳动争议事件模式 |
| 人事 | 中国法律法规全文 | `twang2218/chinese-law-and-regulations` | 劳动法/社保法规原文，作为背景文档 |
| 人事 | CAIL2018 | `china-ai-law-challenge/cail2018` | 217万条案件，参考事件链结构 |
| 架构 | IndustryCorpus2 (计算机通信) | `BAAI/IndustryCorpus2` | 筛选中文技术文档 |
| 架构 | GitHub Issues | `bigcode/the-stack-github-issues` | 筛选中文 issue/PR 对话链 |
| 通用 | LongBench | `THUDM/LongBench` | 参考评测格式和中文长文本样本 |
| 通用 | 中文维基百科 | `fjcanyue/wikipedia-zh-cn` | 补充领域知识背景材料 |

**第二优先级：政府公开数据**

| 领域 | 数据源 | 用途 |
|------|--------|------|
| 工业 | 应急管理部事故调查报告 | 提取真实事故事件时间线 |
| 工业 | 国家标准全文公开系统 (openstd.samr.gov.cn) | 强制性工业标准原文 |
| 人事 | 人社部公开政策文件 | HR 政策变更事件 |

**骨架格式**：

```yaml
scenario_id: IND-001
domain: industrial
role: 化工厂安全生产管理员
time_span: 6个月
difficulty: 3
events:
  - id: E01
    type: routine_inspection
    trigger: 设备老化报告
    depends_on: []
  - id: E02
    type: policy_change
    trigger: 新安全生产法规发布
    depends_on: [E01]
  - id: E03
    type: incident
    trigger: 车间泄漏事故
    depends_on: [E01, E02]
checkpoints: [E05, E10, E15, E20]
```

### 5.2 阶段二：LLM 填充

基于骨架用大模型生成：
- 背景文档（5-10万字，基于真实标准/法规改写）
- 事件详情（每个事件节点的具体信息输入）
- 标准答案（expected_action + expected_output）
- 检查点回溯题

质量控制 prompt 策略：
- 要求生成内容引用背景文档中的具体条款/数据
- 事件间必须有明确的因果依赖
- 干扰信息必须是"看起来合理但违反了之前某个事件/文档中的约束"

### 5.3 阶段三：人工校验

- 每个子领域 2-3 位从业者审核事件链的合理性和专业性
- 不同审核员独立评分，不一致的样本重新修改
- 标准答案确保无歧义

## 6. 评测协议

### 6.1 交互协议：Gym-Style 环境循环

对齐 τ-bench 范式：`reset() → step(action) → (observation, reward, done)`

```python
env = LongHorizonEnv(scenario_id="IND-001")
observation = env.reset()

for step in range(max_steps):
    action = agent(observation)
    observation, reward, done = env.step(action)
    if done:
        break

score = env.evaluate()
```

### 6.2 数据集格式：JSONL + Pydantic Schema

```jsonl
{
  "scenario_id": "IND-001",
  "domain": "industrial",
  "role": "化工厂安全生产管理员",
  "difficulty": 2,
  "background_docs": ["doc_001.txt", "doc_002.txt"],
  "background_tokens": 85000,
  "events": [
    {
      "event_id": "E01",
      "type": "routine_inspection",
      "input": "收到设备科提交的3号车间季度巡检报告...",
      "depends_on": [],
      "node_type": "action",
      "expected_action": {"tool": "submit_inspection_report", "kwargs": {}},
      "expected_output": "...",
      "is_checkpoint": false,
      "dimensions": ["domain_knowledge"]
    },
    {
      "event_id": "E05",
      "type": "checkpoint",
      "input": null,
      "node_type": "checkpoint",
      "checkpoint_queries": [
        {"query": "E02中发现的隐患当前状态？", "expected": "...", "dimension": "long_term_memory"},
        {"query": "你在E03和E04中的处置是否一致？", "expected": "...", "dimension": "consistency"}
      ],
      "is_checkpoint": true
    }
  ],
  "total_events": 20,
  "total_checkpoints": 4,
  "annotator": "expert_001",
  "metadata": {"source": "BAAI/IndustryCorpus2", "time_span": "6months"}
}
```

### 6.3 Agent 动作空间

模型通过 tool-calling 产出决策，每个领域定义独立工具集：

```python
# 工业领域工具集
INDUSTRIAL_TOOLS = {
    "submit_inspection_report":    {"params": ["target", "findings", "risk_level"]},
    "update_safety_ledger":        {"params": ["item_id", "status", "remarks"]},
    "file_incident_report":        {"params": ["incident_type", "severity", "description"]},
    "issue_rectification_order":   {"params": ["target_dept", "issues", "deadline"]},
    "request_equipment_shutdown":  {"params": ["equipment_id", "reason", "duration"]},
    "approve_work_permit":         {"params": ["permit_type", "conditions", "approved"]},
    "escalate_to_management":      {"params": ["issue_summary", "urgency", "recommendation"]},
    "notify_regulatory_body":      {"params": ["authority", "event_type", "details"]},
    "respond_to_query":            {"params": ["recipient", "content"]},
    "schedule_training":           {"params": ["topic", "participants", "date"]},
    "allocate_budget":             {"params": ["item", "amount", "justification"]},
    "assign_personnel":            {"params": ["task", "assignee", "priority"]},
}
```

### 6.4 Contamination 防护

- 背景文档基于真实数据改写，非原文照搬
- 事件骨架来自真实案例但细节由 LLM 填充，保证唯一性
- Test set 答案不公开，通过提交平台评测（对齐 GAIA 模式）

## 7. 评分体系

### 7.1 主指标：Pass^k

对齐 τ-bench，每条事件链运行 k 次（k=5）：
- **事件级**：该事件的 action 和 output 是否完全正确（二元 0/1）
- **链级**：所有事件全部正确才算 pass（严格二元）
- **Pass^k**：k 次运行中至少 1 次全 pass 的概率

### 7.2 辅助指标（按维度分层报告）

| 维度 | 指标 | 评分方式 |
|------|------|---------|
| 长程记忆 | Checkpoint 回溯题准确率 | 自动：关键信息精确匹配 |
| 多步推理 | Action 序列正确率 | 自动：与 expected_action 比对 |
| 一致性 | 跨事件决策矛盾率 | 规则校验：检测逻辑冲突 |
| 领域知识 | 工具选择准确率 | 自动：tool name 精确匹配 |
| 信息整合 | 引用事件覆盖率 | 自动：cited events 与 depends_on 比对 |

LLM-as-Judge 仅在 `respond_to_query`（自由文本回复）场景作为补充，非主评分方式。

## 8. 工业领域详细设计（第一版核心）

### 8.1 角色设定

| 角色 | 典型场景 | 背景文档来源 | 难度 |
|------|---------|-------------|------|
| 化工厂安全管理员 | 危化品管理、应急预案、事故处置 | GB 30871 + IndustryCorpus2(petrochemical) | L3 |
| 制造企业质量工程师 | 质量体系运行、不合格品处理、供应商审核 | ISO 9001 中文版 + IndustryCorpus2(manufacturing) | L2 |
| 矿山安全工程师 | 矿山安全监测、隐患排查、应急救援 | 煤矿安全规程 + IndustryCorpus2(mining) | L3 |
| 设备运维主管 | 预防性维护、故障诊断、备件管理 | 设备管理标准 + IndustryCorpus2(manufacturing) | L2 |
| 环保合规专员 | 排放监测、环评跟踪、政策应对 | 环保法规 + IndustryCorpus2(fire_safety) | L1 |

难度分级（对齐 GAIA 三级体系）：
- **Level 1**：3-5步推理，事件依赖简单
- **Level 2**：8-12步推理，有交叉依赖
- **Level 3**：15+步推理，多事件并发 + 信息冲突

### 8.2 事件类型体系

```
常规类：日常巡检、定期报告、培训通知、会议纪要
变更类：法规更新、标准修订、组织架构调整、设备升级
异常类：设备故障、指标超标、员工违规、供应商问题
危机类：安全事故、环保事件、监管检查、媒体曝光
决策类：预算分配、方案选型、人员调配、优先级排序
```

事件链构造规则：
- 每条链 15-25 个事件，至少包含 3 种事件类型
- 前 5 个事件以常规类为主（建立基线上下文）
- 中段引入变更和异常（制造复杂度）
- 后段出现危机或重大决策（考验长程记忆和一致性）
- 事件间必须有因果依赖图，不允许孤立事件

### 8.3 第一版规模

| 角色 | 链数 | 难度 | 事件数/链 | 检查点/链 |
|------|------|------|----------|----------|
| 化工厂安全管理员 | 3 | L1×1, L2×1, L3×1 | 15/20/25 | 3/4/5 |
| 质量工程师 | 2 | L1×1, L2×1 | 15/20 | 3/4 |
| 矿山安全工程师 | 2 | L2×1, L3×1 | 20/25 | 4/5 |
| 设备运维主管 | 2 | L1×1, L2×1 | 15/20 | 3/4 |
| 环保合规专员 | 1 | L1×1 | 15 | 3 |

总计：10条链，约195个事件节点，约39个检查点。

## 9. 技术架构

### 9.1 项目仓库结构

```
long-horizon-bench/
├── README.md
├── LICENSE                          # Apache 2.0
├── pyproject.toml                   # pip install longhorizon-bench
├── data/
│   ├── scenarios/                   # 场景定义（JSONL）
│   │   ├── industrial/
│   │   ├── hr/
│   │   └── architecture/
│   ├── background_docs/             # 背景文档（按 scenario_id 组织）
│   └── skeletons/                   # 事件骨架（构造中间产物）
├── longhorizon_bench/
│   ├── __init__.py
│   ├── env.py                       # LongHorizonEnv（Gym-style 核心）
│   ├── schema.py                    # Pydantic 数据模型
│   ├── tools/                       # 各领域工具集定义
│   │   ├── industrial.py
│   │   ├── hr.py
│   │   └── architecture.py
│   ├── evaluation/
│   │   ├── scorer.py                # 主评分逻辑（二元 pass/fail）
│   │   ├── metrics.py               # Pass^k、维度指标计算
│   │   └── consistency_checker.py   # 一致性规则校验
│   └── runners/
│       ├── base_runner.py           # Agent 运行器基类
│       └── openai_runner.py         # OpenAI API 适配示例
├── scripts/
│   ├── build_skeleton.py            # 阶段一：骨架提取
│   ├── fill_with_llm.py             # 阶段二：LLM 填充
│   ├── validate_scenario.py         # 阶段三：数据校验
│   └── run_baseline.py              # 跑 baseline 实验
├── docker/
│   └── Dockerfile                   # 一键复现环境
└── docs/
    └── paper/                       # 论文相关
```

### 9.2 核心运行流程

```python
from longhorizon_bench import LongHorizonEnv, load_scenario

scenario = load_scenario("industrial/IND-001")
env = LongHorizonEnv(scenario)

obs = env.reset()
# obs = {
#   "role": "化工厂安全生产管理员",
#   "background_docs": "...(5-10万字)...",
#   "current_event": {事件E01},
#   "available_tools": [...],
#   "history": []
# }

while not done:
    action = your_agent(obs)
    obs, reward, done, info = env.step(action)

results = env.evaluate()
# results = {
#   "pass": True/False,
#   "event_scores": [...],
#   "dimension_scores": {"long_term_memory": 0.85, ...},
#   "checkpoint_details": [...]
# }
```

## 10. 发布与复现

- 数据格式：JSONL，HuggingFace Datasets 托管
- 评测框架：Python 包，`pip install longhorizon-bench`
- Docker 化环境，一键复现
- 许可证：Apache 2.0
- Leaderboard：HuggingFace Spaces
- Test set 答案不公开，通过提交平台评测

### Baseline 模型

| 模型 | 来源 | 上下文窗口 |
|------|------|-----------|
| GPT-4o | OpenAI | 128K |
| Claude 3.5 Sonnet / Claude 4 | Anthropic | 200K |
| Qwen2.5-72B / Qwen3 | 阿里 | 128K |
| DeepSeek-V3 | DeepSeek | 128K |
| GLM-4 | 智谱 | 128K |
| Yi-Large | 零一万物 | 200K |

## 11. 论文结构

1. Introduction — 长程岗位模拟的动机，与现有 benchmark 差异
2. Related Work — 长上下文 vs Agent benchmark，中文评测现状
3. Benchmark Design — 场景定义、事件链构造、交互协议、工具集
4. Data Construction — 三阶段混合流程、质量控制、统计分析
5. Evaluation Protocol — Pass^k 主指标、维度辅助指标、评分实现
6. Experiments — Baseline 结果（中外模型对比）
7. Analysis — 记忆衰减曲线、一致性漂移、难度/领域/链长度消融
8. Conclusion
