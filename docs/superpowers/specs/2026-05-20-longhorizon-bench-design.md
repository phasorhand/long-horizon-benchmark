# LongHorizon-Bench 设计文档

## 1. 项目定位

**一句话定位**：面向中文的长程岗位模拟 Benchmark，通过事件链驱动，评测大模型在持续工作流中的长程记忆、多步推理与决策一致性。

**差异化**：现有 benchmark 要么偏长上下文理解（LongBench、CLongEval），要么偏 Agent 多步操作（SWE-bench、τ-bench），且几乎无中文行业垂直场景。LongHorizon-Bench 将两者结合——长文档输入 + 多步事件链决策，聚焦中文工业垂直领域。

**目标路径**：先学术发表建立影响力，再工具化为可集成的评测 SDK。

## 2. 核心概念

- **Scenario（场景）**：一个具体的岗位角色 + 行业背景，附带大量背景文档（5-10万字）
- **Event Chain（事件链）**：该角色在数月工作中经历的一系列事件（15-25个），事件驱动推进，非按天模拟
- **Checkpoint（检查点）**：在事件链中非固定间隔处设置显式回溯题（间隔 3-7 个事件，随机化），专门度量记忆保持和一致性。此外，部分普通事件隐含对早期事件的依赖（隐式检查点），模型不会意识到正在被测试记忆
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

数据源按可靠性分级。每个来源标注许可证和使用方式，确保合法再发布。

**第一优先级：HuggingFace 数据集**

| 领域 | 数据集 | HF 路径 | 许可证 | 使用方式 | 用途 |
|------|--------|---------|--------|---------|------|
| 工业 | IndustryCorpus2 (石化/采矿/消防/制造) | `BAAI/IndustryCorpus2` | Apache 2.0 | 筛选后改写，不直接引用原文 | 提取事件骨架 |
| 人事 | DISC-Law-SFT | `ShengbinYue/DISC-Law-SFT` | Apache 2.0 | 提取事件模式，改写内容 | 劳动争议事件模式 |
| 人事 | 中国法律法规全文 | `twang2218/chinese-law-and-regulations` | MIT | 法规本身为公共领域，可直接引用 | 背景文档素材 |
| 人事 | CAIL2018 | `china-ai-law-challenge/cail2018` | CC-BY-4.0 | 引用需标注来源 | 参考事件链结构 |
| 架构 | IndustryCorpus2 (计算机通信) | `BAAI/IndustryCorpus2` | Apache 2.0 | 同上 | 中文技术文档 |
| 架构 | GitHub Issues | `bigcode/the-stack-github-issues` | 混合(per-repo) | 仅提取骨架模式，不引用原始用户内容 | issue/PR 对话链模式 |
| 通用 | LongBench | `THUDM/LongBench` | MIT | 参考格式，不包含其数据 | 参考评测格式 |
| 通用 | 中文维基百科 | `fjcanyue/wikipedia-zh-cn` | CC-BY-SA-3.0 | 改写后用于背景材料，标注来源 | 领域知识补充 |

**第二优先级：政府公开数据**

| 领域 | 数据源 | 法律依据 | 使用方式 | 用途 |
|------|--------|---------|---------|------|
| 工业 | 应急管理部事故调查报告 | 政府信息公开条例 | 提取事件时间线骨架，改写细节 | 真实事故事件线 |
| 工业 | 国家标准全文公开系统 | 强制性国标免费公开 | 引用标准编号和条款要点，不全文复制 | 工业标准引用 |
| 人事 | 人社部公开政策文件 | 政府信息公开条例 | 提取政策要点，改写为场景素材 | HR 政策变更事件 |

**许可证策略**：代码部分 Apache 2.0；数据集单独采用 CC-BY-SA-4.0，附 DATA_LICENSE 文件详细说明每个来源的许可证和使用方式。不直接再发布任何受限原始数据，所有背景文档和事件内容均为基于真实数据改写的合成材料。

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
- 标准答案（expected_action + scoring_rule）
- 检查点回溯题

**生成模型隔离**：填充阶段使用的 LLM 必须与 baseline 被测模型不同，且记录模型 ID 和版本。避免 benchmark 隐式测量"是否符合生成模型偏好"。

**每个事件必须附带 evidence map**：
```yaml
event_id: E14
evidence:
  required_facts:        # 正确决策必须基于的事实
    - "E01: 3号车间2台压力容器接近检验周期"
    - "E03: 新条例第42条要求储罐间距≥15m"
  forbidden_actions:     # 明确的错误决策
    - "approve_work_permit without shutdown"
    - "respond_to_query without escalation"
  acceptable_actions:    # 可接受的正确决策（允许多种）
    - ["file_incident_report", "request_equipment_shutdown", "escalate_to_management"]
    - ["file_incident_report", "request_equipment_shutdown", "notify_regulatory_body"]
```

质量控制 prompt 策略：
- 要求生成内容引用背景文档中的具体条款/数据
- 事件间必须有明确的因果依赖
- 每个事件的 evidence map 必须可追溯到背景文档或前序事件的具体位置

### 5.3 阶段三：人工校验

专家审核的对象是 evidence map 和评分规则，而不仅是"这条链看起来合理"：
- 每个子领域 2-3 位从业者，独立审核 evidence map 的正确性和完备性
- 审核标准：required_facts 是否充分？forbidden_actions 是否真的错？acceptable_actions 是否遗漏了合理选项？
- 不同审核员独立评分，Cohen's Kappa ≥ 0.7 的样本通过，低于阈值的重新修改
- 记录 inter-annotator agreement 用于论文报告

## 6. 评测协议

### 6.1 环境状态模型：Fixed Replay

v1 采用 **fixed event replay**：事件序列预先确定，不因模型的 action 产生分支。模型的错误 action 不会改变后续事件输入，但会被记录并影响评分。

这是一个明确的设计选择：
- **为什么不做 stateful simulation**：状态分支会导致构造成本指数级增长，且不同模型走不同路径后无法直接比较。
- **为什么这仍然有效**：核心评测的是决策质量（选对工具、传对参数、引用对事实），而非探索能力。τ-bench 的 retail/airline 场景同样是预设任务序列。
- **论文中需明确声明**：本 benchmark 评测的是"给定工作流的决策能力"，而非"自主探索能力"。

### 6.2 上下文协议：三种模式

这是本 benchmark 区别于长上下文 benchmark 的核心设计。评测提供三种上下文模式，作为论文的对照实验条件：

**Mode A — Full Context（对照组）**：
reset 时给完整背景文档 + 每步累积所有历史事件。等价于长上下文测试，用于建立 baseline。

**Mode B — Rolling Window（主实验）**：
reset 时给完整背景文档，但历史事件只保留最近 N 个（N=5）的原文。更早的事件仅提供一行摘要（事件 ID + 类型 + 时间）。模型若需引用早期事件细节，必须依赖自身记忆。

**Mode C — Memory Only（高难度）**：
reset 时给背景文档摘要（而非全文），历史事件只提供 ID 列表，无任何内容。模型必须完全依赖内部记忆。增加 `retrieve_policy` 和 `inspect_history` 工具，允许模型主动查询（但查询次数有限，计入评分）。

**论文核心分析**：Mode A vs Mode B 的 delta 直接度量"长程记忆能力"——如果模型在 Mode A 表现好但 Mode B 大幅下降，说明它依赖上下文读取而非真正记忆。

### 6.3 交互协议：Gym-Style 环境循环

对齐 τ-bench 范式：`reset() → step(action) → (observation, reward, done, info)`

```python
env = LongHorizonEnv(scenario_id="IND-001", mode="rolling_window")
obs = env.reset()

done = False
while not done:
    action = agent(obs)
    obs, reward, done, info = env.step(action)

results = env.evaluate()
```

### 6.4 数据集格式：JSON per scenario + Pydantic Schema

每个 scenario 一个 JSON 文件（非 JSONL，因为单个 scenario 包含完整事件链，不适合逐行拆分）。HuggingFace 托管时自动转为 Parquet。

```json
{
  "scenario_id": "IND-001",
  "domain": "industrial",
  "role": "化工厂安全生产管理员",
  "difficulty": 2,
  "background_docs": ["doc_001.txt", "doc_002.txt"],
  "background_tokens": 85000,
  "total_events": 20,
  "total_checkpoints": 4,
  "annotator": "expert_001",
  "annotator_agreement": 0.82,
  "generation_model": "claude-sonnet-4-20250514",
  "metadata": {"source": "BAAI/IndustryCorpus2", "time_span": "6months"},
  "events": [
    {
      "event_id": "E01",
      "type": "routine_inspection",
      "input": "收到设备科提交的3号车间季度巡检报告...",
      "depends_on": [],
      "node_type": "action",
      "is_checkpoint": false,
      "is_critical": true,
      "dimensions": ["domain_knowledge"],
      "scoring_rule": {
        "tool": {"expected": "submit_inspection_report", "match": "exact"},
        "params": {
          "target": {"expected": "3号车间", "match": "contains"},
          "findings": {"required_keywords": ["压力容器", "检验周期"], "match": "keyword_coverage"},
          "risk_level": {"expected": ["medium", "high"], "match": "enum"}
        }
      },
      "evidence": {
        "required_facts": ["背景文档-第3章: 压力容器定期检验周期为3年"],
        "forbidden_actions": ["approve_work_permit"],
        "acceptable_actions": [
          {"tool": "submit_inspection_report"},
          {"tool": "update_safety_ledger"}
        ]
      }
    },
    {
      "event_id": "E05",
      "type": "checkpoint",
      "input": null,
      "node_type": "checkpoint",
      "is_checkpoint": true,
      "checkpoint_queries": [
        {
          "query": "E02中发现的隐患当前状态？",
          "expected_keywords": ["整改中", "未完成", "3号车间"],
          "dimension": "long_term_memory",
          "match": "keyword_coverage"
        },
        {
          "query": "你在E03和E04中的处置是否一致？如有差异请说明原因。",
          "consistency_rule": "risk_level_monotonic",
          "dimension": "consistency"
        }
      ]
    }
  ]
}
```

### 6.5 Agent 动作空间

模型通过 tool-calling 产出决策。每个事件允许返回 action list（多个工具调用），对齐真实工作中一个事件触发多个动作的场景。

每个领域定义独立工具集，所有领域共享通用工具：

```python
# 通用工具（所有领域共享）
COMMON_TOOLS = {
    "retrieve_policy":       {"params": ["query"], "note": "查询背景文档中的相关条款，Mode C 下有调用次数限制"},
    "inspect_history":       {"params": ["event_id"], "note": "查看指定历史事件的详情，Mode C 下有调用次数限制"},
    "request_clarification": {"params": ["question"], "note": "向上级/同事请求补充信息"},
    "no_action":             {"params": ["reason"], "note": "判断当前事件无需采取行动，需说明理由"},
    "respond_to_query":      {"params": ["recipient", "content"]},
}

# 工业领域专用工具
INDUSTRIAL_TOOLS = {
    "submit_inspection_report":    {"params": {"target": "str", "findings": "str", "risk_level": "enum[low,medium,high,critical]"}},
    "update_safety_ledger":        {"params": {"item_id": "str", "status": "enum[open,in_progress,resolved,overdue]", "remarks": "str"}},
    "file_incident_report":        {"params": {"incident_type": "enum[injury,leak,fire,equipment_failure,violation]", "severity": "enum[minor,major,critical]", "description": "str"}},
    "issue_rectification_order":   {"params": {"target_dept": "str", "issues": "list[str]", "deadline": "date"}},
    "request_equipment_shutdown":  {"params": {"equipment_id": "str", "reason": "str", "duration": "str"}},
    "approve_work_permit":         {"params": {"permit_type": "enum[hot_work,confined_space,height,electrical]", "conditions": "list[str]", "approved": "bool"}},
    "escalate_to_management":      {"params": {"issue_summary": "str", "urgency": "enum[routine,urgent,emergency]", "recommendation": "str"}},
    "notify_regulatory_body":      {"params": {"authority": "str", "event_type": "str", "details": "str"}},
    "schedule_training":           {"params": {"topic": "str", "participants": "list[str]", "date": "date"}},
    "allocate_budget":             {"params": {"item": "str", "amount": "float", "justification": "str"}},
    "assign_personnel":            {"params": {"task": "str", "assignee": "str", "priority": "enum[low,medium,high]"}},
}
```

**参数类型约束**：枚举参数用 exact match，日期参数允许 ±3 天容差，数值参数允许 ±10% 容差，字符串参数用 keyword_coverage 匹配。每个事件的 `scoring_rule` 可覆盖默认规则。
```

### 6.6 Contamination 防护

- 背景文档基于真实数据改写，非原文照搬
- 事件骨架来自真实案例但细节由 LLM 填充，保证唯一性
- Test set 答案不公开，通过提交平台评测（对齐 GAIA 模式）
- 记录构造阶段使用的 LLM 模型 ID、构造日期和数据版本
- 发布前对 test set 做 n-gram overlap 检查（与常见训练语料比对），确保无高度重叠样本
- dev/test split：dev set 公开（含答案，用于开发调试），test set 延迟发布答案

## 7. 评分体系

### 7.1 多层指标体系

采用多层指标而非单一 Pass^k，解决区分度问题：

**L1 — 事件级指标（最细粒度）**：
- **Event Accuracy**：单事件得分。tool 选择 + 参数匹配按 scoring_rule 打分，满分 1.0。
- **Critical Event Accuracy**：仅统计标记为 `is_critical: true` 的关键事件。这些事件的错误会导致后续事件链逻辑崩塌。

**L2 — 维度级指标（核心分析维度）**：

| 维度 | 指标 | 评分方式 |
|------|------|---------|
| 长程记忆 | Checkpoint 回溯题得分 | keyword_coverage：required keywords 命中比例 |
| 多步推理 | Action 序列得分 | 加权平均：tool match × 0.4 + param match × 0.6 |
| 一致性 | 一致性违规率 | 规则校验（见下方具体规则） |
| 领域知识 | 工具选择准确率 | exact match on tool name |
| 信息整合 | 证据引用覆盖率 | cited events ∩ required_facts / required_facts |

**L3 — 链级指标（排序用）**：
- **Chain Score**：链上所有事件 Event Accuracy 的加权平均（critical 事件权重 ×2）
- **Chain Pass (strict)**：所有事件 tool 选择正确 且 无一致性违规
- **Pass^k**：k=5 次运行中 Chain Pass 的比例。作为 robustness 指标报告，不作为唯一排序依据

**Leaderboard 排序主指标**：Chain Score（加权平均），按 Mode B (Rolling Window) 的结果排序。

### 7.2 一致性规则（具体定义）

一致性校验不是笼统的"检测逻辑冲突"，而是以下可编程规则：

| 规则 ID | 名称 | 检测逻辑 | 适用领域 |
|---------|------|---------|---------|
| C01 | 状态矛盾 | 同一 item_id 在 update_safety_ledger 中状态倒退（如 resolved → open 且无新事件触发） | 工业 |
| C02 | 风险等级单调性 | 事故发生后 risk_level 不应低于事故前同类事件的评级 | 工业 |
| C03 | 审批前置条件 | approve_work_permit 前必须有对应的 inspection_report | 工业 |
| C04 | 整改闭环 | issue_rectification_order 后必须在后续事件中有 update_safety_ledger(resolved) | 工业 |
| C05 | 责任人一致 | 同一任务的 assignee 不应在无说明的情况下变更 | 通用 |
| C06 | 时间线合理 | deadline 不应早于当前事件时间 | 通用 |

每条链预定义适用的规则子集，一致性违规率 = 违规次数 / 适用规则检查次数。

### 7.3 评分方式

所有评分均为自动化，不依赖人工：
- **工具选择**：exact match
- **枚举参数**：exact match
- **数值参数**：±10% 容差
- **日期参数**：±3 天容差
- **字符串参数**：keyword_coverage（required_keywords 命中比例）
- **respond_to_query**：LLM-as-Judge 作为补充（使用与构造模型不同的第三方模型）

每个事件的 `scoring_rule` 字段覆盖默认规则，允许事件级定制。

## 8. 工业领域详细设计（第一版核心）

### 8.1 角色设定

| 角色 | 典型场景 | 背景文档来源 | 难度 |
|------|---------|-------------|------|
| 化工厂安全管理员 | 危化品管理、应急预案、事故处置 | GB 30871 + IndustryCorpus2(petrochemical) | L3 |
| 制造企业质量工程师 | 质量体系运行、不合格品处理、供应商审核 | ISO 9001 中文版 + IndustryCorpus2(manufacturing) | L2 |
| 矿山安全工程师 | 矿山安全监测、隐患排查、应急救援 | 煤矿安全规程 + IndustryCorpus2(mining) | L3 |
| 设备运维主管 | 预防性维护、故障诊断、备件管理 | 设备管理标准 + IndustryCorpus2(manufacturing) | L2 |
| 环保合规专员 | 排放监测、环评跟踪、政策应对 | 环保法规 + IndustryCorpus2(other_manufacturing) | L1 |

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

总计：10条链，约195个决策事件节点 + 约39个显式检查点（检查点不计入决策事件数）。

**规模定位**：v1 定位为 pilot benchmark，论文中所有模型比较报告 95% bootstrap 置信区间。后续版本可按角色维度扩展链数。

## 9. 技术架构

### 9.1 项目仓库结构

```
long-horizon-bench/
├── README.md
├── LICENSE                          # Apache 2.0（仅覆盖代码）
├── DATA_LICENSE                     # CC-BY-SA-4.0 + 各来源许可证明细
├── pyproject.toml                   # pip install longhorizon-bench
├── data/
│   ├── scenarios/                   # 场景定义（JSON per scenario）
│   │   ├── industrial/
│   │   ├── hr/
│   │   └── architecture/
│   ├── background_docs/             # 背景文档（按 scenario_id 组织）
│   └── skeletons/                   # 事件骨架（构造中间产物）
├── longhorizon_bench/
│   ├── __init__.py
│   ├── env.py                       # LongHorizonEnv（Gym-style 核心）
│   ├── schema.py                    # Pydantic 数据模型
│   ├── tools/
│   │   ├── common.py                # 通用工具（retrieve_policy, inspect_history, ...）
│   │   ├── industrial.py
│   │   ├── hr.py
│   │   └── architecture.py
│   ├── evaluation/
│   │   ├── scorer.py                # 事件级评分（按 scoring_rule）
│   │   ├── metrics.py               # 多层指标计算
│   │   └── consistency_checker.py   # 一致性规则引擎（C01-C06）
│   └── runners/
│       ├── base_runner.py           # Agent 运行器基类
│       └── openai_runner.py         # OpenAI API 适配示例
├── scripts/
│   ├── build_skeleton.py            # 阶段一：骨架提取
│   ├── fill_with_llm.py             # 阶段二：LLM 填充
│   ├── validate_scenario.py         # 阶段三：数据校验
│   ├── contamination_check.py       # n-gram overlap 检查
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
# obs 内容取决于 mode：
# Mode A: 完整背景文档 + 当前事件 + 全部历史
# Mode B: 完整背景文档 + 当前事件 + 最近5个事件原文 + 更早事件摘要
# Mode C: 背景文档摘要 + 当前事件 + 历史事件 ID 列表

done = False
while not done:
    action = your_agent(obs)  # 返回 action list（一个或多个工具调用）
    obs, reward, done, info = env.step(action)

results = env.evaluate()
# results = {
#   "chain_score": 0.82,
#   "chain_pass": False,
#   "event_scores": [...],
#   "dimension_scores": {"long_term_memory": 0.85, ...},
#   "consistency_violations": [...],
#   "checkpoint_details": [...]
# }
```

## 10. 发布与复现

- 数据格式：JSON per scenario，HuggingFace Datasets 托管（自动转 Parquet）
- 评测框架：Python 包，`pip install longhorizon-bench`
- Docker 化环境，一键复现
- 许可证：代码 Apache 2.0，数据 CC-BY-SA-4.0（附 DATA_LICENSE 说明每个来源）
- Leaderboard：HuggingFace Spaces
- dev set 公开含答案，test set 答案不公开，通过提交平台评测

### Baseline 模型

实验时记录具体 model ID、API 调用日期、temperature、tool-call 格式等参数。以下为计划覆盖的模型家族（具体版本以实验时最新可用为准）：

- OpenAI GPT-4o 系列
- Anthropic Claude 系列
- 阿里 Qwen 系列
- DeepSeek 系列
- 智谱 GLM 系列
- 零一万物 Yi 系列

重点对比中外模型在中文工业场景下的表现差异。

## 11. 论文结构

1. Introduction — 长程岗位模拟的动机，与现有 benchmark 差异
2. Related Work — 长上下文 vs Agent benchmark，中文评测现状
3. Benchmark Design — 场景定义、事件链构造、交互协议、工具集
4. Data Construction — 三阶段混合流程、质量控制、统计分析
5. Evaluation Protocol — Pass^k 主指标、维度辅助指标、评分实现
6. Experiments — Baseline 结果（中外模型对比）
7. Analysis — 记忆衰减曲线、一致性漂移、难度/领域/链长度消融
8. Conclusion
