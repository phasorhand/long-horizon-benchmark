# LongHorizon-Bench 数据构建 Pipeline 设计文档

## 1. 目标

构建全自动数据生成 pipeline，从公开数据集出发，生成符合 `schema.py` 定义的完整工业域场景数据。人工仅在最终产出审核阶段介入。

**第一批规模**：1 条种子链打磨质量 → 扩到 3 条链（2 安全生产 + 1 环保合规），~50-75 个决策事件。

**核心方法论**（基于 2024-2025 最新研究）：
- τ-bench 的 "Seed-LLM + 确定性组合"：LLM 生成叙事文本，代码生成结构化字段
- APIGen-MT 的 "反向任务重组"：先验证原子事件，再组合成链
- APIGen-MT 的 "LLM 委员会 + 反思循环"：多模型投票质审
- WildClawBench 的 "环境状态校验"：用 PerfectAgent/BadAgent 模拟跑验证区分度

## 2. Pipeline 总体架构

四阶段串行 pipeline，每步一个 click 子命令，中间产物 JSON/YAML 文件落盘：

```
lhb-pipeline download   →  data/raw_corpus/        (原始数据 → 筛选后文档片段)
lhb-pipeline extract    →  data/skeletons/          (文档片段 → 原子事件 → 事件链骨架)
lhb-pipeline generate   →  data/scenarios/          (骨架 → 完整场景 JSON)
lhb-pipeline validate   →  data/validated/          (多层验证 + 质量评审)
lhb-pipeline run-all                                (一条命令串联全流程)
```

**入口**：`longhorizon_bench/pipeline/cli.py`，注册为 `lhb-pipeline` console_scripts。

**中间产物目录结构**：

```
data/
├── raw_corpus/                    # Stage 1 输出
│   ├── petrochemical/             # IndustryCorpus2 石化子集
│   ├── mining/                    # IndustryCorpus2 采矿子集
│   ├── fire_safety/               # IndustryCorpus2 消防子集
│   └── regulations/               # 法规全文
├── skeletons/                     # Stage 2 输出
│   ├── atoms/                     # 原子事件（验证通过）
│   │   ├── ATOM-petro-001.yaml
│   │   └── ATOM-petro-002.yaml
│   └── chains/                    # 组合后的事件链骨架
│       ├── IND-001.yaml
│       └── IND-002.yaml
├── scenarios/                     # Stage 3 输出（最终产物）
│   └── industrial/
│       ├── IND-001.json
│       └── IND-002.json
├── background_docs/               # Stage 3 同步生成
│   ├── IND-001/
│   └── IND-002/
├── validated/                     # Stage 4 输出
│   └── industrial/
│       ├── IND-001.json           # 通过验证的场景（同 scenarios/ 内容）
│       └── IND-002.json
└── review_reports/                # Stage 4 评审报告
    ├── IND-001_report.json
    └── IND-002_report.json
```

每个阶段可独立运行、可重跑、可从任意中间产物恢复。

## 3. Stage 1 — 数据下载与筛选

**输入**：ModelScope / HuggingFace 数据集名
**输出**：`data/raw_corpus/` 下的筛选后文档片段

### 3.1 下载策略

**IndustryCorpus2（via ModelScope，无需审批）**：
- 只下载 `chinese/high/` 质量层的 parquet 文件
- 3 个子集：`BAAI/IndustryCorpus2_petrochemical`、`BAAI/IndustryCorpus2_mining`、`BAAI/IndustryCorpus2_fire_safety_food_safety`
- 下载命令：`modelscope download --dataset BAAI/IndustryCorpus2_petrochemical`
- 预计 high 层总量 ~3 GB

**法规全文（via HuggingFace，开放访问）**：
- `twang2218/chinese-law-and-regulations`，全量下载（22K 条，体积小）
- `datasets.load_dataset()` 直接加载

### 3.2 筛选规则

**工业语料筛选**：
- 关键词命中（至少 2 个）：`安全生产、压力容器、危险化学品、应急预案、隐患排查、设备检修、作业许可、安全评价、职业病、环境监测、污染防治、排放标准`
- 文档长度 ≥ 500 字
- 去重：simhash 相似度 > 0.9 的只保留一条

**法规筛选**：
- `title` 或 `office` 包含：`安全生产、应急管理、矿山、消防、危险化学品、环境保护、污染防治`
- `status` 为现行有效

### 3.3 输出格式

每条筛选后文档一个 JSON：

```json
{
  "doc_id": "petro_high_00123",
  "source": "IndustryCorpus2_petrochemical",
  "text": "...",
  "keywords_matched": ["压力容器", "安全评价"],
  "char_count": 2340
}
```

预期留存：工业语料 ~2000-5000 条，法规 ~500-1000 条。

## 4. Stage 2 — 原子事件生成与链式组合

**输入**：`data/raw_corpus/` 筛选后文档
**输出**：`data/skeletons/` 骨架文件

### 4.1 Step 2a — 文档聚类与主题提取

- TF-IDF 向量化 + K-Means 聚类，分出 ~10-15 个主题簇（如"压力容器检验"、"化学品泄漏应急"、"废气排放超标"）
- 每个簇取 top-5 代表文档 + 匹配的法规条文，形成一个"主题包"

### 4.2 Step 2b — 原子事件生成（Seed-LLM + 确定性组合）

对每个主题包，Claude 生成 3-5 个独立的原子事件：
- **Claude 负责**：事件类型、触发描述、自然语言叙事、evidence map
- **代码负责**：`event_id` 编号、时间戳序列、`scoring_rule` 参数值和 match type、工具名从 `ToolRegistry` 选取

原子事件中间格式：

```yaml
atom_id: ATOM-petro-001
source_cluster: petrochemical_pressure_vessel
type: routine_inspection
trigger: "设备科提交3号车间季度巡检报告，发现1台压力容器接近检验周期"
expected_tool: submit_inspection_report
params:
  target: {value: "3号车间", match: contains}
  risk_level: {value: [medium, high], match: enum}
  findings: {keywords: [压力容器, 检验周期], match: keyword_coverage}
evidence:
  required_facts: ["压力容器定期检验周期为3年（安全生产法第34条）"]
  forbidden_actions: [approve_work_permit]
dimensions: [domain_knowledge]
is_critical: true
```

### 4.3 Step 2c — 原子事件自动验证

每个原子事件必须通过：
- `expected_tool` 在 `ToolRegistry` 中存在
- 所有 `params` 的键在对应工具的 `ToolDef.params` 中存在
- `evidence.required_facts` 至少 1 条且可追溯到源文档或法规
- `match` 类型合法（exact / contains / enum / keyword_coverage）
- 不通过的带错误信息反馈给 Claude 重生成，最多 3 轮

### 4.4 Step 2d — 链式组合（反向任务重组）

从验证通过的原子事件池中组合完整链：

- **Claude 负责**：设计因果依赖图（`depends_on`）、插入检查点位置（间隔 3-7 事件）、合成时间线叙事连贯性
- **代码负责**：验证 DAG 无环、检查点间隔合规、`is_critical` 比例 ≥ 30%、三大维度各 ≥ 3 次覆盖

每条链 15-25 个决策事件 + 3-5 个检查点。

**质量门控**：组合后的骨架提交 DeepSeek 审核——"这条事件链的因果逻辑是否连贯？有无事件可删除而不影响链条完整性？" 双模型都通过才进入 Stage 3。

骨架输出格式：

```yaml
scenario_id: IND-001
domain: industrial
subdomain: safety_production
role: 化工厂安全生产管理员
time_span: 6个月
difficulty: 3
source_docs:
  - petro_high_00123
  - petro_high_00456
  - reg_安全生产法
events:
  - id: E01
    type: routine_inspection
    trigger: 设备老化报告
    depends_on: []
    is_critical: true
    dimensions: [domain_knowledge]
    atom_ref: ATOM-petro-001
  - id: E02
    type: policy_change
    trigger: 新安全生产法规发布
    depends_on: [E01]
    is_critical: false
    dimensions: [domain_knowledge, multi_step_reasoning]
    atom_ref: ATOM-petro-003
checkpoints:
  - id: CP01
    after: E05
    queries_target_dimensions: [long_term_memory]
  - id: CP02
    after: E12
    queries_target_dimensions: [consistency, long_term_memory]
  - id: CP03
    after: E19
    queries_target_dimensions: [multi_step_reasoning, long_term_memory]
```

## 5. Stage 3 — LLM 填充生成完整场景

**输入**：`data/skeletons/` 骨架 YAML
**输出**：`data/scenarios/industrial/` 场景 JSON + `data/background_docs/` 背景文档

### 5.1 Step 3a — 背景文档生成（分层构造）

**底层（~5000 字，真实改写）**：
- 从 raw_corpus 中提取与该骨架关联的真实法规/标准条文
- 改写为场景化语言，保留条款编号和核心数据点（如"检验周期3年"、"间距≥15m"）
- 改写公司名、人名等细节，避免直接再发布原文

**扩展层（扩展至 ~2 万字，Claude 生成）**：
- 基于底层法规文本扩展为完整背景文档
- 包括：公司概况、组织架构、人员配置、设备台账、检验记录、历史事故记录、整改情况
- 要求每个数据点都可被后续事件的 `evidence.required_facts` 引用

### 5.2 Step 3b — 事件详情填充（DeepSeek/Qwen 批量）

对骨架中每个事件节点，批量调用 DeepSeek 生成：

- `input`：事件具体描述文本（200-500 字），必须引用背景文档中的具体数据
- `scoring_rule`：已在原子事件中确定，代码直接转换格式
- `evidence`：已在原子事件中确定，补充具体文档位置引用
- 检查点 `checkpoint_queries`：基于前序事件生成回溯题

**prompt 策略**：
- 提供完整背景文档 + 当前事件骨架 + 前序事件已生成的 input
- 要求 input 文本中必须包含 `scoring_rule.params` 涉及的关键信息
- 要求检查点问题的答案必须依赖 3 个以上事件之前的信息（强制长程记忆测试）

### 5.3 Step 3c — Schema 转换与组装

代码将中间产物组装为符合 `Scenario` model 的 JSON：

- 骨架元数据 → `scenario_id`, `domain`, `role`, `difficulty`
- 原子事件 → `ActionEvent`（含 `scoring_rule`, `evidence`）
- 检查点 → `CheckpointEvent`（含 `checkpoint_queries`）
- 填充 `total_events`, `total_checkpoints`, `background_tokens` 等计数字段
- 记录 `generation_model`（如 "claude-sonnet-4-20250514+deepseek-v3"）
- 组装后调用 `Scenario.model_validate()` 确保 schema 合法

## 6. Stage 4 — 自动验证与质量评审

**输入**：`data/scenarios/industrial/` 场景 JSON
**输出**：`data/validated/` 通过验证的场景 + `data/review_reports/` 质量报告

### 6.1 Layer 1 — 结构验证（代码，秒级）

- `Scenario.model_validate()` schema 合法性
- `validate_scenario.py` 已有检查：事件数一致、ID 唯一、depends_on 引用合法
- 新增检查：
  - 背景文档字数 ≥ 15000 字
  - 每个 `required_facts` 至少能在背景文档或前序事件 input 中 keyword 命中
  - `scoring_rule` 中每个 param 的 expected 值在对应事件 input 中有语义关联
  - 检查点间隔 3-7 个 action event
  - 一致性规则 C01-C06 在 PerfectAgent 模拟跑中零违规

### 6.2 Layer 2 — LLM 委员会评审（Claude + DeepSeek 双模型投票）

每个场景提交给两个模型独立评分，5 个维度各 1-5 分：

| 维度 | 评审标准 |
|------|---------|
| 因果连贯性 | 事件间 depends_on 关系是否合理，去掉任一事件是否破坏链条 |
| 证据可追溯 | required_facts 是否真的能从背景文档找到 |
| 难度梯度 | 事件链是否从简单到复杂递进 |
| 答案区分度 | scoring_rule 是否能区分好坏 agent |
| 专业准确性 | 法规引用、行业术语是否正确 |

**通过标准**：
- 两个模型在每个维度评分差异 ≤ 1 分
- 每个维度均分 ≥ 3.5
- 不满足一致性的维度，自动带两个模型的反馈重新生成该部分，最多 2 轮
- 仍不通过的标记为 `needs_human_review`

### 6.3 Layer 3 — PerfectAgent + BadAgent 模拟跑

用 `LongHorizonEnv` 跑完整评测（复用已有代码）：

- PerfectAgent（按 scoring_rule 构造完美回答）：chain_score 必须 > 0.95，chain_pass = True
- BadAgent（随机工具）：chain_score 必须 < 0.3
- 两者分差 > 0.6（确保场景有区分度）

### 6.4 输出报告

每个场景一份 JSON 评审报告：

```json
{
  "scenario_id": "IND-001",
  "structural_checks": {"passed": 12, "failed": 0},
  "committee_scores": {
    "claude": {"因果连贯性": 4, "证据可追溯": 5, "难度梯度": 4, "答案区分度": 5, "专业准确性": 4},
    "deepseek": {"因果连贯性": 4, "证据可追溯": 4, "难度梯度": 4, "答案区分度": 4, "专业准确性": 4}
  },
  "simulation": {
    "perfect_agent_score": 0.97,
    "bad_agent_score": 0.12,
    "delta": 0.85
  },
  "verdict": "PASS"
}
```

## 7. LLM 使用策略

| 阶段 | 模型 | 用途 | 原因 |
|------|------|------|------|
| Stage 2b 原子事件生成 | Claude | 种子事件叙事 + evidence 设计 | 需要高质量推理能力 |
| Stage 2d 链式组合 | Claude | 因果依赖设计 + 叙事连贯性 | 需要全局理解 |
| Stage 2d 质量门控 | DeepSeek | 独立审核骨架逻辑 | 交叉验证，避免单模型偏差 |
| Stage 3a 背景文档扩展 | Claude | 扩展法规为完整背景 | 需要长文本生成质量 |
| Stage 3b 事件详情填充 | DeepSeek/Qwen | 批量生成事件 input | 成本可控，规模化 |
| Stage 4 委员会评审 | Claude + DeepSeek | 双模型独立评分 | 避免循环偏差 |

**模型隔离原则**：生成阶段使用的模型（Claude + DeepSeek）与主要被测 baseline 模型（GPT-4o）不同，且 `generation_model` 字段记录完整模型 ID 和版本。

**API 调用方式**：
- Claude：通过 `anthropic` SDK 调用，模型 ID 如 `claude-sonnet-4-20250514`
- DeepSeek：通过 `openai` SDK 调用（OpenAI 兼容接口），`base_url="https://api.deepseek.com"`，模型 ID 如 `deepseek-chat`
- `llm_client.py` 统一封装，通过 `provider` 参数切换后端，对上层暴露相同接口

## 8. 依赖与新增包

```
# 新增 pipeline 依赖
modelscope>=1.0          # 数据集下载
datasets>=2.0            # HuggingFace datasets 加载
scikit-learn>=1.0        # TF-IDF + K-Means 聚类
simhash>=2.0             # 文档去重
pyyaml>=6.0              # 骨架 YAML 读写
anthropic>=0.30          # Claude API
openai>=1.0              # DeepSeek API（OpenAI 兼容）
```

## 9. 文件结构

```
longhorizon_bench/pipeline/
├── __init__.py
├── cli.py                  # click 命令入口（download/extract/generate/validate/run-all）
├── downloader.py           # Stage 1: 数据下载与筛选
├── clustering.py           # Stage 2a: TF-IDF 聚类
├── atom_generator.py       # Stage 2b: 原子事件生成
├── atom_validator.py       # Stage 2c: 原子事件验证
├── chain_composer.py       # Stage 2d: 链式组合
├── bg_generator.py         # Stage 3a: 背景文档生成
├── event_filler.py         # Stage 3b: 事件详情填充
├── assembler.py            # Stage 3c: Schema 组装
├── structural_validator.py # Stage 4 Layer 1: 结构验证
├── committee_reviewer.py   # Stage 4 Layer 2: LLM 委员会评审
├── simulation_validator.py # Stage 4 Layer 3: 模拟跑验证
└── llm_client.py           # 统一 LLM 调用封装（Claude/DeepSeek 切换）
```
