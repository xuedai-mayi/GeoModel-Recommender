# GeoModel Recommender

[English](README.md) | [中文](README.zh-CN.md)

GeoModel Recommender 是面向地理模型的需求驱动推荐引擎，结合混合检索与能力证据匹配，根据研究任务、分析对象、目标输出和已有数据推荐模型。

支持单模型推荐、多步骤分析流程规划和带来源引用的推荐解释。

## 推荐流程

```text
研究需求 → 需求解析 → 名称 BM25 / 正文 BM25 / 向量检索
                     ↓
                 RRF 排名融合
                     ↓
            功能证据、输入输出和条件核对
                     ↓
            候选分组、排序与冲突诊断
```

引擎将复杂需求拆成多个分析步骤，为各步骤选择模型并检查输入输出衔接。RAG 模块生成推荐解释，并核对引用是否来自对应候选的原文。

- 三路检索各取最多 50 个候选，推荐引擎融合召回最多 40 个。
- 候选先按证据状态分组，再结合需求中明确指定的模型名、用户偏好和检索分数排序。
- 功能核对依据能力档案中的任务、对象、输出、输入条件及原文证据。检索相关性用于召回，接口声明作为待核验资料。
- 输出包括候选模型、条件缺口、冲突诊断和分析流程，供后续建模使用。

## 运行环境

本地验证环境为 Python 3.12，核心依赖版本如下：

```bash
git clone https://github.com/xuedai-mayi/GeoModel-Recommender.git
cd GeoModel-Recommender
python -m pip install numpy==2.5.4 bm25s==0.3.16 haystack-ai==3.4.1 pydantic==2.14.3
```

在仓库根目录运行调用代码，通过 `recommendation` 包导入算法接口。

## 最小调用示例

以下示例使用构造的模型记录和二维向量演示调用流程。实际使用时，用同一个嵌入模型生成文档向量和查询向量，并替换示例中的 `embed_query` 函数。

```python
from recommendation import Intent, build_engine
from recommendation.schemas import Port

records = [
    {
        "model_id": "demo-flood",
        "name": "DemoFlood",
        "text": "Flood simulation and water depth estimation.",
        "description": "Flood simulation and water depth estimation.",
        "tier": "A",
        "url": "https://example.org/demo-flood",
    },
    {
        "model_id": "demo-forest",
        "name": "DemoForest",
        "text": "Forest vegetation classification.",
        "description": "Forest vegetation classification.",
        "tier": "A",
        "url": "https://example.org/demo-forest",
    },
]

embeddings = [[1.0, 0.0], [0.0, 1.0]]

def embed_query(text):
    # 演示占位函数；实际使用时替换为同一嵌入模型的编码函数。
    return [1.0, 0.0]

engine = build_engine(records, embeddings, embed_query)
intent = Intent(
    goal="洪水模拟，输出水深",
    topic="洪水",
    task="模拟",
    object="洪水",
    output_requirements=[Port(name="水深", quantity="water_depth")],
)
result = engine.recommend(intent.goal, intent, limit=5)

for item in result["items"]:
    print(item["name"], item["match"]["status"])
```

示例中的候选返回 `needs_review`，表示需要补充已核验能力档案。`tier` 保存模型元数据，功能核验结果见 `match.status`。

## 输入与结果

入口函数：

```python
build_engine(records, embeddings, embed_query, *, applications=(), capabilities=())
```

| 参数 | 格式与要求 |
| --- | --- |
| `records` | 非空模型记录序列；`model_id`、`name`、`text`、`tier`、`url` 均为非空字符串，ID 唯一 |
| `embeddings` | `N × D` 数值数组，与记录顺序一致；每行必须为有限、非零向量 |
| `embed_query` | 接收查询字符串，返回同一向量空间中的 `D` 维有限、非零向量 |
| `applications` | 可选应用记录，包含 `id`、`relatedModelItems` 和原始 `mdl` 声明等字段 |
| `capabilities` | 可选能力档案，遵循 [Capability](recommendation/schemas.py) 定义 |

记录可附带 `description` 和 `slot_domain`、`slot_task`、`slot_method`、`slot_input`、`slot_output`、`slot_scale` 标签。读取模型记录并生成向量后，调用 `build_engine()` 在内存中构建索引；嵌入模型的加载和缓存由接入程序管理。

能力档案通过 `model_id` 关联模型，可通过 `application_id` 区分具体实现。已核验档案需要提供任务、对象、输入输出及原文证据；证据片段必须属于对应模型描述或应用的 MDL 原文。字段和验证规则见 [schemas.py](recommendation/schemas.py) 与 [catalog.py](recommendation/catalog.py)。

`engine.recommend()` 返回：

| 字段 | 含义 |
| --- | --- |
| `items` | 排除明确冲突项后，按请求数量上限返回的候选 |
| `pool` | 完整召回候选及其不同实现的核对结果 |
| `diagnostics` | 存在明确冲突的候选与原因 |
| `candidates` | 参与核对的召回模型数量 |
| `match_summary` | 本次推荐的文字概述 |

候选的 `match.status` 包含 `suitable`（当前条件适用）、`conditional`（附条件）、`needs_review`（待核验）和 `incompatible`（明确冲突）。其中，`suitable` 表示通过现有能力档案与需求条件的核对。检索分数用于候选排序；模型的实际运行效果需在建模阶段验证。

## 需求解析与流程推荐

在上面的 `engine` 基础上，可以直接处理自然语言需求：

```python
from recommendation.workflow import run_recommendation

response = run_recommendation(
    engine=engine,
    message="洪水模拟，输出水深",
    previous=None,
    client=None,
    task_mode="auto",
)
result = response["result"]
```

单需求支持本地解析和匹配。`task_mode="auto"` 按解析结果判断是否拆解；`task_mode="workflow"` 显式请求流程规划。生成复杂需求的计划时，通过 `client` 参数传入 AI 客户端。

[JSONClient](recommendation/backend.py) 接收 `api_key`、`base_url`、`model` 配置，调用支持结构化 JSON 输出的聊天接口。API 密钥通过运行时配置传入。自定义客户端需实现 `request(task, payload, schema)` 方法。

流程入口返回 `intent`、`is_workflow`、`result` 和 `notices`。生成 RAG 解释时，先按 [rag.py](recommendation/rag.py) 中 `evidence_context()` 的字段要求组织候选响应，再调用 `generate_explanation()`。

## 源码结构

| 文件 | 职责 |
| --- | --- |
| `retrieval.py` | 分词、BM25、余弦检索和 RRF 融合 |
| `engine.py`、`matching.py` | 候选核对、分组、排序和诊断 |
| `catalog.py`、`schemas.py` | 模型记录、能力档案、证据与字段验证 |
| `semantics.py` | 显式中英文术语对应 |
| `intelligence.py` | 需求解析、检索查询和计划生成 |
| `coordination.py`、`validation.py` | 逐步骤候选选择、输入输出衔接与条件检查 |
| `workflow.py` | 需求到推荐结果的统一入口 |
| `rag.py`、`dialogue.py` | 来源约束解释与追问生成 |
| `backend.py` | 引擎构建与结构化 AI 客户端 |

