# GeoModel Recommender

面向地理模型的需求驱动推荐引擎。根据研究任务、分析对象、目标输出和已有数据，检索候选模型，并核对功能证据与使用条件。

本仓库从 OpenGMS 模型推荐项目中提取，包含算法源码。模型目录、能力档案、向量、嵌入模型和 API 配置均由调用方提供，仓库不附带本地数据或模型权重。

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

复杂需求可以拆成多个分析步骤，逐步选择模型并检查输入输出衔接。带引用的解释由独立 RAG 模块生成，引用会与候选所属原文核对。

- 三路检索各取最多 50 个候选，推荐引擎融合召回最多 40 个。
- 候选优先按证据状态分组，再结合明确模型名、用户偏好和检索分数排序。
- 功能核对使用任务、对象、输出、输入条件及来源证据；检索相关性与接口声明不自动视为适用性证据。
- 输出为推荐和流程描述；算法不执行地理模型。

## 运行环境

本地验证环境为 Python 3.12，核心依赖版本如下：

```bash
git clone https://github.com/xuedai-mayi/GeoModel-Recommender.git
cd GeoModel-Recommender
python -m pip install numpy==2.5.4 bm25s==0.3.16 haystack-ai==3.4.1 pydantic==2.14.3
```

在仓库根目录运行调用代码。当前仓库直接提供 `recommendation` 源码包，没有打包安装配置。

## 最小调用示例

以下记录与二维向量均为演示构造，只展示接口如何调用。实际检索应使用同一个嵌入模型生成文档向量和查询向量。

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

示例没有提供已核验能力档案，因此候选为 `needs_review`。记录中的 `tier` 是元数据字段，不代表功能已经核验。

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

记录可附带 `description` 和 `slot_domain`、`slot_task`、`slot_method`、`slot_input`、`slot_output`、`slot_scale` 标签。引擎在内存中构建索引，数据读取、嵌入模型加载和缓存由调用方管理。

能力档案通过 `model_id` 关联模型，可通过 `application_id` 区分具体实现。已核验档案需要提供任务、对象、输入输出及原文证据；证据片段必须属于对应模型描述或应用的 MDL 原文。字段和验证规则见 [schemas.py](recommendation/schemas.py) 与 [catalog.py](recommendation/catalog.py)。

`engine.recommend()` 返回：

| 字段 | 含义 |
| --- | --- |
| `items` | 按请求数量返回的候选，不含明确冲突项 |
| `pool` | 完整召回候选及其不同实现的核对结果 |
| `diagnostics` | 存在明确冲突的候选与原因 |
| `candidates` | 参与核对的召回模型数量 |
| `match_summary` | 本次推荐的文字概述 |

候选的 `match.status` 包含 `suitable`（当前条件适用）、`conditional`（附条件）、`needs_review`（待核验）和 `incompatible`（明确冲突）。检索分数不是适用概率，`suitable` 表示通过现有档案与条件核对，不代表已实际运行模型。

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

单需求支持本地解析和匹配。`task_mode="auto"` 按解析结果判断是否拆解；`task_mode="workflow"` 显式请求流程规划。复杂需求的计划生成需要调用方提供 AI 客户端。

[JSONClient](recommendation/backend.py) 接收 `api_key`、`base_url`、`model` 配置，调用支持结构化 JSON 输出的聊天接口。密钥由调用方管理，不应写入源码。也可以传入实现 `request(task, payload, schema)` 方法的客户端。

流程入口返回 `intent`、`is_workflow`、`result` 和 `notices`。RAG 解释通过 [rag.py](recommendation/rag.py) 的 `generate_explanation()` 单独调用；该函数需要符合其 `evidence_context()` 定义的候选响应结构，并非直接接收整个流程入口返回值。

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

## 验证范围

提取时完成了 8 项本地检查，覆盖独立调用、流程输出、调用方输入隔离、无效记录与向量、证据归属，以及合成输入下与原算法推荐结果的一致性。测试文件与测试数据未随本仓库上传。

这些检查验证提取后的接口和行为，不构成真实模型推荐准确率评测。实际推荐质量取决于调用方的模型目录、嵌入模型和能力证据覆盖情况。
