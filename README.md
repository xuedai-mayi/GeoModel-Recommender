# GeoModel Recommender

[English](README.md) | [中文](README.zh-CN.md)

GeoModel Recommender recommends geographic models based on research tasks, analysis objects, required outputs, and available data. It combines hybrid retrieval with checks against documented model capabilities.

It supports individual model recommendations, planning for analysis workflows with multiple steps, and recommendation explanations with source citations.

## Recommendation pipeline

```text
Research request → Requirement parsing → Name BM25 / Body BM25 / Vector search
                                             ↓
                                     RRF rank fusion
                                             ↓
                       Capability evidence, I/O, and condition checks
                                             ↓
                          Candidate grouping, ranking, and diagnostics
```

For complex requests, the engine generates analysis steps, selects models for each step, and checks the connections between their inputs and outputs. The RAG module generates explanations and verifies each citation against the corresponding candidate's source text.

- Each retrieval channel returns up to 50 candidates. The recommendation engine uses up to 40 candidates after fusion.
- Candidates are grouped by evidence status, then ranked using explicitly requested model names, user preferences, and retrieval scores.
- Capability checks use documented tasks, analysis objects, outputs, input conditions, and source evidence. Retrieval relevance guides candidate selection; interface declarations are treated as material awaiting review.
- Results include candidate models, unmet conditions, conflict diagnostics, and analysis workflows for subsequent modeling work.

## Setup

The locally verified environment uses Python 3.12 and the following dependency versions:

```bash
git clone https://github.com/xuedai-mayi/GeoModel-Recommender.git
cd GeoModel-Recommender
python -m pip install numpy==2.5.4 bm25s==0.3.16 haystack-ai==3.4.1 pydantic==2.14.3
```

Run your code from the repository root and import the API from `recommendation`.

## Minimal example

This example uses synthetic model records and two-dimensional vectors to demonstrate the API. For actual retrieval, generate document and query vectors with the same embedding model and replace `embed_query` with that model's encoder. The sample request asks in Chinese for flood simulation with water depth as the output.

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
    # Demo placeholder; replace with the encoder used for document vectors.
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

The candidates in this example return `needs_review`, indicating that reviewed capability profiles are needed. `tier` stores model metadata; `match.status` reports the result of capability checks.

## Inputs and results

Build an engine with:

```python
build_engine(records, embeddings, embed_query, *, applications=(), capabilities=())
```

| Parameter | Format and requirements |
| --- | --- |
| `records` | A nonempty sequence of model records. `model_id`, `name`, `text`, `tier`, and `url` must be nonempty strings, with unique model IDs. |
| `embeddings` | An `N × D` numeric array in the same order as the records. Each row must be a finite, nonzero vector. |
| `embed_query` | A function that takes a query string and returns a finite, nonzero vector of dimension `D` in the same embedding space. |
| `applications` | Optional application records with fields such as `id`, `relatedModelItems`, and raw `mdl` declarations. |
| `capabilities` | Optional capability profiles following the [Capability](recommendation/schemas.py) schema. |

Records can also include `description` and the labels `slot_domain`, `slot_task`, `slot_method`, `slot_input`, `slot_output`, and `slot_scale`. After loading records and generating vectors, call `build_engine()` to construct the index in memory. The integrating application manages embedding model loading and caching.

Capability profiles link to models through `model_id` and can distinguish implementations through `application_id`. Reviewed profiles provide tasks, analysis objects, inputs, outputs, and source evidence. Evidence excerpts must appear in the corresponding model description or application MDL text. See [schemas.py](recommendation/schemas.py) and [catalog.py](recommendation/catalog.py) for fields and validation rules.

`engine.recommend()` returns:

| Field | Meaning |
| --- | --- |
| `items` | Candidates up to the requested limit, after excluding explicit conflicts. |
| `pool` | All retrieved candidates and the check results for their implementations. |
| `diagnostics` | Candidates with explicit conflicts and the reasons for those conflicts. |
| `candidates` | The number of retrieved models checked. |
| `match_summary` | A text summary of the recommendation. |

Each candidate's `match.status` is `suitable`, `conditional`, `needs_review`, or `incompatible`. `suitable` means the candidate passes the checks against its available capability profile and the request conditions. Retrieval scores rank candidates; actual model performance is evaluated during modeling.

## Request parsing and workflow planning

Using the `engine` from the example above, process a natural language request with:

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

Single requests support local parsing and matching. `task_mode="auto"` determines whether to decompose a request from the parsed intent; `task_mode="workflow"` explicitly requests workflow planning. Pass an AI client through `client` to generate plans for complex requests.

[JSONClient](recommendation/backend.py) accepts `api_key`, `base_url`, and `model` settings and calls a chat API that supports structured JSON output. Supply API keys through runtime configuration. A custom client must implement `request(task, payload, schema)`.

The workflow entry point returns `intent`, `is_workflow`, `result`, and `notices`. To generate a RAG explanation, assemble a candidate response using the fields expected by `evidence_context()` in [rag.py](recommendation/rag.py), then call `generate_explanation()`.

## Source structure

| File | Responsibility |
| --- | --- |
| `retrieval.py` | Tokenization, BM25, cosine retrieval, and RRF fusion. |
| `engine.py`, `matching.py` | Candidate checks, grouping, ranking, and diagnostics. |
| `catalog.py`, `schemas.py` | Model records, capability profiles, evidence, and field validation. |
| `semantics.py` | Explicit mappings between Chinese and English terms. |
| `intelligence.py` | Request parsing, retrieval queries, and plan generation. |
| `coordination.py`, `validation.py` | Candidate selection per step, I/O connections, and condition checks. |
| `workflow.py` | The shared entry point from request to recommendation. |
| `rag.py`, `dialogue.py` | Explanations grounded in sources and clarification questions. |
| `backend.py` | Engine construction and the structured AI client. |
