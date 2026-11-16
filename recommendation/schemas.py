"""JSON contracts shared by planning, persistence, API and evaluation."""
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlanningError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class GeneratedContractError(ValueError):
    """Retain generated data for bounded correction, without logging provider state."""
    def __init__(self, document, errors):
        super().__init__('生成内容不符合结构契约')
        self.document = document
        self.errors = errors


class Intent(Contract):
    goal: str = Field(default="", max_length=2000)
    topic: str = Field(default="", max_length=300)
    topic_en: str = Field(default="", max_length=300)
    identifiers: list[str] = Field(default_factory=list, max_length=30)
    task: str = Field(default="", max_length=100)
    object: str = Field(default="", max_length=200)
    output_requirements: list['Port'] = Field(default_factory=list, max_length=8)
    region: str = Field(default="", max_length=200)
    time_range: str = Field(default="", max_length=200)
    available_data: list[str] = Field(default_factory=list, max_length=30)
    unavailable_data: list[str] = Field(default_factory=list, max_length=30)
    desired_output: str = Field(default="", max_length=400)
    preferences: dict[str, str] = Field(default_factory=dict)
    complex: bool = False
    unobtainable_data: list[str] = Field(default_factory=list, max_length=30)
    data_assets: list['DataAsset'] = Field(default_factory=list, max_length=30)
    constraints: list['Constraint'] = Field(default_factory=list, max_length=30)
    required_models: list[str] = Field(default_factory=list, max_length=8)
    required_outputs: list[str] = Field(default_factory=list, max_length=8)


class Port(Contract):
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["raster", "vector", "table", "timeseries", "report", "unknown"] = "unknown"
    from_step: str | None = Field(default=None, description='输入端口或最终交付引用的来源步骤；步骤自身的 outputs 必须为空，不引用自己或其他步骤')
    from_output: str = Field(default='', description='from_step 非空时必须填写：来源步骤 outputs 中的精确 name，与本地端口 name 可以不同')
    unit: str = ""
    crs: str = ""
    resolution: str = ""
    time_step: str = ""
    quantity: str = ""
    extent: str = ""

    @model_validator(mode='after')
    def explicit_source(self):
        if bool(self.from_step) != bool(self.from_output):
            raise ValueError('来源引用必须同时填写 from_step 和 from_output；外部输入及步骤自身输出均留空')
        return self


class DataAsset(Port):
    availability: Literal['available', 'missing', 'unavailable', 'unknown'] = 'unknown'


class Constraint(Contract):
    field: Literal['scale', 'method', 'model_name', 'input_kind', 'output_kind']
    value: str = Field(min_length=1, max_length=200)
    strength: Literal['hard', 'soft'] = 'hard'
    scope: Literal['global', 'final', 'step'] = 'global'
    step_id: str = ''

    @model_validator(mode='after')
    def scoped(self):
        if (self.scope == 'step') != bool(self.step_id):
            raise ValueError('步骤约束必须指定 step_id；其他约束不得指定 step_id')
        return self


class Evidence(Contract):
    source: Literal['description', 'mdl']
    quote: str = Field(min_length=1, max_length=4000)


class CapabilityPort(Port):
    required: bool = True
    condition: str = ''
    evidence: list[Evidence] = Field(default_factory=list, max_length=10)


class Capability(Contract):
    model_id: str
    application_id: str = ''
    version: str = ''
    status: Literal['declared', 'reviewed'] = 'declared'
    function: str = ''
    tasks: list[str] = Field(default_factory=list, max_length=10)
    objects: list[str] = Field(default_factory=list, max_length=10)
    scale: str = ''
    method: str = ''
    conditions: list[str] = Field(default_factory=list, max_length=10)
    inputs: list[CapabilityPort] = Field(default_factory=list, max_length=30)
    outputs: list[CapabilityPort] = Field(default_factory=list, max_length=30)
    inputs_complete: bool = False
    outputs_complete: bool = False
    evidence: list[Evidence] = Field(default_factory=list, max_length=20)

    @property
    def key(self):
        import json
        return json.dumps([self.model_id, self.application_id, self.version], ensure_ascii=False, separators=(',', ':'))

    @model_validator(mode='after')
    def grounded_review(self):
        if self.status == 'reviewed':
            if not self.evidence or any(not p.evidence for p in self.inputs + self.outputs):
                raise ValueError('核验能力及各端口必须具有来源证据')
        if any(p.from_step for p in self.inputs + self.outputs):
            raise ValueError('模型能力端口不得包含规划步骤引用')
        return self


class Step(Contract):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    title: str = Field(min_length=1, max_length=120)
    kind: Literal["model", "tool", "check"]
    objective: str = Field(min_length=1, max_length=800)
    task: str = Field(default='', max_length=100)
    object: str = Field(default='', max_length=200)
    query: str = Field(default="", max_length=500)
    depends_on: list[str] = Field(default_factory=list, max_length=10)
    inputs: list[Port] = Field(default_factory=list, max_length=15, description='上游输入必须同时指定 from_step 和 from_output；name 是本步骤的输入名称')
    outputs: list[Port] = Field(default_factory=list, max_length=10, description='本步骤产生的输出；每个输出 from_step 必须为 null')
    tool_id: str | None = None


class Deliverable(Port):
    from_step: str = Field(min_length=1, description='产生最终结果的步骤 id')
    from_output: str = Field(min_length=1, description='来源步骤 outputs 中的精确 name；可不同于面向用户的交付物名称 name')


class Plan(Contract):
    title: str = Field(min_length=1, max_length=160)
    steps: list[Step] = Field(min_length=1, max_length=8)
    questions: list[str] = Field(default_factory=list, max_length=6)
    assumptions: list[str] = Field(default_factory=list, max_length=8)
    deliverables: list[Deliverable] = Field(default_factory=list, max_length=8, description='最终交付的用户名称及其步骤、输出引用；必须填写 from_step 和 from_output')


SLOT_LABELS = {"slot_domain": "研究领域", "slot_task": "任务", "slot_method": "方法",
               "slot_scale": "尺度", "slot_input": "输入形式", "slot_output": "输出形式"}


def ordered_steps(plan: Plan) -> list[Step]:
    """Reject missing references, cycles and invalid port references."""
    by_id = {s.id: s for s in plan.steps}
    if len(by_id) != len(plan.steps):
        raise ValueError("步骤编号重复")
    for step in plan.steps:
        if len(set(step.depends_on)) != len(step.depends_on):
            raise ValueError("步骤依赖重复")
        if any(d not in by_id for d in step.depends_on):
            raise ValueError("引用了不存在的前置步骤")
        for port in step.inputs:
            if port.from_step and port.from_step not in step.depends_on:
                raise ValueError("输入来源必须列入前置步骤")
            if port.from_step and sum(p.name == port.from_output for p in by_id[port.from_step].outputs) != 1:
                raise ValueError(f"{step.id} 的输入 {port.name} 引用了不存在的输出 {port.from_step}/{port.from_output}")
        if any(p.from_step for p in step.outputs):
            raise ValueError("输出不能引用其他步骤")
    ordered, remaining = [], dict(by_id)
    while remaining:
        ready = [s for s in remaining.values() if all(d not in remaining for d in s.depends_on)]
        if not ready:
            raise ValueError("流程存在循环依赖")
        for step in ready:
            ordered.append(step)
            del remaining[step.id]
    return ordered


def plan_from_workflow(workflow):
    data = {key: workflow[key] for key in Plan.model_fields if key in workflow}
    data['steps'] = [{key: step[key] for key in Step.model_fields if key in step}
                     for step in workflow['steps']]
    return Plan.model_validate(data)


def validate_plan(plan, required_outputs=()):
    plan.steps = ordered_steps(plan)
    nodes = {s.id: s for s in plan.steps}
    if not plan.deliverables:
        raise ValueError('计划必须声明最终产出 deliverables')
    names = [p.name for p in plan.deliverables]
    if len(set(names)) != len(names):
        raise ValueError('最终产出名称重复')
    if not set(required_outputs).issubset(names):
        raise ValueError('计划遗漏已确认的最终产出')
    for output in plan.deliverables:
        source = nodes.get(output.from_step)
        if source is None or sum(p.name == output.from_output for p in source.outputs) != 1:
            raise ValueError(f'最终产出 {output.name} 的引用 {output.from_step}/{output.from_output} 不存在；请核对来源步骤 outputs')
    # Every node must contribute to a deliverable or explicitly verify another node.
    return plan


Intent.model_rebuild()
