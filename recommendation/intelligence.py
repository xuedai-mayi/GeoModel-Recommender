"""Intent and plan generation with bounded, explicit offline fallbacks."""
import re
import logging
import json
from pydantic import ValidationError
from .schemas import Intent, Plan, Step, Port, Deliverable, validate_plan, PlanningError, GeneratedContractError
from .catalog import TOOLS
from .semantics import explicit_terms, canonical, TERMS

TOPICS = [
    (r"洪水|洪涝|淹没|内涝|flood|inundation", "洪水淹没", "flood inundation"),
    (r"侵蚀|产沙|erosion|sediment", "土壤侵蚀", "soil erosion"),
    (r"土地利用|土地变化|land.use|land.change", "土地利用变化", "land use change"),
    (r"植被|生产力|\bNPP\b|vegetation", "植被生产力", "vegetation productivity"),
    (r"径流|runoff|SWAT", "流域径流", "watershed runoff"),
]
DATA_NAMES = {"DEM": r"DEM|数字高程", "降雨": r"降雨|降水|rainfall", "人口": r"人口|population",
              "土地利用": r"土地利用|land.use", "遥感影像": r"遥感影像|卫星影像|satellite imagery",
              "土壤": r"土壤数据|soil data", "气象": r"气象数据|weather data",
              "水深": r"水深|water depth|flood depth", "淹没范围": r"淹没范围|inundation extent|flood extent"}
IDENTIFIER = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z]{2,}[A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)(?![A-Za-z0-9])")
CHANGE_GOAL = re.compile(r"(?:改为|改成|换成|改做)(?!\s*(?:全球|区域|流域|城市|坡面)尺度)|change (?:to|the topic)|instead", re.I)


def subject_text(message):
    """Preserve the user's scientific terms, excluding data/interaction clauses."""
    parts = []
    for clause in re.split(r"[，,。；;]", message):
        clause = re.split(r"已有|现有|拥有|(?<!有)没有|缺少|缺乏|不具备|(?:有|提供)\s*(?:DEM|降雨|人口|土地利用|遥感|土壤数据|气象数据)|\b(?:I|we) have\b", clause, maxsplit=1, flags=re.I)[0]
        clause = re.sub(r"(?:全球|区域|流域|城市|坡面)尺度", "", clause)
        clause = re.sub(r"(?:不要|不用|不再使用|取消)\s*[A-Za-z][A-Za-z0-9.-]*", "", clause)
        clause = re.sub(r"^(?:现在|改为|改成|换成|改做|请|帮我|我想|想|关于|有没有|推荐|找个|找一个|做点|做|研究|一下)+", "", clause.strip())
        clause = re.sub(r"(?:的研究|相关的模型|相关模型|模型推荐|模型|吗|吧|的)$", "", clause).strip()
        if clause and not re.fullmatch(r"[我想要帮推荐一个些模型分析做研究请吧\s]+", clause) and not re.search(r"给出.*流程|不要流程|不需要流程|只推荐|只要模型|不限.*尺度|取消.*尺度", clause):
            parts.append(clause)
    return " ".join(parts).strip()[:300]


def retrieval_query(intent):
    topic = intent.topic.replace(intent.region, "").strip() if intent.region else intent.topic
    parts = [*intent.identifiers, topic, intent.topic_en, intent.task, intent.object, intent.desired_output,
             *[p.name for p in intent.output_requirements], *intent.required_outputs]
    return " ".join(dict.fromkeys(p.strip() for p in parts if p.strip())) or intent.goal


def apply_data_directives(message, intent):
    # Split on sentence boundaries, not list punctuation, to preserve '有DEM、人口'.
    for clause in re.split(r"[，,。；;]", message):
        for name, pattern in DATA_NAMES.items():
            match = re.search(pattern, clause, re.I)
            if not match:
                continue
            prefix = clause[:match.start()]
            signals = list(re.finditer(r"无法获取|不能获取|无法提供|没有|缺少|缺乏|不具备|do not have|don't have|no |已有|现有|拥有|有|提供|have|available|无", prefix, re.I))
            signal = signals[-1].group().lower() if signals else ""
            impossible = signal in ('无法获取', '不能获取', '无法提供')
            negative = impossible or signal in ("没有", "缺少", "缺乏", "无", "不具备", "do not have", "don't have", "no ")
            positive = bool(signal) and not negative
            if negative or positive:
                for field in ('available_data', 'unavailable_data', 'unobtainable_data'):
                    setattr(intent, field, [x for x in getattr(intent, field) if not re.search(pattern, x, re.I)])
            if negative:
                intent.available_data = [x for x in intent.available_data if x != name]
                if name not in intent.unavailable_data:
                    intent.unavailable_data.append(name)
                if impossible and name not in intent.unobtainable_data:
                    intent.unobtainable_data.append(name)
            elif positive:
                intent.unavailable_data = [x for x in intent.unavailable_data if x != name]
                intent.unobtainable_data = [x for x in intent.unobtainable_data if x != name]
                if name not in intent.available_data:
                    intent.available_data.append(name)
            if negative or positive:
                for asset in intent.data_assets:
                    if re.search(pattern, asset.name, re.I):
                        asset.availability = 'unavailable' if impossible else 'missing' if negative else 'available'
    return intent


def apply_explicit_directives(message, intent, previous=None):
    """Explicit user choices have the same meaning with or without AI parsing."""
    for scale in ("全球", "区域", "流域", "城市", "坡面"):
        if re.search(scale + r"尺度", message):
            intent.preferences["slot_scale"] = scale
    if re.search(r"(?:不限|不再限定|取消).{0,12}尺度", message):
        intent.preferences.pop("slot_scale", None)
        intent.constraints = [c for c in intent.constraints if c.field != 'scale']
    if re.search(r"不要流程|不需要流程|只推荐模型|只要模型", message):
        intent.complex = False
    elif re.search(r"流程|步骤|子任务|然后|并统计|并分析|workflow|step.by.step", message, re.I):
        intent.complex = True
    prior = previous.identifiers if previous and not CHANGE_GOAL.search(message) else []
    removed = {s.casefold() for s in re.findall(r"(?:不要|不用|不再使用|取消)\s*([A-Za-z][A-Za-z0-9.-]*)", message)}
    required = re.findall(r'(?:必须使用|只允许使用)\s*([A-Za-z][A-Za-z0-9.+-]*)', message)
    intent.required_models = list(dict.fromkeys(n for n in intent.required_models + required if n.casefold() not in removed))
    explicit = IDENTIFIER.findall(subject_text(message))
    intent.identifiers = list(dict.fromkeys(s for s in [*prior, *explicit]
                                          if s.casefold() not in removed))[:30]
    for name in removed:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])"
        for field in ("goal", "topic", "topic_en", "desired_output"):
            setattr(intent, field, re.sub(pattern, "", getattr(intent, field), flags=re.I).strip())
    if not intent.object:
        objects = explicit_terms(subject_text(intent.topic),
                                 ('地表水流', '洪水', '流域径流', '决口冲积扇', '人口暴露'))
        if len(objects) == 1:
            intent.object = objects[0]
    if not intent.output_requirements:
        outputs = explicit_terms(subject_text(intent.topic) + ' ' + intent.desired_output,
                                 ('water_depth', 'water_discharge', 'flood_hydrograph', 'flood_extent',
                                  'population_count', 'economic_loss'))
        intent.output_requirements = [Port(name=TERMS[q][0], quantity=q) for q in outputs]
    return apply_data_directives(message, intent)


def offline_intent(message, previous=None):
    intent = previous.model_copy(deep=True) if previous and not CHANGE_GOAL.search(message) else Intent(goal=message)
    if previous and not CHANGE_GOAL.search(message):
        intent.goal = (previous.goal + "；补充：" + message)[-2000:]
    else:
        intent.goal = message
    subject = subject_text(message)
    if subject and (not previous or CHANGE_GOAL.search(message) or re.search(r"研究|模拟|评估|预测|估算", subject)):
        intent.topic, intent.topic_en = subject, ""
        intent.object = ''
        intent.output_requirements = []
    for pattern, topic, english in TOPICS:
        if re.search(pattern, intent.topic, re.I):
            intent.topic_en = english
            break
    if CHANGE_GOAL.search(message):
        intent.task = ""
    for task in ("模拟", "评估", "预测", "分类", "制图", "优化", "计算"):
        if task in message:
            intent.task = task
            break
    if "估算" in message:
        intent.task = "估算"
    if not intent.task:
        for pattern, task in ((r"simulation|simulate|modelling", "模拟"), (r"assessment|evaluate", "评估"),
                              (r"predict", "预测"), (r"classif", "分类"), (r"estimat|calculat", "计算")):
            if re.search(pattern, message, re.I):
                intent.task = task
                break
    for region in ("长三角", "中国", "全球"):
        if region in message:
            intent.region = region
    return apply_explicit_directives(message, intent, previous)


def understand(message, previous, client, notices):
    if client:
        try:
            result = client.request(
                "更新用户需求状态。保留用户此前未修改的信息，最新明确修改覆盖旧值。"
                "goal是合并后的完整目标，不要把取消的条件保留在goal。"
                "topic和topic_en写研究对象，并保留用户明确指定的模型名称，不包含地名或研究区；task写任务。用户已有与缺少的数据严格分开，不能把需要的数据当作已有。"
                "preferences仅使用slot_domain/task/method/input/scale/output，只填写本轮明确更新的偏好；"
                "明确取消某个偏好时将该字段设为空字符串。"
                "复杂问题含多个相互依赖的分析目标时complex=true。没有明确主题或任务则留空。"
                "required_outputs列出用户最终需要的产出名称，不包含中间结果。"
                "object写分析对象或过程，如地表水流、流域径流、人口暴露，不能只写宽泛领域。"
                "output_requirements将所需输出表示为带name、quantity及用户明确属性的端口，不能推测单位或数据类型。"
                "用户已有水深、淹没范围等中间成果时复用成果，重新判断剩余功能与complex，不重复要求上游模拟。"
                "constraints记录用户明确硬条件或软偏好，并注明global/final/step作用范围；最终产出形式只能约束final。"
                "required_models只记录用户明确要求必须使用的模型名称。unobtainable_data只记录用户明确无法获取的数据。"
                "data_assets记录用户明确提供的数据属性，不推测单位、格式、坐标系等。"
                "constraints、required_models、required_outputs、data_assets只在更新时返回完整新列表，明确取消返回空列表。",
                {"previous": previous.model_dump() if previous else None, "message": message}, Intent)
            result.goal = ((previous.goal + '；补充：' + message)[-2000:]
                           if previous and not CHANGE_GOAL.search(message) else message)
            if not re.search(r'必须|只允许|只用|仅用|must use|only use', message, re.I):
                result.required_models = list(previous.required_models) if previous and not CHANGE_GOAL.search(message) else []
            if not re.search(r'无法获取|不能获取|无法提供|cannot obtain|cannot provide', message, re.I):
                result.unobtainable_data = list(previous.unobtainable_data) if previous and not CHANGE_GOAL.search(message) else []
            from .schemas import SLOT_LABELS
            preferences = dict(previous.preferences) if previous else {}
            for k, v in result.preferences.items():
                if k in SLOT_LABELS:
                    if v:
                        preferences[k] = v
                    else:
                        preferences.pop(k, None)
            result.preferences = preferences
            # Positive/negative availability statements override each other;
            # omitted data never silently disappear on a follow-up.
            if previous:
                result.available_data = list(dict.fromkeys(previous.available_data + result.available_data))
                result.unavailable_data = list(dict.fromkeys(
                    [d for d in previous.unavailable_data if d not in result.available_data] + result.unavailable_data))
                if not CHANGE_GOAL.search(message):
                    for field in ('constraints', 'required_models', 'required_outputs', 'data_assets', 'object', 'output_requirements'):
                        if field not in result.model_fields_set:
                            setattr(result, field, getattr(previous, field))
                    result.unobtainable_data = list(dict.fromkeys(
                        [d for d in previous.unobtainable_data if d not in result.available_data] + result.unobtainable_data))
            result.available_data = [x for x in result.available_data if x not in result.unavailable_data + result.unobtainable_data]
            return apply_explicit_directives(message, result, previous)
        except Exception as error:
            logging.getLogger(__name__).warning('Intent parsing failed: %s', type(error).__name__)
            notices.append("AI 需求理解暂不可用，已使用本地规则；请核对需求摘要。")
    return offline_intent(message, previous)


def basic_plan(intent):
    outputs = [p.model_copy(deep=True) for p in intent.output_requirements]
    for name in intent.required_outputs or ([intent.desired_output] if intent.desired_output and not outputs else []):
        if name not in {p.name for p in outputs}:
            outputs.append(Port(name=name, quantity=canonical(name) if canonical(name) in TERMS else ''))
    outputs = outputs or [Port(name='分析结果')]
    return Plan(title="模型推荐方案", steps=[Step(
        id="analysis", title=intent.topic or "明确分析目标", kind="model",
        task=intent.task, object=intent.object,
        objective=intent.goal, query=" ".join([intent.topic, intent.task, intent.topic_en]),
        outputs=outputs)], deliverables=[Deliverable(**p.model_dump(exclude={'from_step', 'from_output'}), from_step='analysis', from_output=p.name) for p in outputs])


def create_plan(intent, client, notices):
    if not client:
        raise PlanningError('planning_requires_ai', '复杂任务规划需要开启 AI；请配置后重试')
    payload = {"intent": intent.model_dump(), "tools": TOOLS}
    for attempt in range(2):
        plan = None
        try:
            plan = client.request(
                "为地理分析目标规划1至6个必要步骤，输出有向无环依赖图。每步objective不超过120字。"
                "省略有默认值且为空的字段，避免冗长输出。"
                "各步骤输入输出是计划要求，不能声称已经有模型实现。model节点只给检索query，不指定模型。"
                "科学建模、模拟、推演、估算等核心子任务使用model节点；query同时含精简中文和英文术语，省略地名。"
                "每个model节点的task写具体任务类型，object写分析对象，outputs写所需变量语义。"
                "复用用户已提供的中间成果，删除仅用于生成这些成果的上游任务；拆解止于能检索模型的功能粒度。"
                "缺少数据时仍保留必要建模步骤并在输入列明缺失数据，不能擅自替换为简化假设或人工阈值法。"
                "用户明确缺少的数据保留为外部输入；未明确要求时不得新增估算、合成该数据的步骤。外部输入名称沿用用户数据名称。"
                "tool节点只能使用输入tools中的tool_id，每节点对应一次调用，输入输出类型和数量遵守目录。"
                "例如分区统计输出为带统计属性的矢量，不能直接说成统计表；重投影一次处理一份栅格。"
                "其他操作设为check并写明待人工完成。"
                "每条来自上游的输入必须写from_step与from_output，并列入depends_on。from_output精确引用来源步骤的outputs.name，本地name可不同。"
                "步骤自身outputs和外部输入：from_step=null且from_output为空字符串，两者不能只填一个。"
                "缺数据问题写在计划最外层questions数组；Step没有questions字段。不能虚构已具备的数据。"
                "deliverables列出全部最终产出，name为用户需要的名称，from_step指向实际产出步骤，from_output必须等于该步骤某个outputs.name。"
                "必须保留intent.required_outputs中的原始名称。quantity使用英文snake_case变量名，如water_depth、water_discharge、population_count、population_density；未知属性留空。"
                "检查任务与科学目标对应，人口暴露不能写成经济损失或完整风险。"
                "无需自动执行、无需编写代码。参数未知就留空。保留验证步骤，避免重复无意义步骤；删除没有下游使用且不属于最终产出的预处理步骤。",
                payload, Plan)
            # An explicit input reference already declares a dependency. Derive
            # its redundant graph edge; never invent inputs/outputs or capabilities.
            for step in plan.steps:
                step.depends_on = list(dict.fromkeys(step.depends_on +
                    [p.from_step for p in step.inputs if p.from_step]))
            validate_plan(plan, intent.required_outputs)
            from .validation import validate_declared_inputs
            validate_declared_inputs(plan, intent)
            for step in plan.steps:
                if step.kind == "tool" and step.tool_id not in TOOLS:
                    raise ValueError("工具不在已知目录中")
                if step.kind == "model" and not step.query.strip():
                    raise ValueError("模型步骤缺少检索描述")
            return plan
        except Exception as error:
            # Structural errors contain only plan fields. Never log provider errors
            # or Pydantic's input dump, which may include arbitrary response text.
            reason = (json.dumps(error.errors, ensure_ascii=False) if isinstance(error, GeneratedContractError) else
                      json.dumps([{'field': list(e['loc']), 'rule': e['type'], 'message': e['msg']}
                                  for e in error.errors(include_input=False)], ensure_ascii=False)
                      if isinstance(error, ValidationError) else str(error)[:500] if type(error) is ValueError else type(error).__name__)
            logging.getLogger(__name__).warning("Plan attempt %s failed: %s", attempt + 1, reason)
            if plan is not None:
                payload["previous_plan"] = plan.model_dump(exclude_defaults=True)
            elif isinstance(error, GeneratedContractError):
                payload['previous_plan'] = error.document
            payload["repair"] = "请直接修正previous_plan中被标记的字段；extra_forbidden字段必须删除，其他字段遵循Schema。检查错误：" + (
                reason)
    raise PlanningError('invalid_plan', 'AI 计划未通过结构检查，请明确最终产出后重试')
