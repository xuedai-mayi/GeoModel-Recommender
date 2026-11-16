"""A single requirement-to-model path, with decomposition only when needed."""
from .intelligence import understand, create_plan, basic_plan, CHANGE_GOAL
from .coordination import coordinate
from .validation import data_asset


def reusable_plan(previous, intent, plan):
    if previous is None or plan is None:
        return False
    fields = ('topic', 'task', 'object', 'desired_output', 'required_outputs', 'output_requirements',
              'complex', 'identifiers', 'required_models', 'constraints', 'region', 'time_range')
    if any(getattr(previous, key) != getattr(intent, key) for key in fields):
        return False
    # Newly supplied intermediate results change the functional entry point.
    for step in plan.steps:
        for output in step.outputs:
            before = data_asset(output.model_dump(), previous)
            after = data_asset(output.model_dump(), intent)
            if before != after and after['availability'] == 'available':
                return False
    return True


def run_recommendation(engine, message, previous, client, task_mode, intent=None,
                       prior_plan=None, prior_pins=None, prior_is_workflow=False):
    notices = [] if client else ['当前使用本地需求解析和混合检索。']
    intent = intent or understand(message, previous, client, notices)
    is_workflow = task_mode == 'workflow' or (task_mode == 'auto' and intent.complex)
    reuse = (is_workflow == prior_is_workflow and not CHANGE_GOAL.search(message)
             and reusable_plan(previous, intent, prior_plan))
    plan = prior_plan if reuse else create_plan(intent, client, notices) if is_workflow else basic_plan(intent)
    pins = prior_pins if reuse else {}
    workflow, pools = coordinate(plan, intent, engine, single=not is_workflow, pins=pins)
    result = {'workflow': workflow, 'items': [], 'pool': [], 'candidates': 0,
              'diagnostics': [], 'match_summary': f"已按当前研究目标整理 {len(workflow['steps'])} 个分析步骤，并匹配各步骤可选的模型。"}
    if not is_workflow:
        step = workflow['steps'][0]
        result.update(pools[step['id']])
        # The engine already groups implementations by model and ranks by evidence.
        result['items'] = result['items'][:5]
        pinned = (pins or {}).get(step['id'])
        if pinned:
            selected = next(c for c in step['candidates'] if c['capability_id'] == pinned)
            result['items'] = [selected] + [c for c in result['items'] if c['id'] != selected['id']]
    return {'intent': intent, 'is_workflow': is_workflow, 'result': result, 'notices': notices}
