"""Match each function independently, then check the selected connections once."""
import copy
from .schemas import validate_plan
from .validation import validate_workflow, selected_candidate
from .intelligence import retrieval_query
from .matching import candidate_key, summarize

CANDIDATES_PER_STEP = 3


def bind(step, candidate):
    step['selected_model'] = candidate['id'] if candidate else None
    step['selected_capability'] = candidate['capability_id'] if candidate else None


def retrieve_candidates(plan, intent, engine, candidate_limit=CANDIDATES_PER_STEP, single=False, pins=None):
    workflow, pools = plan.model_dump(), {}
    final_ids = {p.from_step for p in plan.deliverables}
    for step in workflow['steps']:
        if step['kind'] != 'model':
            continue
        preferences = dict(intent.preferences)
        if len(plan.steps) > 1:
            preferences.pop('slot_task', None)
            preferences.pop('slot_input', None)
            if step['id'] not in final_ids:
                preferences.pop('slot_output', None)
        local = intent if single else intent.model_copy(update=dict(
            goal=step['objective'], topic=step['query'], topic_en='', task=step['task'],
            object=step['object'], desired_output='、'.join(p['name'] for p in step['outputs']),
            required_outputs=[], output_requirements=[], identifiers=[], preferences=preferences, complex=False))
        result = engine.recommend(retrieval_query(local), local, limit=40, plan=plan, step_id=step['id'])
        pools[step['id']] = result
        candidates = sorted([c for r in result['pool'] for c in r['implementations']], key=candidate_key)
        usable = [c for c in candidates if c['match']['status'] != 'incompatible']
        # Distinct models get the first slots; implementations remain separately selectable.
        shortlist, seen = [], set()
        for candidate in usable:
            if candidate['id'] not in seen:
                shortlist.append(candidate)
                seen.add(candidate['id'])
        shortlist = shortlist[:candidate_limit]
        for candidate in usable:
            if len(shortlist) >= candidate_limit:
                break
            if candidate not in shortlist:
                shortlist.append(candidate)
        pinned = (pins or {}).get(step['id'])
        if pinned:
            chosen = next((c for c in candidates if c['capability_id'] == pinned), None)
            if chosen is None:
                raise ValueError('用户已选模型实现不在当前候选中，请重新选择')
            if chosen not in shortlist:
                shortlist = shortlist[:candidate_limit - 1] + [chosen]
        step.update(candidates=shortlist,
                    rejected_candidates=[c for c in candidates if c['match']['status'] == 'incompatible'],
                    match_summary=result['match_summary'])
        bind(step, None)
    return workflow, pools


def select_candidates(workflow, intent, pins=None):
    """Keep other choices fixed; surface conflicts instead of searching combinations."""
    checked = copy.deepcopy(workflow)
    pins = pins or {}
    nodes = [s for s in checked['steps'] if s['kind'] == 'model']
    if set(pins) - {s['id'] for s in nodes}:
        raise ValueError('所选模型所在的功能已改变，请重新选择')
    for step in nodes:
        choices = step.get('candidates', [])
        if step['id'] in pins:
            selected = next((c for c in choices if c['capability_id'] == pins[step['id']]), None)
            if selected is None:
                raise ValueError('所选模型实现不属于当前步骤候选')
        else:
            selected = selected_candidate(step) or next(
                (c for c in choices if c['match']['status'] != 'incompatible'), None)
        bind(step, selected)
    validate_workflow(checked, intent)
    checked['selection'] = {'pins': pins, 'method': 'per_function',
                            'no_known_conflicts': not bool(checked['conflicts'])}
    for step in nodes:
        selected = selected_candidate(step)
        if selected:
            checks = [c for c in checked['checks'] if c['step_id'] == step['id']]
            step['selected_match'] = summarize(checks)
    return checked


def coordinate(plan, intent, engine, single=False, pins=None):
    validate_plan(plan, intent.required_outputs)
    raw, pools = retrieve_candidates(plan, intent, engine, 5 if single else CANDIDATES_PER_STEP, single, pins)
    return select_candidates(raw, intent, pins), pools


def planning_question(workflow, asked, turn, mode):
    if mode == 'off' or turn >= 3:
        return None
    for gap in workflow['data_gaps']:
        key = 'data:' + gap['name']
        if key not in asked and gap['availability'] in ('unknown', 'missing'):
            return {'slot': key, 'text': f"功能 {gap['step_id']} 需要「{gap['name']}」，这项数据目前是什么情况？",
                    'options': ['已具备', '可以补充', '无法获取'], 'mode': 'planning',
                    'data_name': gap['name'], 'step_id': gap['step_id']}
    return None
