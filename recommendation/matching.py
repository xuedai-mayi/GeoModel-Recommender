"""Evidence-based requirement matching, shared by direct and decomposed requests."""
from .schemas import Capability
from .validation import validate_workflow

MATCH_ORDER = {'suitable': 0, 'conditional': 1, 'needs_review': 2, 'incompatible': 3}
MATCH_LABELS = {'suitable': '当前条件适用', 'conditional': '附条件候选',
                'needs_review': '主题相关', 'incompatible': '当前需求不适用'}


def summarize(checks):
    conflicts = [c for c in checks if c['status'] == 'conflict']
    functional = [c for c in checks if c['code'] in ('model_task', 'model_object', 'model_output')]
    complete = bool(functional) and all(c['status'] == 'supported' for c in functional)
    status = ('incompatible' if conflicts else 'needs_review' if not complete else
              'conditional' if any(c['status'] == 'unknown' for c in checks) else 'suitable')
    return {'status': status, 'label': MATCH_LABELS[status], 'checks': checks,
            'supported': [c['message'] for c in checks if c['status'] == 'supported'],
            'gaps': [c['message'] for c in checks if c['status'] == 'unknown'],
            'conflicts': [c['message'] for c in conflicts]}


def assess(candidate, intent, plan, step_id):
    """Probe one implementation without combining facts from other implementations."""
    probe = plan.model_dump()
    target = next(s for s in probe['steps'] if s['id'] == step_id)
    target.update(candidates=[candidate], selected_model=candidate['id'],
                  selected_capability=candidate['capability_id'])
    checked = validate_workflow(probe, intent)
    checks = [c for c in checked['checks'] if c['step_id'] == step_id and not c['from_step']
              and c['code'] not in ('required_model', 'constraint_scope')]
    # A required model is a local restriction only for a single modelling function.
    if sum(s.kind == 'model' for s in plan.steps) == 1:
        checks += [c for c in checked['checks'] if c['code'] == 'required_model']
    match = summarize(checks)
    candidate.update(match=match, local_conflicts=[c for c in checks if c['status'] == 'conflict'])
    if match['conflicts']:
        candidate['why'] = '；'.join(dict.fromkeys(match['conflicts']))
    return candidate


def candidate_key(candidate):
    return (MATCH_ORDER[candidate['match']['status']], not candidate['exact_name'],
            -candidate['preference_matches'], -candidate['score'], candidate['capability_id'])


def implementations(row, intent, plan, step_id):
    candidates = []
    for profile in row.get('capabilities') or [Capability(model_id=row['id']).model_dump()]:
        cap = Capability.model_validate(profile)
        candidate = {k: v for k, v in row.items() if k != 'capabilities'}
        candidate.update(capability=cap.model_dump(), capability_id=cap.key)
        candidates.append(assess(candidate, intent, plan, step_id))
    return sorted(candidates, key=candidate_key)
