"""Evidence-aware checks for requirements, capabilities and actual data bindings."""
import re
from .catalog import TOOLS
from .schemas import ordered_steps, plan_from_workflow
from .semantics import canonical


def norm(value):
    return ''.join(str(value or '').casefold().split())


def mentions_data(name, text):
    term = norm(name).removesuffix('数据')
    aliases = (('降雨', '降水', 'rainfall', 'precipitation'), ('人口', 'population'),
               ('dem', '数字高程', 'digital_elevation_model'), ('土地利用', 'land_use'),
               ('气象', 'weather'), ('土壤', 'soil'))
    terms = next((group for group in aliases if term in group), (term,))
    return bool(term) and any(value in norm(text) for value in terms)


def same_variable(a, b):
    if a.get('quantity') and b.get('quantity'):
        return canonical(a['quantity']) == canonical(b['quantity'])
    return canonical(a.get('quantity') or a.get('name')) == canonical(b.get('quantity') or b.get('name'))


def validate_declared_inputs(plan, intent):
    for step in plan.steps:
        for missing in intent.unavailable_data + intent.unobtainable_data:
            text = step.objective + ' ' + step.query
            term = re.escape(norm(missing).removesuffix('数据'))
            consumes = re.search(r'(?:使用|需要|依赖|输入|using|requires?|inputs?).{0,80}' + term, norm(text))
            if (step.kind != 'check' and consumes
                    and not any(mentions_data(missing, p.name) for p in step.inputs)):
                raise ValueError(f'{step.id} 的目标使用 {missing}，但 inputs 未列出该数据')


def data_asset(port, intent):
    matches = [a.model_dump() for a in intent.data_assets if same_variable(a.model_dump(), port)]
    asset = matches[0] if len(matches) == 1 else {'name': port['name'], 'availability': 'unknown'}
    for names, status in ((intent.available_data, 'available'), (intent.unavailable_data, 'missing'),
                          (intent.unobtainable_data, 'unavailable')):
        if any(same_variable({'name': name}, port) or mentions_data(name, port['name']) for name in names):
            asset = {**asset, 'availability': status}
    return asset


def selected_candidate(step):
    return next((c for c in step.get('candidates', []) if c['id'] == step.get('selected_model')
                 and c.get('capability_id') == step.get('selected_capability')), None)


def compare_ports(actual, expected, identity_bound=False):
    checks = []
    for field, label in (('quantity', '变量语义'), ('kind', '数据类型'), ('unit', '单位'),
                         ('crs', '坐标系'), ('resolution', '分辨率'), ('time_step', '时间步长'), ('extent', '空间范围')):
        wanted, found = expected.get(field), actual.get(field)
        if not wanted or wanted == 'unknown':
            continue
        equal = canonical(found) == canonical(wanted) if field == 'quantity' else found == wanted
        status = 'unknown' if not found or found == 'unknown' else 'supported' if equal else 'conflict'
        message = f'{label}资料未明确：{wanted}' if status == 'unknown' else f'{label}一致：{wanted}' if status == 'supported' else f'{label}声明不一致：{found} / {wanted}'
        checks.append((status, field, message))
    if not identity_bound and not same_variable(actual, expected):
        checks.append(('unknown', 'quantity', '资料未明确变量含义与对应关系'))
    return checks


def validate_workflow(workflow, intent):
    order = ordered_steps(plan_from_workflow(workflow))
    nodes = {s['id']: s for s in workflow['steps']}
    workflow['steps'] = [nodes[s.id] for s in order]
    checks, edges, gaps = [], [], []

    def add(step, code, status, message, field='', source=None, evidence=()):
        row = dict(step_id=step['id'], code=code, status=status, message=message,
                   field=field, from_step=source, evidence=list(evidence))
        checks.append(row)
        return row

    def external(step, port):
        asset = data_asset(port, intent)
        state = asset['availability']
        if state != 'available':
            label = {'missing': '缺少外部数据：', 'unavailable': '无法获取必需数据：', 'unknown': '需确认外部数据：'}[state]
            gap = dict(step_id=step['id'], name=port['name'], availability=state)
            if gap not in gaps:
                gaps.append(gap)
                add(step, 'data_gap', 'conflict' if state == 'unavailable' else 'unknown', label + port['name'], 'data')
        else:
            add(step, 'data_available', 'supported', '已有数据：' + port['name'], 'data')
            for status, field, message in compare_ports(asset, port):
                add(step, 'data_contract', status, message, field)
        return asset

    final_ids = {p.get('from_step') for p in workflow.get('deliverables', [])}
    for step in workflow['steps']:
        step['bindings'] = {'inputs': [], 'outputs': []}
        selected = selected_candidate(step)
        cap = (selected or {}).get('capability') or {}
        reviewed = cap.get('status') == 'reviewed'
        for condition in cap.get('conditions', []):
            add(step, 'applicability', 'unknown', '适用条件：' + condition, evidence=cap.get('evidence', []))
        if step['kind'] == 'model':
            if not step.get('selected_model'):
                add(step, 'no_model', 'unknown', '尚未找到已确认匹配的模型，保留该功能需求')
            elif not reviewed:
                add(step, 'capability_unknown', 'unknown', '模型资料尚未明确输入、输出及适用条件')
            for field, label, key in (('task', '任务功能', 'tasks'), ('object', '适用对象', 'objects')):
                wanted = step.get(field, '')
                values = cap.get(key, []) if reviewed else []
                supported = bool(wanted) and any(canonical(wanted) == canonical(v) for v in values)
                add(step, 'model_' + field, 'supported' if supported else 'unknown',
                    (label + '依据：' if supported else label + '资料未明确：') + (wanted or '需求未明确'),
                    field, evidence=cap.get('evidence', []) if reviewed else [])
            if not step['outputs']:
                add(step, 'model_output', 'unknown', '所需输出尚未明确', 'output')
            if reviewed and not cap.get('inputs_complete'):
                add(step, 'incomplete_inputs', 'unknown', '模型必需输入清单尚不完整')
            actual_outputs = cap.get('outputs', []) if reviewed else []
            for expected in step['outputs']:
                matches = [p for p in actual_outputs if same_variable(p, expected)]
                actual = matches[0] if len(matches) == 1 else {}
                if not actual:
                    conflict = bool(reviewed and cap.get('outputs_complete') and expected.get('quantity')
                                    and actual_outputs and all(p.get('quantity') for p in actual_outputs))
                    add(step, 'model_output', 'conflict' if conflict else 'unknown',
                        ('模型未提供所需输出：' if conflict else '资料未明确所需输出：') + expected['name'],
                        'output', evidence=cap.get('evidence', []))
                else:
                    add(step, 'model_output', 'supported', '模型输出依据：' + expected['name'], 'output', evidence=actual.get('evidence', []))
                    for status, field, message in compare_ports(actual, expected):
                        add(step, 'model_output', status, message, field, evidence=actual.get('evidence', []))
                step['bindings']['outputs'].append({'name': expected['name'], 'actual': actual})
                used_downstream = any(p.get('from_step') == step['id'] and p.get('from_output') == expected['name']
                                      for other in nodes.values() for p in other['inputs'])
                missing_source = any(mentions_data(name, expected['name'] + ' ' + expected.get('quantity', ''))
                                     for name in intent.unavailable_data + intent.unobtainable_data)
                if used_downstream and missing_source and (not actual or not cap.get('inputs_complete')):
                    # An unverified data-generation step cannot fill an explicit user data gap.
                    external(step, expected)
        elif step['kind'] == 'tool':
            tool = TOOLS.get(step.get('tool_id'))
            if not tool:
                add(step, 'unknown_tool', 'conflict', '工具不在已知目录中')
            else:
                step['tool'] = tool
                kinds = [p['kind'] for p in step['inputs']]
                for kind in set(tool['inputs']):
                    if kinds.count(kind) < tool['inputs'].count(kind):
                        add(step, 'tool_inputs', 'conflict', '工具必需的输入类型/数量待补充：' + kind)
                for kind in tool['outputs']:
                    if kind not in [p['kind'] for p in step['outputs']]:
                        add(step, 'tool_outputs', 'conflict', '工具输出类型不符合目录声明：' + kind)
                add(step, 'tool_parameters', 'unknown', '需确定工具参数、变量语义及转换条件')
                for p in step['outputs']:
                    actual = {'name': p['name'], 'kind': p['kind']} if p['kind'] in tool['outputs'] else {}
                    step['bindings']['outputs'].append({'name': p['name'], 'actual': actual})
                if len(nodes) > 1 and step['id'] not in final_ids and not any(step['id'] in s['depends_on'] for s in nodes.values()):
                    add(step, 'unused_tool', 'unknown', '该工具输出未接入后续步骤；请确认是否遗漏依赖')
        else:
            add(step, 'manual_check', 'unknown', '该步骤需要人工完成并核验结果')
            for port in step['outputs']:
                asset = data_asset(port, intent)
                if asset['availability'] != 'unknown':
                    actual = external(step, port)
                    step['bindings']['outputs'].append({'name': port['name'], 'actual': actual if actual['availability'] == 'available' else {}})

        for port in step['inputs']:
            source_id = port.get('from_step')
            if not source_id:
                actual = external(step, port)
            else:
                source = nodes[source_id]
                declared = next(p for p in source['outputs'] if p['name'] == port['from_output'])
                edge_checks = []
                for status, field, message in compare_ports(declared, port, identity_bound=True):
                    if status == 'conflict':
                        edge_checks.append(add(step, 'declared_edge', status, message, field, source_id))
                binding = next((b for b in source['bindings']['outputs'] if b['name'] == port['from_output']), {})
                actual = binding.get('actual') or {}
                if actual:
                    for status, field, message in compare_ports(actual, port, identity_bound=True):
                        edge_checks.append(add(step, 'actual_edge', status, message, field, source_id, actual.get('evidence', [])))
                else:
                    edge_checks.append(add(step, 'actual_edge', 'unknown', '上游真实输出尚未核验：' + port['name'], 'output', source_id))
                status = 'incompatible' if any(c['status'] == 'conflict' for c in edge_checks) else 'needs_review'
                edges.append(dict(**{'from': source_id, 'to': step['id']}, data=port['name'], status=status,
                                  from_output=port['from_output'],
                                  reasons=[c['message'] for c in edge_checks] or ['计划声明未见冲突，实际接口信息尚不完整']))
            step['bindings']['inputs'].append({'name': port['name'], 'from_step': source_id, 'from_output': port['from_output'], 'actual': actual})

        if reviewed:
            for required in cap.get('inputs', []):
                if not required.get('required', True):
                    continue
                if required.get('condition'):
                    add(step, 'conditional_input', 'unknown', '条件性输入需确认：' + required['name'] + '；' + required['condition'])
                    continue
                declared_inputs = {p['name']: p for p in step['inputs']}
                matches = [b for b in step['bindings']['inputs'] if norm(b['name']) == norm(required['name'])
                           or same_variable(b['actual'] or declared_inputs.get(b['name'], {'name': b['name']}), required)]
                if len(matches) == 1:
                    actual, source_id = matches[0]['actual'], matches[0]['from_step']
                elif not matches:
                    actual, source_id = external(step, required), None
                    step['bindings']['inputs'].append({'name': required['name'], 'from_step': None, 'from_output': '', 'actual': actual})
                else:
                    add(step, 'ambiguous_binding', 'unknown', '模型输入有多个可能来源：' + required['name'])
                    continue
                for status, field, message in compare_ports(actual, required):
                    add(step, 'model_input', status, message, field, source_id, required.get('evidence', []))

        for condition in intent.constraints:
            if condition.scope == 'step' and condition.step_id != step['id']:
                continue
            if condition.scope == 'final' and step['id'] not in final_ids:
                continue
            if step['kind'] != 'model' and not (condition.scope == 'final' and condition.field in ('input_kind', 'output_kind')):
                continue
            if condition.field == 'model_name':
                values = [(selected or {}).get('name', '')]
            elif condition.field in ('scale', 'method'):
                values = [cap.get(condition.field, '')] if reviewed else []
            else:
                direction = 'inputs' if condition.field == 'input_kind' else 'outputs'
                values = (TOOLS.get(step.get('tool_id'), {}).get(direction, []) if step['kind'] == 'tool'
                          else [p.get('kind', '') for p in cap.get(direction, [])] if reviewed else [])
            known = [v for v in values if v and v != 'unknown']
            complete = step['kind'] == 'tool' or condition.field in ('scale', 'method', 'model_name') or cap.get('inputs_complete' if condition.field == 'input_kind' else 'outputs_complete')
            status = 'supported' if any(norm(v) == norm(condition.value) for v in known) else 'conflict' if known and complete else 'unknown'
            if status == 'conflict' and condition.strength == 'soft':
                status = 'unknown'
            add(step, 'constraint', status, f'{condition.field} 要求 {condition.value}；模型声明 {"、".join(known) or "未知"}', condition.field, evidence=cap.get('evidence', []))

    for deliverable in workflow.get('deliverables', []):
        source = nodes.get(deliverable.get('from_step'))
        if source is None:
            add(workflow['steps'][0], 'deliverable', 'conflict', '最终产出引用不存在的步骤')
            continue
        declared = next((p for p in source['outputs'] if p['name'] == deliverable['from_output']), {})
        actual = next((b['actual'] for b in source['bindings']['outputs'] if b['name'] == deliverable['from_output']), {})
        requirement = {**deliverable, 'name': deliverable['from_output']}
        if not declared:
            add(source, 'deliverable', 'conflict', '最终产出未在源步骤声明：' + deliverable['name'])
        for status, field, message in compare_ports(declared, requirement):
            if status == 'conflict':
                add(source, 'deliverable', status, message, field)
        if not actual:
            add(source, 'deliverable', 'unknown', '资料未明确最终产出：' + deliverable['name'])
        else:
            for status, field, message in compare_ports(actual, requirement):
                add(source, 'deliverable', status, message, field, evidence=actual.get('evidence', []))
    for condition in intent.constraints:
        if condition.scope == 'step' and condition.step_id not in nodes:
            add(workflow['steps'][0], 'constraint_scope', 'conflict', '约束引用不存在的步骤：' + condition.step_id)
    selected_names = {norm((selected_candidate(s) or {}).get('name')) for s in nodes.values()}
    for name in intent.required_models:
        if norm(name) not in selected_names:
            add(workflow['steps'][0], 'required_model', 'conflict', '必须使用的模型尚未绑定：' + name)
    for edge in edges:
        conflicts = [c for c in checks if c['step_id'] == edge['to'] and c['from_step'] == edge['from'] and c['status'] == 'conflict']
        if conflicts:
            edge['status'] = 'incompatible'
            edge['reasons'] = list(dict.fromkeys(edge['reasons'] + [c['message'] for c in conflicts]))
    for step in workflow['steps']:
        own = [c for c in checks if c['step_id'] == step['id']]
        issues = [c['message'] for c in own if c['status'] != 'supported']
        blocked = any(c['status'] == 'conflict' for c in own) or any(g['step_id'] == step['id'] and g['availability'] == 'missing' for g in gaps)
        for dep in step['depends_on']:
            if nodes[dep].get('status') == 'blocked':
                blocked = True
                issues.append('前置步骤尚未满足条件：' + dep)
        step['issues'] = list(dict.fromkeys(issues))
        step['status'] = 'blocked' if blocked else 'needs_review' if issues else 'supported'
    workflow.update(edges=edges, checks=checks, conflicts=[c for c in checks if c['status'] == 'conflict'], data_gaps=gaps,
                    structure_status='valid', evidence_status='needs_review' if any(c['status'] == 'unknown' and c['code'] != 'data_gap' for c in checks) else 'supported',
                    data_status='incomplete' if gaps else 'ready', executable=False)
    workflow['status'] = 'blocked' if any(s['status'] == 'blocked' for s in nodes.values()) else 'needs_review' if any(s['status'] == 'needs_review' for s in nodes.values()) else 'supported'
    return workflow
