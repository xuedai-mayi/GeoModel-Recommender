"""Small, explicit vocabulary for matching; unknown terms keep their own identity."""
import re


def normalized(value):
    return re.sub(r'[\s_-]+', '', str(value or '').casefold())


TERMS = {
    '模拟': ('simulate', 'simulates', 'simulation', 'modelling', 'modeling'),
    '预测': ('predict', 'predicts', 'prediction'),
    '计算': ('calculate', 'calculates', 'calculation'),
    '估算': ('estimate', 'estimates', 'estimation'),
    '评估': ('assess', 'assessment', 'evaluate', 'evaluation'),
    '分类': ('classify', 'classification'),
    '地表水流': ('overland flow', '地表径流'),
    '洪水': ('flood', 'flooding'),
    '流域径流': ('watershed runoff', 'basin runoff'),
    '决口冲积扇': ('crevasse splay',),
    '人口暴露': ('population exposure',),
    'water_depth': ('水深', '洪水水深', 'water depth', 'flood depth'),
    'water_discharge': ('流量', '水流量', 'water discharge'),
    'flood_hydrograph': ('洪水流量过程', '洪水过程线', 'flood hydrograph', 'flood hydrographs'),
    'flood_extent': ('淹没范围', '洪水淹没范围', 'flood extent', 'inundation extent'),
    'population_count': ('人口数量', '受影响人口数量', 'population count'),
    'economic_loss': ('经济损失', 'economic loss'),
}
ALIASES = {normalized(term): key for key, values in TERMS.items() for term in (key, *values)}


def canonical(value):
    return ALIASES.get(normalized(value), normalized(value))


def explicit_terms(text, keys):
    """Find explicit nouns in a requirement, not inferred outputs of a topic."""
    matches = []
    for key in keys:
        for term in (key, *TERMS[key]):
            pattern = re.escape(term)
            if term.isascii():
                pattern = r'(?<![a-z])' + pattern + r'(?![a-z])'
            for match in re.finditer(pattern, text, re.I):
                if re.search(r'(?:不要|不需要|无需|不要求)(?:输出|计算|预测|估算|的)?\s*$|(?:without|not|no)\s*$',
                             text[max(0, match.start()-16):match.start()], re.I):
                    continue
                matches.append((match.start(), match.end(), key))
    selected = [m for m in matches if not any(a <= m[0] and b >= m[1] and b-a > m[1]-m[0]
                                              for a, b, _ in matches)]
    return list(dict.fromkeys(key for _, _, key in sorted(selected)))
