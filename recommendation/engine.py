"""Shared hybrid retrieval, intent-aware ordering and source-grounded cards."""
import math
import re

from .catalog import exact_evidence
from .retrieval import tokenize
from .schemas import SLOT_LABELS
from .matching import implementations, candidate_key


def explicit_name(name, query):
    """An explicit model identifier has priority over generic semantic matches."""
    name = name.strip()
    if name.casefold() == query.strip().casefold():
        return True
    if not re.search(r"[A-Z]{2,}|[a-z][A-Z]|[A-Za-z][0-9]", name) or not re.fullmatch(r"[A-Za-z0-9 ._+()-]+", name):
        return False
    return bool(re.search(r"(?<![A-Za-z0-9_.+-])" + re.escape(name) + r"(?![A-Za-z0-9_.+-])", query, re.I))


def description_excerpt(record, query):
    description = record.get("description", "").strip()
    if record["tier"] == "C" or description.casefold() == record["name"].strip().casefold():
        return ""
    terms = set(tokenize(query))
    parts = [m.group().strip() for m in re.finditer(r"[^。！？\n]+[。！？]?", description)]
    excerpt = max(parts, key=lambda part: len(terms & set(tokenize(part))), default="")[:400]
    return excerpt if exact_evidence(excerpt, record) and excerpt in description else ""


class RecommendationEngine:
    def __init__(self, backend, catalog):
        self.backend, self.catalog = backend, catalog

    def recommend(self, query, intent, limit=5, plan=None, step_id=None):
        if plan is None:
            from .intelligence import basic_plan
            plan = basic_plan(intent.model_copy(update={'goal': intent.goal or query}))
        step_id = step_id or plan.steps[0].id
        rows, seen = [], set()
        for doc in self.backend.search(query, limit=40):
            if doc.id not in self.catalog.records:
                continue
            if doc.score is None or not math.isfinite(float(doc.score)):
                raise RuntimeError("检索返回了无效分数")
            record = self.catalog.records[doc.id]
            identity = doc.id
            if identity in seen:
                continue
            seen.add(identity)
            card = self.catalog.card(doc.id)
            evidence = description_excerpt(record, query)
            card.update(score=round(float(doc.score or 0), 6), relevance=None,
                        evidence=evidence, evidence_status="source_excerpt" if evidence else "insufficient",
                        constraints=[], rank_method="hybrid_rrf", exact_name=explicit_name(record["name"], query))
            rows.append(card)
        for row in rows:
            matches = 0
            for slot, value in intent.preferences.items():
                actual = row["labels"].get(slot, "unknown")
                status = "unknown" if actual in ("unknown", "", None) else "match" if actual == value else "unconfirmed"
                matches += status == "match"
                row["constraints"].append({"field": SLOT_LABELS.get(slot, slot), "wanted": value,
                                           "label": actual, "status": status})
            row["preference_matches"] = matches
            row["why"] = ("原文描述：" + row["evidence"] if row["evidence"] else "检索相关，能力描述待补充")
        checked = []
        for row in rows:
            variants = implementations(row, intent, plan, step_id)
            checked.append({**variants[0], 'implementations': variants})
        checked.sort(key=candidate_key)
        usable = [r for r in checked if r['match']['status'] != 'incompatible']
        def card(row):
            return {k: v for k, v in row.items() if k != 'implementations'}
        return {"items": [card(r) for r in usable[:limit]], "pool": checked, "notices": [], "candidates": len(rows),
                "diagnostics": [card(r) for r in checked if r['match']['status'] == 'incompatible'],
                "match_summary": ('已找到功能相关的模型，使用条件见推荐说明' if any(
                    r['match']['status'] in ('suitable', 'conditional') for r in usable) else
                    '以下模型与研究主题相关，具体用途和资料范围见推荐说明' if usable else
                    '召回候选与当前需求存在明确冲突' if checked else '未召回候选，保留原需求')}

    def question(self, result, intent, asked, turn, mode):
        if mode == "off" or turn >= 3:
            return None
        if not intent.topic and "slot_domain" not in asked:
            return {"slot": "slot_domain", "text": "你主要研究什么领域？",
                    "options": ["水文/洪水", "生态/植被", "城市", "土地利用"], "mode": "gap"}
        if not intent.task and "slot_task" not in asked:
            return {"slot": "slot_task", "text": "你希望完成哪类分析？",
                    "options": ["模拟", "评估", "预测", "计算"], "mode": "gap"}
        return None
