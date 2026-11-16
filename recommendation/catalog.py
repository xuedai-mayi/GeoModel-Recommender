"""Model metadata and MDL declarations; never equate a declaration with execution."""
import xml.etree.ElementTree as ET
from collections import defaultdict
from .schemas import Capability


TOOLS = {
    "reproject": {"name": "QGIS / GDAL 栅格重投影", "operation": "gdal:warpreproject",
                  "url": "https://docs.qgis.org/3.44/en/docs/user_manual/processing_algs/gdal/rasterprojections.html",
                  "description": "将栅格转换到目标坐标系，并设置分辨率与重采样方法。",
                  "inputs": ["raster"], "outputs": ["raster"]},
    "clip_vector": {"name": "QGIS 矢量裁剪", "operation": "native:clip",
                    "url": "https://docs.qgis.org/3.44/en/docs/user_manual/processing_algs/qgis/vectoroverlay.html",
                    "description": "用多边形范围裁剪矢量图层。", "inputs": ["vector", "vector"], "outputs": ["vector"]},
    "intersection": {"name": "QGIS 矢量相交", "operation": "native:intersection",
                     "url": "https://docs.qgis.org/3.44/en/docs/user_manual/processing_algs/qgis/vectoroverlay.html",
                     "description": "计算矢量图层相交部分及属性。", "inputs": ["vector", "vector"], "outputs": ["vector"]},
    "zonal_stats": {"name": "QGIS 分区统计", "operation": "native:zonalstatisticsfb",
                    "url": "https://docs.qgis.org/3.44/en/docs/user_manual/processing_algs/qgis/rasteranalysis.html",
                    "description": "按多边形分区统计栅格；需确认数值是总量还是密度及聚合方式。",
                    "inputs": ["raster", "vector"], "outputs": ["vector"]},
    "raster_calculator": {"name": "QGIS 栅格计算器", "operation": "native:rastercalc",
                          "url": "https://docs.qgis.org/3.44/en/docs/user_manual/processing_algs/qgis/rasteranalysis.html",
                          "description": "对栅格执行表达式计算；表达式与栅格对齐需另行确定。",
                          "inputs": ["raster"], "outputs": ["raster"]},
}


def mdl_interfaces(rows):
    result = defaultdict(list)
    for row in rows:
        try:
            root = ET.fromstring(row.get("mdl") or "")
        except ET.ParseError:
            continue
        ports = []
        for node in root.iter():
            if node.tag.split("}")[-1] != "Event":
                continue
            attrs = node.attrib
            desc = attrs.get("description", "")
            # Event direction is preserved as a raw declaration. The user sees
            # the original description, not a guessed executable signature.
            ports.append({"name": attrs.get("name", ""), "event_type": attrs.get("type", ""),
                          "optional": attrs.get("optional", ""), "description": desc[:1500]})
        if not ports:
            continue
        interface = {"application_id": row.get("id"), "name": row.get("name", ""),
                     "status": "declared_unverified", "ports": ports,
                     "source": "caller_supplied_application"}
        for mid in row.get("relatedModelItems") or []:
            result[mid].append(interface)
    return dict(result)


class Catalog:
    def __init__(self, records, *, applications=(), capabilities=()):
        """Build from caller-owned records; no files or model data are bundled.

        Records require model_id, name, text, tier and url; description and
        slot_* labels are optional. Embeddings must follow the same row order.
        Application and capability records follow the schemas in this package.
        """
        from copy import deepcopy
        rows = deepcopy(list(records))
        required = ("model_id", "name", "text", "tier", "url")
        if not rows or any(any(not isinstance(r.get(k), str) or not r[k].strip()
                               for k in required) for r in rows):
            raise ValueError("模型记录不能为空，且 model_id、name、text、tier、url 必须为非空字符串")
        self.records = {r["model_id"]: r for r in rows}
        if len(self.records) != len(rows):
            raise ValueError("模型 ID 必须唯一")
        applications = deepcopy(list(applications))
        self.interfaces = mdl_interfaces(applications)
        self.capabilities = load_capabilities(deepcopy(list(capabilities)), self.records, applications)

    def card(self, mid):
        r = self.records[mid]
        return {"id": mid, "name": r["name"], "url": r["url"], "tier": r["tier"],
                "description": r.get("description", ""),
                "labels": {k: r.get(k, "unknown") for k in
                           ("slot_domain", "slot_task", "slot_method", "slot_input", "slot_output", "slot_scale")},
                "interfaces": self.interfaces.get(mid, []),
                "capabilities": self.capabilities.get(mid, []),
                "verification": "接口待核验" if self.interfaces.get(mid) else "输入输出待补充"}

    def summary(self):
        return {"models": len(self.records), "models_with_mdl": len(set(self.records) & set(self.interfaces)),
                "tools": len(TOOLS), "executable_models": 0,
                "reviewed_capabilities": sum(c['status'] == 'reviewed' for cs in self.capabilities.values() for c in cs)}


def exact_evidence(quote, record):
    return bool(quote.strip()) and any(quote in (record.get(k) or "")
                                     for k in ("name", "description", "text"))


def load_capabilities(rows, records, applications):
    """Only locally reviewed, source-checked facts may support hard decisions.

    Raw MDL Event direction is not interpreted as a typed scientific interface.
    Application variants remain separate, including when their ports are unknown.
    """
    apps = {row['id']: row for row in applications if row.get('id')}
    result, keys = defaultdict(list), set()
    for row in rows:
        cap = Capability.model_validate(row)
        record = records.get(cap.model_id)
        if record is None:
            raise ValueError('能力档案引用不存在的模型：' + cap.model_id)
        app = apps.get(cap.application_id)
        if cap.application_id and (not app or cap.model_id not in (app.get('relatedModelItems') or [])):
            raise ValueError('能力档案的 application 不属于该模型')
        sources = {'description': record.get('description', ''), 'mdl': (app or {}).get('mdl', '') or ''}
        evidence = cap.evidence + [e for p in cap.inputs + cap.outputs for e in p.evidence]
        if any(not e.quote.strip() or e.quote not in sources[e.source] for e in evidence):
            raise ValueError('能力证据不属于所引用的模型或 application 原文：' + cap.model_id)
        if cap.key in keys:
            raise ValueError('重复的模型实现能力档案')
        keys.add(cap.key)
        result[cap.model_id].append(cap.model_dump())
    by_model = defaultdict(list)
    for app in applications:
        for mid in app.get('relatedModelItems') or []:
            by_model[mid].append(app)
    for mid in records:
        represented = {c['application_id'] for c in result[mid]}
        for app in by_model[mid]:
            aid = app.get('id', '')
            if aid and mid in (app.get('relatedModelItems') or []) and aid not in represented:
                result[mid].append(Capability(model_id=mid, application_id=aid).model_dump())
                represented.add(aid)
        if not result[mid]:
            result[mid].append(Capability(model_id=mid).model_dump())
    return dict(result)
