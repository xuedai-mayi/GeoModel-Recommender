"""Resident lexical indexes and exact cosine retrieval for the model catalog."""
import re
import unicodedata
from dataclasses import replace
from functools import lru_cache

import bm25s
import numpy as np
from haystack import Document
from haystack.components.joiners import DocumentJoiner

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*|[\u3400-\u9fff]")
RECALL_K = 50


def tokenize(text):
    # Keep identifiers such as LISFLOOD-FP and PM2.5 intact in both indexes.
    return TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", text).casefold())


def normalized_matrix(embeddings, count):
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != count or not matrix.shape[1]:
        raise ValueError("向量数量或维度与模型库不一致")
    if not np.isfinite(matrix).all():
        raise ValueError("向量缓存含有非有限值")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("向量缓存含有零向量")
    matrix = np.ascontiguousarray(matrix / norms)
    matrix.setflags(write=False)
    return matrix


class HybridIndex:
    """One immutable catalog snapshot; queries never rebuild document arrays."""

    def __init__(self, records, embeddings, embed_query):
        ids = [r["model_id"] for r in records]
        if not ids or len(set(ids)) != len(ids) or any(not mid for mid in ids):
            raise ValueError("模型库不能为空，且模型 ID 必须非空、唯一")
        self.matrix = normalized_matrix(embeddings, len(records))
        self.embed_query = embed_query
        self.documents = tuple(Document(id=r["model_id"], content=r["text"], meta={
            "name": r["name"], "tier": r["tier"],
        }) for r in records)
        self.body = bm25s.BM25(method="lucene", dtype="float64")
        self.names = bm25s.BM25(method="lucene", dtype="float64")
        self.body.index([tokenize(r["text"]) for r in records], show_progress=False)
        self.names.index([tokenize(r["name"]) for r in records], show_progress=False)
        self.joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion")

    @lru_cache(maxsize=256)
    def query_vector(self, query):
        vector = np.asarray(self.embed_query(query), dtype=np.float32)
        if vector.shape != (self.matrix.shape[1],) or not np.isfinite(vector).all():
            raise ValueError("查询向量维度错误或含有非有限值")
        norm = np.linalg.norm(vector)
        if not norm:
            raise ValueError("查询向量不能为零向量")
        vector = vector / norm
        vector.setflags(write=False)
        return vector

    def _top(self, scores, limit, positive_only=False):
        indices = np.argsort(-scores, kind="stable")[:limit]
        return [replace(self.documents[int(i)], score=float(scores[i])) for i in indices
                if not positive_only or scores[i] > 0]

    def search(self, query, limit=40):
        if not query.strip():
            raise ValueError("检索查询不能为空")
        if limit < 1:
            raise ValueError("检索数量必须大于零")
        tokens = list(dict.fromkeys(tokenize(query)))
        body = self._top(self.body.get_scores(tokens), RECALL_K, positive_only=True)
        names = self._top(self.names.get_scores(tokens), RECALL_K, positive_only=True)
        dense = self._top(self.matrix @ self.query_vector(query), RECALL_K)
        return self.joiner.run(documents=[body, dense, names], top_k=limit)["documents"]

