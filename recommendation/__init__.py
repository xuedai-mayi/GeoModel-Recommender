"""GeoModel Recommender: retrieval, evidence matching and workflow planning.

Runtime dependencies: numpy, bm25s, haystack-ai and pydantic (version 2).
Use build_engine(records, embeddings, embed_query, applications=...,
capabilities=...) with caller-supplied inputs, then engine.recommend(query,
Intent(...)). For demand parsing and workflow planning use
recommendation.workflow.run_recommendation. Complex planning and RAG accept
an optional JSONClient configured by the caller; no credentials are bundled.
"""
from .backend import build_engine
from .schemas import Intent

__all__ = ["build_engine", "Intent"]
