"""
Enrichment & reasoning routes — extraction, link prediction, dedup, reasoning.
"""

import asyncio

from fastapi import APIRouter, Depends

from ..dependencies import get_session
from ..schemas import (
    DedupRequest,
    DedupResponse,
    EnrichExtractRequest,
    EnrichExtractResponse,
    LinkPredictionRequest,
    LinkPredictionResponse,
    ReasoningRequest,
    ReasoningResponse,
)
from ..session import GraphSession

router = APIRouter(tags=["Enrichment"])


@router.post("/api/enrich/extract", response_model=EnrichExtractResponse)
async def extract_entities(
    body: EnrichExtractRequest,
    session: GraphSession = Depends(get_session),
):
    """Extract entities and relations from free text."""
    try:
        from ...semantic_extract.methods import extract_entities as _extract_entities
        from ...semantic_extract.methods import extract_relations as _extract_relations

        entities = await asyncio.to_thread(_extract_entities, body.text)
        relations = await asyncio.to_thread(_extract_relations, body.text)

        ent_list = entities if isinstance(entities, list) else getattr(entities, "entities", [])
        rel_list = relations if isinstance(relations, list) else getattr(relations, "relations", [])

        return EnrichExtractResponse(
            entities=[_safe_dict(e) for e in ent_list],
            relations=[_safe_dict(r) for r in rel_list],
        )
    except ImportError:
        raise ValueError(
            "semantic_extract module not available. "
            "Ensure spacy and transformers are installed."
        )
    except Exception as exc:
        raise ValueError(f"Extraction failed: {exc}")


@router.post("/api/enrich/links", response_model=LinkPredictionResponse)
async def predict_links(
    body: LinkPredictionRequest,
    session: GraphSession = Depends(get_session),
):
    """Predict likely new edges for a node."""
    predictor = session.link_predictor
    if predictor is None:
        raise ValueError("LinkPredictor not available — KG extras may not be installed.")

    node = await asyncio.to_thread(session.get_node, body.node_id)
    if node is None:
        raise KeyError(body.node_id)

    # Get all nodes and edges to build candidate pairs manually,
    # because ContextGraph.nodes is a dict (not callable) and is incompatible
    # with LinkPredictor._get_all_nodes which requires nodes() or get_all_nodes().
    nodes, _ = await asyncio.to_thread(session.get_nodes, skip=0, limit=999_999)
    edges, _ = await asyncio.to_thread(session.get_edges, skip=0, limit=999_999)

    # Pre-compute already-connected node IDs so we skip them.
    # (LinkPredictor._edge_exists returns False for ContextGraph since it has no
    # has_edge/get_edge method, so we handle exclusion here instead.)
    existing_neighbours = {
        e.get("target") for e in edges if e.get("source") == body.node_id
    } | {
        e.get("source") for e in edges if e.get("target") == body.node_id
    }

    # Score each candidate via score_link (which works with ContextGraph because
    # it falls back to has_node / get_neighbors).
    scored = []
    for n in nodes:
        candidate = n.get("id")
        if not candidate or candidate == body.node_id or candidate in existing_neighbours:
            continue
        try:
            score = predictor.score_link(session.graph, body.node_id, candidate)
            if score > 0:
                scored.append(
                    {"target": candidate, "score": score, "type": n.get("type", "entity")}
                )
        except Exception:
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)

    return LinkPredictionResponse(
        node_id=body.node_id,
        predictions=scored[: body.top_n],
    )


@router.post("/api/enrich/dedup", response_model=DedupResponse)
async def detect_duplicates(
    body: DedupRequest,
    session: GraphSession = Depends(get_session),
):
    """Run a deduplication scan over graph entities."""
    try:
        from ...deduplication import DuplicateDetector

        detector = DuplicateDetector()
        # Use asyncio.to_thread — get_nodes acquires an RLock and must not block
        # the event loop.
        nodes, _ = await asyncio.to_thread(session.get_nodes, skip=0, limit=999_999)

        entities = [
            {
                "id": n.get("id"),
                "text": n.get("content", n.get("id", "")),
                "type": n.get("type", "entity"),
            }
            for n in nodes
        ]

        dups = await asyncio.to_thread(
            detector.detect_duplicates, entities, threshold=body.threshold
        )
        dup_list = dups if isinstance(dups, list) else getattr(dups, "duplicates", [])
        return DedupResponse(
            duplicates=[_safe_dict(d) for d in dup_list],
            total_flagged=len(dup_list),
        )
    except ImportError:
        raise ValueError("Deduplication module not available.")
    except Exception as exc:
        raise ValueError(f"Dedup scan failed: {exc}")


@router.post("/api/reason", response_model=ReasoningResponse)
async def run_reasoning(
    body: ReasoningRequest,
    session: GraphSession = Depends(get_session),
):
    """Run inference rules over facts."""
    try:
        from ...reasoning.reasoner import Reasoner

        reasoner = Reasoner()
        inferred = await asyncio.to_thread(
            reasoner.infer_facts, body.facts, body.rules
        )

        return ReasoningResponse(
            inferred_facts=inferred if isinstance(inferred, list) else [],
            rules_fired=len(inferred) if isinstance(inferred, list) else 0,
        )
    except ImportError:
        raise ValueError("Reasoning module not available.")
    except Exception as exc:
        raise ValueError(f"Reasoning failed: {exc}")


def _safe_dict(obj) -> dict:
    """Convert an object to a JSON-safe dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {"value": str(obj)}
