"""
Decision routes : decision listing, causal chains, precedents, compliance.

Uses ContextGraph-native queries so it works without a Neo4j/FalkorDB backend.
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_session
from ..schemas import (
    CausalChainResponse,
    ComplianceResponse,
    DecisionResponse,
)
from ..session import GraphSession

router = APIRouter(prefix="/api/decisions", tags=["Decisions"])


def _node_to_decision(n: dict) -> DecisionResponse:
    """Map a ContextGraph node dict to a DecisionResponse."""
    meta = n.get("metadata", {})
    return DecisionResponse(
        decision_id=n.get("id", ""),
        category=meta.get("category", ""),
        scenario=meta.get("scenario", ""),
        reasoning=meta.get("reasoning", ""),
        outcome=meta.get("outcome", ""),
        confidence=float(meta.get("confidence", 0.0)),
        timestamp=meta.get("timestamp"),
        metadata=meta,
    )


@router.get("", response_model=list[DecisionResponse])
async def list_decisions(
    category: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    session: GraphSession = Depends(get_session),
):
    """List decision nodes (type='decision') with optional category filter."""
    nodes, _ = await asyncio.to_thread(
        session.get_nodes, node_type="decision", skip=0, limit=999_999
    )

    if category:
        nodes = [
            n for n in nodes
            if n.get("metadata", {}).get("category", "").lower() == category.lower()
        ]

    page = nodes[skip: skip + limit]
    return [_node_to_decision(n) for n in page]


@router.get("/{decision_id}", response_model=DecisionResponse)
async def get_decision(
    decision_id: str,
    session: GraphSession = Depends(get_session),
):
    """Get a single decision by ID."""
    node = await asyncio.to_thread(session.get_node, decision_id)
    if node is None:
        raise KeyError(decision_id)
    return _node_to_decision(node)


@router.get("/{decision_id}/chain", response_model=CausalChainResponse)
async def get_causal_chain(
    decision_id: str,
    session: GraphSession = Depends(get_session),
):
    """
    Trace the causal chain for a decision.

    Uses BFS neighbour traversal over ``caused_by`` / ``influences``
    relationship types as a lightweight, backend-agnostic fallback.
    """
    node = await asyncio.to_thread(session.get_node, decision_id)
    if node is None:
        raise KeyError(decision_id)

    # Walk outbound causal edges (up to 5 hops)
    neighbors = await asyncio.to_thread(session.get_neighbors, decision_id, depth=5)
    chain = [
        {
            "id": nb.get("id"),
            "type": nb.get("type"),
            "relationship": nb.get("relationship"),
            "hop": nb.get("hop"),
            "content": nb.get("content", ""),
        }
        for nb in neighbors
    ]
    return CausalChainResponse(decision_id=decision_id, chain=chain)


@router.get("/{decision_id}/precedents", response_model=list[DecisionResponse])
async def get_precedents(
    decision_id: str,
    limit: int = Query(10, ge=1, le=100),
    session: GraphSession = Depends(get_session),
):
    """
    Find precedent decisions similar to the given decision.

    Lightweight: looks for other decision-type nodes and ranks by shared
    category and keyword overlap.
    """
    node = await asyncio.to_thread(session.get_node, decision_id)
    if node is None:
        raise KeyError(decision_id)

    meta = node.get("metadata", {})
    category = meta.get("category", "")
    scenario_words = set(meta.get("scenario", "").lower().split())

    all_decisions, _ = await asyncio.to_thread(
        session.get_nodes, node_type="decision", skip=0, limit=999_999
    )

    scored = []
    for d in all_decisions:
        if d.get("id") == decision_id:
            continue
        d_meta = d.get("metadata", {})
        score = 0.0
        if d_meta.get("category", "").lower() == category.lower() and category:
            score += 0.5
        d_words = set(d_meta.get("scenario", "").lower().split())
        if scenario_words and d_words:
            overlap = len(scenario_words & d_words) / max(len(scenario_words | d_words), 1)
            score += 0.5 * overlap
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [_node_to_decision(d) for _, d in scored[:limit]]


@router.get("/{decision_id}/compliance", response_model=ComplianceResponse)
async def check_compliance(
    decision_id: str,
    session: GraphSession = Depends(get_session),
):
    """
    Check policy compliance for a decision.

    Inspects edges of type ``violates``, ``non_compliant``, or ``breaches``
    originating from the decision node.  Returns ``compliant=True`` when no
    such edges are found, which is the correct result for graphs that have
    no policy-violation edges defined.
    """
    node = await asyncio.to_thread(session.get_node, decision_id)
    if node is None:
        raise KeyError(decision_id)

    edges, _ = await asyncio.to_thread(session.get_edges, skip=0, limit=999_999)

    _VIOLATION_TYPES = {"violates", "non_compliant", "breaches"}
    violation_edges = [
        e for e in edges
        if e.get("source") == decision_id and e.get("type") in _VIOLATION_TYPES
    ]

    violations = [
        {
            "policy_id": e.get("target"),
            "type": e.get("type"),
            "metadata": e.get("metadata", {}),
        }
        for e in violation_edges
    ]

    return ComplianceResponse(
        decision_id=decision_id,
        compliant=len(violations) == 0,
        violations=violations,
    )
