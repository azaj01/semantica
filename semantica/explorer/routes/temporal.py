"""
Temporal routes — snapshot, diff, patterns.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_session
from ..schemas import (
    NodeResponse,
    TemporalDiffResponse,
    TemporalPatternResponse,
    TemporalSnapshotResponse,
)
from ..session import GraphSession

router = APIRouter(prefix="/api/temporal", tags=["Temporal"])


def _node_dict_to_response(n: dict) -> NodeResponse:
    meta = n.get("metadata", {})
    return NodeResponse(
        id=n.get("id", ""),
        type=n.get("type", "entity"),
        content=n.get("content", meta.get("content", "")),
        properties=meta,
        valid_from=meta.get("valid_from"),
        valid_until=meta.get("valid_until"),
    )


@router.get("/snapshot", response_model=TemporalSnapshotResponse)
async def temporal_snapshot(
    at: Optional[str] = Query(
        None, description="ISO-8601 datetime; defaults to now."
    ),
    session: GraphSession = Depends(get_session),
):
    """
    Return the graph as it existed at a given timestamp.

    Only nodes whose ``valid_from`` / ``valid_until`` window includes
    the requested time are returned.
    """
    if at:
        ts_str = at.replace("Z", "+00:00")
        at_time = datetime.fromisoformat(ts_str)
    else:
        at_time = datetime.now(timezone.utc)

    active = await asyncio.to_thread(session.get_active_nodes, at_time=at_time)
    return TemporalSnapshotResponse(
        timestamp=at_time.isoformat(),
        active_nodes=[_node_dict_to_response(n) for n in active],
        active_node_count=len(active),
    )


@router.get("/diff", response_model=TemporalDiffResponse)
async def temporal_diff(
    from_time: str = Query(..., description="Start ISO-8601 datetime"),
    to_time: str = Query(..., description="End ISO-8601 datetime"),
    session: GraphSession = Depends(get_session),
):
    """
    Diff the graph between two points in time.

    Returns node IDs that were added (active at ``to_time`` but not
    ``from_time``) and removed (active at ``from_time`` but not
    ``to_time``).
    """
    t1 = datetime.fromisoformat(from_time.replace("Z", "+00:00"))
    t2 = datetime.fromisoformat(to_time.replace("Z", "+00:00"))

    active_t1 = await asyncio.to_thread(session.get_active_nodes, at_time=t1)
    active_t2 = await asyncio.to_thread(session.get_active_nodes, at_time=t2)

    ids_t1 = {n.get("id") for n in active_t1}
    ids_t2 = {n.get("id") for n in active_t2}

    return TemporalDiffResponse(
        from_time=t1.isoformat(),
        to_time=t2.isoformat(),
        added_nodes=sorted(ids_t2 - ids_t1),
        removed_nodes=sorted(ids_t1 - ids_t2),
    )


@router.get("/patterns", response_model=TemporalPatternResponse)
async def temporal_patterns(
    session: GraphSession = Depends(get_session),
):
    """
    Detect temporal patterns (trends, cycles, anomalies).

    Falls back to a stub when ``TemporalPatternDetector`` is not available.
    """
    try:
        from ...kg import TemporalPatternDetector
        detector = TemporalPatternDetector()


        nodes, _ = await asyncio.to_thread(session.get_nodes, skip=0, limit=999_999)
        edges, _ = await asyncio.to_thread(session.get_edges, skip=0, limit=999_999)
        graph_dict = {
            "entities": [
                {"id": n.get("id"), "type": n.get("type"), "metadata": n.get("metadata", {})}
                for n in nodes
            ],
            "relationships": [
                {"source": e.get("source"), "target": e.get("target"),
                 "type": e.get("type"), "metadata": e.get("metadata", {})}
                for e in edges
            ],
        }

        patterns = await asyncio.to_thread(detector.detect_patterns, graph_dict)
        if isinstance(patterns, dict):
            patterns = patterns.get("patterns", [])
        return TemporalPatternResponse(patterns=patterns if isinstance(patterns, list) else [])
    except ImportError:
        # TemporalPatternDetector is an optional KG extra; return empty gracefully.
        return TemporalPatternResponse(patterns=[])
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("temporal_patterns failed: %s", exc, exc_info=True)
        return TemporalPatternResponse(patterns=[])
