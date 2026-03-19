"""
Graph routes — node / edge / path / search endpoints.
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_session
from ..schemas import (
    EdgeListResponse,
    EdgeResponse,
    GraphStatsResponse,
    NeighborResponse,
    NodeListResponse,
    NodeResponse,
    PathResponse,
    SearchRequest,
    SearchResultItem,
    SearchResultResponse,
)
from ..session import GraphSession

router = APIRouter(prefix="/api/graph", tags=["Graph"])


def _node_dict_to_response(n: dict) -> NodeResponse:
    """Convert a ContextGraph node dict to a NodeResponse."""
    meta = n.get("metadata", {})
    return NodeResponse(
        id=n.get("id", ""),
        type=n.get("type", "entity"),
        content=n.get("content", meta.get("content", "")),
        properties=meta,
        valid_from=meta.get("valid_from"),
        valid_until=meta.get("valid_until"),
    )



@router.get("/nodes", response_model=NodeListResponse)
async def list_nodes(
    type: Optional[str] = Query(None, description="Filter by node type"),
    search: Optional[str] = Query(None, description="Keyword search over node content"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    session: GraphSession = Depends(get_session),
):
    """List nodes with optional filtering and pagination."""
    nodes, total = await asyncio.to_thread(
        session.get_nodes, node_type=type, search=search, skip=skip, limit=limit
    )
    return NodeListResponse(
        nodes=[_node_dict_to_response(n) for n in nodes],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/node/{node_id}", response_model=NodeResponse)
async def get_node(
    node_id: str,
    session: GraphSession = Depends(get_session),
):
    """Get a single node by ID."""
    node = await asyncio.to_thread(session.get_node, node_id)
    if node is None:
        raise KeyError(node_id)
    return _node_dict_to_response(node)


@router.get("/node/{node_id}/neighbors", response_model=list[NeighborResponse])
async def get_neighbors(
    node_id: str,
    depth: int = Query(1, ge=1, le=5),
    session: GraphSession = Depends(get_session),
):
    """Get neighbours of a node via BFS traversal."""
    neighbors = await asyncio.to_thread(session.get_neighbors, node_id, depth)
    return [
        NeighborResponse(
            id=nb.get("id", ""),
            type=nb.get("type", ""),
            content=nb.get("content", ""),
            relationship=nb.get("relationship", ""),
            weight=nb.get("weight", 1.0),
            hop=nb.get("hop", 1),
        )
        for nb in neighbors
    ]




@router.get("/edges", response_model=EdgeListResponse)
async def list_edges(
    type: Optional[str] = Query(None, description="Filter by edge type"),
    source: Optional[str] = Query(None, description="Filter by source node ID"),
    target: Optional[str] = Query(None, description="Filter by target node ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    session: GraphSession = Depends(get_session),
):
    """List edges with optional filtering and pagination."""
    edges, total = await asyncio.to_thread(
        session.get_edges, edge_type=type, source=source, target=target, skip=skip, limit=limit
    )
    return EdgeListResponse(
        edges=[
            EdgeResponse(
                source=e.get("source", ""),
                target=e.get("target", ""),
                type=e.get("type", ""),
                weight=e.get("weight", 1.0),
                properties=e.get("metadata", {}),
            )
            for e in edges
        ],
        total=total,
        skip=skip,
        limit=limit,
    )




@router.get("/node/{node_id}/path", response_model=PathResponse)
async def find_path(
    node_id: str,
    target: str = Query(..., description="Target node ID"),
    algorithm: str = Query("bfs", description="Algorithm: bfs, dijkstra"),
    session: GraphSession = Depends(get_session),
):
    """Find a path between two nodes."""
    pf = session.path_finder
    if pf is None:
        raise ValueError("PathFinder not available — KG extras may not be installed.")

    graph_data = await asyncio.to_thread(session.build_graph_dict)

    # Select algorithm: dijkstra for weighted shortest path, bfs otherwise.
    if algorithm.lower() == "dijkstra":
        path_fn = pf.dijkstra_shortest_path
    else:
        path_fn = pf.bfs_shortest_path

    result = await asyncio.to_thread(path_fn, graph_data, node_id, target)

    path_nodes = result.get("path", []) if isinstance(result, dict) else (result or [])
    total_weight = result.get("total_weight", 0.0) if isinstance(result, dict) else 0.0

    return PathResponse(
        source=node_id,
        target=target,
        algorithm=algorithm,
        path=path_nodes,
        total_weight=total_weight,
    )



@router.post("/search", response_model=SearchResultResponse)
async def search_nodes(
    body: SearchRequest,
    session: GraphSession = Depends(get_session),
):
    """Keyword search over graph nodes."""
    results = await asyncio.to_thread(session.search, body.query, body.limit)
    items = [
        SearchResultItem(
            node=_node_dict_to_response(r.get("node", {})),
            score=r.get("score", 0.0),
        )
        for r in results
    ]
    return SearchResultResponse(results=items, total=len(items), query=body.query)



@router.get("/stats", response_model=GraphStatsResponse)
async def graph_stats(
    session: GraphSession = Depends(get_session),
):
    """Get graph-level statistics."""
    stats = await asyncio.to_thread(session.get_stats)
    return GraphStatsResponse(**stats)



