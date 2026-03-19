"""
Annotation routes — CRUD for collaborative annotations on graph nodes.

Annotations are stored in-memory on the ``GraphSession`` and do not
modify Semantica core.
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_session
from ..schemas import AnnotationCreate, AnnotationResponse
from ..session import GraphSession

router = APIRouter(prefix="/api/annotations", tags=["Annotations"])


@router.get("", response_model=list[AnnotationResponse])
async def list_annotations(
    node_id: Optional[str] = Query(None, description="Filter by node ID"),
    session: GraphSession = Depends(get_session),
):
    """List annotations, optionally filtered by node_id."""
    anns = await asyncio.to_thread(session.get_annotations, node_id)
    return [AnnotationResponse(**a) for a in anns]


@router.post("", response_model=AnnotationResponse, status_code=201)
async def create_annotation(
    body: AnnotationCreate,
    session: GraphSession = Depends(get_session),
):
    """Create a new annotation on a node."""
    node = await asyncio.to_thread(session.get_node, body.node_id)
    if node is None:
        raise KeyError(body.node_id)

    ann_data = body.model_dump()
    # add_annotation mutates ann_data in-place, adding annotation_id and created_at.
    await asyncio.to_thread(session.add_annotation, ann_data)
    return AnnotationResponse(**ann_data)


@router.delete("/{annotation_id}", status_code=204)
async def delete_annotation(
    annotation_id: str,
    session: GraphSession = Depends(get_session),
):
    """Delete an annotation by ID."""
    deleted = await asyncio.to_thread(session.delete_annotation, annotation_id)
    if not deleted:
        raise KeyError(annotation_id)
    return None
