"""
Export & import routes.
"""

import asyncio
import json
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import Response

from ..dependencies import get_session, get_ws_manager
from ..schemas import ExportRequest
from ..session import GraphSession
from ..ws import ConnectionManager

router = APIRouter(tags=["Export / Import"])


_FORMAT_MAP = {
    "json": ("export_json", "application/json", ".json"),
    "json-ld": ("export_json", "application/ld+json", ".jsonld"),
    "turtle": ("export_rdf", "text/turtle", ".ttl"),
    "rdf-xml": ("export_rdf", "application/rdf+xml", ".rdf"),
    "n-triples": ("export_rdf", "application/n-triples", ".nt"),
    "csv": ("export_csv", "text/csv", ".csv"),
    "graphml": ("export_graph", "application/xml", ".graphml"),
    "gexf": ("export_graph", "application/xml", ".gexf"),
    "owl": ("export_owl", "application/rdf+xml", ".owl"),
    "cypher": ("export_lpg", "text/plain", ".cypher"),
    "aql": ("export_arango", "text/plain", ".aql"),
    "yaml": ("export_yaml", "text/yaml", ".yaml"),
}




@router.post("/api/export")
async def export_graph(
    body: ExportRequest,
    session: GraphSession = Depends(get_session),
):
    """Export the current graph in the requested format."""
    fmt = body.format.lower()
    if fmt not in _FORMAT_MAP:
        raise ValueError(
            f"Unsupported format '{fmt}'. Supported: {', '.join(sorted(_FORMAT_MAP))}"
        )

    func_name, content_type, ext = _FORMAT_MAP[fmt]

    kg = await asyncio.to_thread(session.build_graph_dict, body.node_ids)

    try:
        from ...export.methods import (
            export_json, export_rdf, export_csv, export_graph as export_graph_fn,
            export_owl, export_lpg, export_arango, export_yaml,
        )
        fn_map = {
            "export_json": export_json,
            "export_rdf": export_rdf,
            "export_csv": export_csv,
            "export_graph": export_graph_fn,
            "export_owl": export_owl,
            "export_lpg": export_lpg,
            "export_arango": export_arango,
            "export_yaml": export_yaml,
        }
        export_fn = fn_map.get(func_name)
        if export_fn is None:
            raise ValueError(f"Export function {func_name} not found.")

        # Write to a temp file; always clean up even if export or read fails.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="w") as tmp:
                tmp_path = tmp.name

            await asyncio.to_thread(export_fn, kg, tmp_path)

            with open(tmp_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except ImportError:
        content = json.dumps(kg, indent=2, default=str)
        content_type = "application/json"
        ext = ".json"

    filename = f"semantica_export{ext}"
    return Response(
        content=content,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/api/import")
async def import_file(
    file: UploadFile = File(...),
    session: GraphSession = Depends(get_session),
    ws: ConnectionManager = Depends(get_ws_manager),
):
    """
    Import entities from an uploaded file (JSON or CSV).

    For JSON files the expected shape is ``{"nodes": [...], "edges": [...]}``.
    """
    content = await file.read()
    filename = file.filename or "upload"

    await ws.broadcast("import_started", {"filename": filename})

    try:
        if filename.endswith(".json") or filename.endswith(".jsonld"):
            data = json.loads(content)
            nodes = data.get("nodes", data.get("entities", []))
            edges = data.get("edges", data.get("relationships", []))

            for edge in edges:
                if "source" in edge and "source_id" not in edge:
                    edge["source_id"] = edge["source"]
                if "target" in edge and "target_id" not in edge:
                    edge["target_id"] = edge["target"]

            added_nodes = await asyncio.to_thread(session.add_nodes, nodes)
            added_edges = await asyncio.to_thread(session.add_edges, edges)

            result = {
                "status": "success",
                "nodes_added": added_nodes,
                "edges_added": added_edges,
            }
        else:
            result = {
                "status": "unsupported",
                "detail": f"File type not supported yet: {filename}",
            }
    except Exception as exc:
        result = {"status": "error", "detail": str(exc)}

    await ws.broadcast("import_completed", result)
    return result
