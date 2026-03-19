"""
Semantica Explorer — FastAPI Application Factory

Creates and configures the FastAPI app with CORS, error handling,
static file serving, route registration, and WebSocket support.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from .session import GraphSession
from .ws import ConnectionManager


def create_app(session: Optional[GraphSession] = None) -> FastAPI:
    """
    Build a fully-configured FastAPI application.

    Args:
        session: Pre-built ``GraphSession``.  If ``None`` the caller must
                 attach one to ``app.state.session`` before the first
                 request arrives.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if session is not None:
            app.state.session = session
        app.state.ws_manager = ConnectionManager()
        yield

    app = FastAPI(
        title="Semantica Knowledge Explorer",
        description="Interactive dashboard API for exploring Semantica knowledge graphs.",
        version=__version__,
        lifespan=lifespan,
    )

    cors_origins = os.environ.get("EXPLORER_CORS_ORIGINS", "*")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError):
        return JSONResponse(
            status_code=404,
            content={"detail": f"Not found: {exc}"},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        # Let FastAPI's built-in HTTPException handler take precedence so that
        # responses from dependency injection (e.g. 503 from get_session) are
        # not swallowed and converted to 500.
        if isinstance(exc, HTTPException):
            raise exc
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
        )

    from .routes.graph import router as graph_router
    from .routes.analytics import router as analytics_router
    from .routes.decisions import router as decisions_router
    from .routes.temporal import router as temporal_router
    from .routes.enrich import router as enrich_router
    from .routes.export_import import router as export_import_router
    from .routes.annotations import router as annotations_router

    app.include_router(graph_router)
    app.include_router(analytics_router)
    app.include_router(decisions_router)
    app.include_router(temporal_router)
    app.include_router(enrich_router)
    app.include_router(export_import_router)
    app.include_router(annotations_router)

    from fastapi import WebSocket, WebSocketDisconnect

    @app.websocket("/ws/graph-updates")
    async def websocket_endpoint(websocket: WebSocket):
        manager: ConnectionManager = app.state.ws_manager
        await manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    @app.get("/api/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/api/info")
    async def info():
        return {
            "name": "Semantica Knowledge Explorer",
            "version": __version__,
            "status": "active",
        }

    static_dir = Path(__file__).resolve().parent.parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
