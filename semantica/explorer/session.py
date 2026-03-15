"""
Semantica Explorer : Graph Session

Holds a loaded ContextGraph together with lazily-initialized analytics
components.  One session is created at server startup and shared across
all API requests via FastAPI's dependency injection.
"""

import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..context.context_graph import ContextGraph

_KG_AVAILABLE = False
try:
    from ..kg import (
        CentralityCalculator,
        CommunityDetector,
        ConnectivityAnalyzer,
        GraphValidator,
        LinkPredictor,
        NodeEmbedder,
        PathFinder,
        SimilarityCalculator,
    )
    _KG_AVAILABLE = True
except ImportError:
    pass


class GraphSession:
    """
    Holds a loaded graph and its associated analytics components.

    Thread safety: all mutations to the graph or the annotations store
    must go through methods on this class, which are protected by an
    ``RLock``.
    """

    def __init__(self, graph: ContextGraph) -> None:
        self.graph = graph
        self._lock = threading.RLock()


        self.annotations: Dict[str, Dict[str, Any]] = {}


        self._centrality: Any = None
        self._community: Any = None
        self._connectivity: Any = None
        self._path_finder: Any = None
        self._node_embedder: Any = None
        self._similarity: Any = None
        self._link_predictor: Any = None
        self._validator: Any = None


    @classmethod
    def from_file(cls, path: str) -> "GraphSession":
        """Load a ContextGraph from a JSON file and wrap it in a session."""
        graph = ContextGraph()
        graph.load_from_file(path)
        return cls(graph)


    @property
    def centrality(self) -> Any:
        if self._centrality is None and _KG_AVAILABLE:
            self._centrality = CentralityCalculator()
        return self._centrality

    @property
    def community(self) -> Any:
        if self._community is None and _KG_AVAILABLE:
            self._community = CommunityDetector()
        return self._community

    @property
    def connectivity(self) -> Any:
        if self._connectivity is None and _KG_AVAILABLE:
            self._connectivity = ConnectivityAnalyzer()
        return self._connectivity

    @property
    def path_finder(self) -> Any:
        if self._path_finder is None and _KG_AVAILABLE:
            self._path_finder = PathFinder()
        return self._path_finder

    @property
    def node_embedder(self) -> Any:
        if self._node_embedder is None and _KG_AVAILABLE:
            self._node_embedder = NodeEmbedder()
        return self._node_embedder

    @property
    def similarity(self) -> Any:
        if self._similarity is None and _KG_AVAILABLE:
            self._similarity = SimilarityCalculator()
        return self._similarity

    @property
    def link_predictor(self) -> Any:
        if self._link_predictor is None and _KG_AVAILABLE:
            self._link_predictor = LinkPredictor()
        return self._link_predictor

    @property
    def validator(self) -> Any:
        if self._validator is None and _KG_AVAILABLE:
            self._validator = GraphValidator()
        return self._validator


    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node by ID, or ``None``."""
        with self._lock:
            return self.graph.find_node(node_id)

    def get_nodes(
        self,
        node_type: Optional[str] = None,
        search: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return a paginated slice of nodes and the total count."""
        with self._lock:
            # We fetch all nodes if there is a search filter as we need to filter them in memory,
            # otherwise we let the context graph handle the pagination natively.
            if search:
                all_nodes = self.graph.find_nodes(node_type=node_type)
                search_lower = search.lower()
                all_nodes = [
                    n for n in all_nodes
                    if search_lower in n.get("id", "").lower()
                    or search_lower in n.get("content", "").lower()
                    or search_lower in str(n.get("metadata", {})).lower()
                ]
                total = len(all_nodes)
                page = all_nodes[skip: skip + limit]
            else:
                total = self.graph.stats().get("node_types", {}).get(node_type, 0) if node_type else self.graph.stats().get("node_count", 0)
                page = self.graph.find_nodes(node_type=node_type, skip=skip, limit=limit)
                
        return page, total

    def get_edges(
        self,
        edge_type: Optional[str] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return a paginated slice of edges and the total count."""
        with self._lock:
            if source or target:
                all_edges = self.graph.find_edges(edge_type=edge_type)
                if source:
                    all_edges = [e for e in all_edges if e.get("source") == source]
                if target:
                    all_edges = [e for e in all_edges if e.get("target") == target]
                total = len(all_edges)
                page = all_edges[skip: skip + limit]
            else:
                total = self.graph.stats().get("edge_types", {}).get(edge_type, 0) if edge_type else self.graph.stats().get("edge_count", 0)
                page = self.graph.find_edges(edge_type=edge_type, skip=skip, limit=limit)
                
        return page, total

    def get_neighbors(self, node_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        """Get neighbours for a node (BFS). Returns [] for unknown nodes."""
        with self._lock:
            return self.graph.get_neighbors(node_id, hops=depth)

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Keyword search across node content."""
        with self._lock:
            return self.graph.query(query)[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Graph-level statistics."""
        with self._lock:
            return self.graph.stats()

    def get_active_nodes(
        self, at_time: Optional[datetime] = None, node_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Nodes active at a given point in time."""
        with self._lock:
            return self.graph.find_active_nodes(node_type=node_type, at_time=at_time)


    def add_annotation(self, annotation: Dict[str, Any]) -> str:
        """Add an annotation and return its ID."""
        ann_id = str(uuid.uuid4())
        annotation["annotation_id"] = ann_id
        annotation["created_at"] = datetime.utcnow().isoformat()
        with self._lock:
            self.annotations[ann_id] = annotation
        return ann_id

    def get_annotations(self, node_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List annotations, optionally filtered by node_id."""
        with self._lock:
            anns = list(self.annotations.values())
        if node_id:
            anns = [a for a in anns if a.get("node_id") == node_id]
        return anns

    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete an annotation. Returns True if found and deleted."""
        with self._lock:
            return self.annotations.pop(annotation_id, None) is not None
