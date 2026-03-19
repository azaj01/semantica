"""
Integration tests for the Semantica Knowledge Explorer API.

Uses FastAPI's TestClient (from starlette.testclient).
"""

import json
import pytest

from semantica.context.context_graph import ContextGraph
from semantica.explorer.app import create_app
from semantica.explorer.session import GraphSession


try:
    from starlette.testclient import TestClient
except ImportError:
    pytest.skip(
        "starlette (TestClient) is required for explorer tests. "
        "Install with: pip install semantica[explorer]",
        allow_module_level=True,
    )


def _build_sample_graph() -> ContextGraph:
    """Create a small ContextGraph with a handful of nodes and edges."""
    g = ContextGraph(advanced_analytics=False)

    g.add_node("python", node_type="language", content="Python programming language",
               popularity="high")
    g.add_node("javascript", node_type="language", content="JavaScript programming language")
    g.add_node("web_dev", node_type="concept", content="Web Development")
    g.add_node("ml", node_type="concept", content="Machine Learning")
    g.add_node("decision_1", node_type="decision", content="Approve ML framework",
               category="tech", scenario="Choosing ML framework", outcome="approved",
               confidence="0.9", reasoning="Best performance")
    g.add_node("decision_2", node_type="decision", content="Reject legacy stack",
               category="tech", scenario="Choosing ML framework alternative",
               outcome="rejected", confidence="0.4", reasoning="Outdated")
    g.add_node("temporal_node", node_type="event", content="Conference talk",
               valid_from="2025-01-01T00:00:00", valid_until="2025-12-31T23:59:59")

    g.add_edge("python", "ml", edge_type="used_in", weight=0.9)
    g.add_edge("javascript", "web_dev", edge_type="used_in", weight=0.8)
    g.add_edge("python", "web_dev", edge_type="used_in", weight=0.5)
    g.add_edge("decision_1", "ml", edge_type="about")

    return g


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient backed by a sample graph."""
    graph = _build_sample_graph()
    session = GraphSession(graph)
    app = create_app(session=session)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health & Info
# ---------------------------------------------------------------------------

class TestHealthInfo:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_info(self, client):
        r = client.get("/api/info")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "Semantica Knowledge Explorer"
        assert "version" in body
        assert body["status"] == "active"


# ---------------------------------------------------------------------------
# Graph — Nodes
# ---------------------------------------------------------------------------

class TestGraphNodes:
    def test_list_nodes(self, client):
        r = client.get("/api/graph/nodes")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 5
        assert len(body["nodes"]) <= body["total"]
        assert "skip" in body and "limit" in body

    def test_list_nodes_pagination(self, client):
        r = client.get("/api/graph/nodes?skip=0&limit=2")
        assert r.status_code == 200
        body = r.json()
        assert len(body["nodes"]) == 2
        assert body["limit"] == 2

    def test_list_nodes_filter_type(self, client):
        r = client.get("/api/graph/nodes?type=language")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 2
        assert all(n["type"] == "language" for n in body["nodes"])

    def test_list_nodes_search(self, client):
        r = client.get("/api/graph/nodes?search=python")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 1
        assert any("python" in n["id"].lower() or "python" in n["content"].lower()
                   for n in body["nodes"])

    def test_get_node(self, client):
        r = client.get("/api/graph/node/python")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == "python"
        assert body["type"] == "language"
        assert "content" in body

    def test_get_node_not_found(self, client):
        r = client.get("/api/graph/node/nonexistent_xyz")
        assert r.status_code == 404

    def test_get_neighbors(self, client):
        r = client.get("/api/graph/node/python/neighbors")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        ids = [nb["id"] for nb in body]
        assert "ml" in ids or "web_dev" in ids
        # Each neighbour must have required fields
        for nb in body:
            assert "id" in nb and "type" in nb and "hop" in nb

    def test_get_neighbors_depth(self, client):
        r = client.get("/api/graph/node/python/neighbors?depth=2")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# Graph — Edges
# ---------------------------------------------------------------------------

class TestGraphEdges:
    def test_list_edges(self, client):
        r = client.get("/api/graph/edges")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 3
        assert all("source" in e and "target" in e and "type" in e
                   for e in body["edges"])

    def test_list_edges_filter_type(self, client):
        r = client.get("/api/graph/edges?type=used_in")
        assert r.status_code == 200
        body = r.json()
        assert len(body["edges"]) >= 1
        assert all(e["type"] == "used_in" for e in body["edges"])

    def test_list_edges_filter_source(self, client):
        r = client.get("/api/graph/edges?source=python")
        assert r.status_code == 200
        body = r.json()
        assert len(body["edges"]) >= 1
        assert all(e["source"] == "python" for e in body["edges"])

    def test_list_edges_filter_target(self, client):
        r = client.get("/api/graph/edges?target=ml")
        assert r.status_code == 200
        body = r.json()
        assert all(e["target"] == "ml" for e in body["edges"])


# ---------------------------------------------------------------------------
# Search & Stats
# ---------------------------------------------------------------------------

class TestSearchStats:
    def test_search(self, client):
        r = client.post("/api/graph/search", json={"query": "programming", "limit": 5})
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "programming"
        assert len(body["results"]) >= 1
        # Each result must carry a node and a score
        for item in body["results"]:
            assert "node" in item and "score" in item
            assert item["node"]["id"]  # non-empty id

    def test_search_no_results(self, client):
        r = client.post("/api/graph/search", json={"query": "zzznomatchzzz"})
        assert r.status_code == 200
        assert r.json()["total"] == 0

    def test_stats(self, client):
        r = client.get("/api/graph/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["node_count"] >= 5
        assert body["edge_count"] >= 3
        assert "density" in body
        assert "node_types" in body and "edge_types" in body
        assert body["density"] >= 0.0


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

class TestDecisions:
    def test_list_decisions(self, client):
        r = client.get("/api/decisions")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        for d in body:
            assert "decision_id" in d and "category" in d

    def test_list_decisions_category(self, client):
        r = client.get("/api/decisions?category=tech")
        assert r.status_code == 200
        body = r.json()
        assert len(body) >= 1
        assert all(d["category"] == "tech" for d in body)

    def test_get_decision(self, client):
        r = client.get("/api/decisions/decision_1")
        assert r.status_code == 200
        body = r.json()
        assert body["decision_id"] == "decision_1"
        assert body["outcome"] == "approved"

    def test_get_decision_not_found(self, client):
        r = client.get("/api/decisions/nope")
        assert r.status_code == 404

    def test_causal_chain(self, client):
        r = client.get("/api/decisions/decision_1/chain")
        assert r.status_code == 200
        body = r.json()
        assert body["decision_id"] == "decision_1"
        assert isinstance(body["chain"], list)

    def test_precedents(self, client):
        r = client.get("/api/decisions/decision_1/precedents")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        # decision_2 shares category "tech" so should appear
        ids = [d["decision_id"] for d in body]
        assert "decision_2" in ids

    def test_compliance_no_violations(self, client):
        """Graph has no violation edges so compliance must be True."""
        r = client.get("/api/decisions/decision_1/compliance")
        assert r.status_code == 200
        body = r.json()
        assert "compliant" in body and "violations" in body
        assert body["compliant"] is True
        assert isinstance(body["violations"], list)

    def test_compliance_with_violation(self, client):
        """Add a violation edge then check compliance detects it."""
        # Add a policy node and a violates edge directly on the underlying graph
        session = GraphSession(client.app.state.session.graph)
        session.graph.add_node("policy_1", node_type="policy", content="Data policy")
        session.graph.add_edge("decision_1", "policy_1", edge_type="violates")

        r = client.get("/api/decisions/decision_1/compliance")
        assert r.status_code == 200
        body = r.json()
        assert body["compliant"] is False
        assert len(body["violations"]) >= 1
        assert body["violations"][0]["policy_id"] == "policy_1"

        # Clean up — remove the test edge so other tests are not affected
        # (ContextGraph doesn't expose edge removal; re-create the session fixture
        #  to isolate: this test intentionally runs after all other decision tests)


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------

class TestTemporal:
    def test_snapshot_now(self, client):
        r = client.get("/api/temporal/snapshot")
        assert r.status_code == 200
        body = r.json()
        assert "active_node_count" in body
        assert "timestamp" in body
        assert isinstance(body["active_nodes"], list)

    def test_snapshot_at_includes_temporal_node(self, client):
        r = client.get("/api/temporal/snapshot?at=2025-06-15T00:00:00")
        assert r.status_code == 200
        body = r.json()
        ids = [n["id"] for n in body["active_nodes"]]
        assert "temporal_node" in ids

    def test_snapshot_at_excludes_temporal_node(self, client):
        r = client.get("/api/temporal/snapshot?at=2026-01-01T00:00:00")
        assert r.status_code == 200
        body = r.json()
        ids = [n["id"] for n in body["active_nodes"]]
        assert "temporal_node" not in ids

    def test_diff(self, client):
        r = client.get(
            "/api/temporal/diff"
            "?from_time=2024-01-01T00:00:00"
            "&to_time=2025-06-15T00:00:00"
        )
        assert r.status_code == 200
        body = r.json()
        assert "added_nodes" in body and "removed_nodes" in body
        # temporal_node became active between t1 and t2
        assert "temporal_node" in body["added_nodes"]

    def test_patterns(self, client):
        r = client.get("/api/temporal/patterns")
        assert r.status_code == 200
        body = r.json()
        assert "patterns" in body
        assert isinstance(body["patterns"], list)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class TestAnalytics:
    def test_analytics_response_shape(self, client):
        r = client.get("/api/analytics")
        assert r.status_code == 200
        body = r.json()
        # At least one analytics key must be present (or all None when KG extras absent)
        assert set(body.keys()) >= {"centrality", "community", "connectivity"} or \
               all(v is None for v in body.values())

    def test_analytics_select_metric(self, client):
        r = client.get("/api/analytics?metrics=centrality")
        assert r.status_code == 200
        # When KG extras are absent the key is still present but None; otherwise a dict
        body = r.json()
        assert "centrality" in body

    def test_validation(self, client):
        r = client.get("/api/analytics/validation")
        assert r.status_code == 200
        body = r.json()
        assert "valid" in body
        assert "error_count" in body and "warning_count" in body
        assert isinstance(body["issues"], list)


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------

class TestReasoning:
    def test_reason_forward(self, client):
        r = client.post(
            "/api/reason",
            json={
                "facts": ["Person(Alice)", "Knows(Alice, Bob)"],
                "rules": ["IF Knows(?x, ?y) THEN Connected(?x, ?y)"],
                "mode": "forward",
            },
        )
        # 200 when reasoning module is available; 422 when it's not installed.
        # Either is acceptable — the endpoint must not 500.
        assert r.status_code in (200, 422), (
            f"Unexpected status {r.status_code}: {r.text}"
        )
        if r.status_code == 200:
            body = r.json()
            assert "inferred_facts" in body
            assert "rules_fired" in body
            assert isinstance(body["inferred_facts"], list)
            assert isinstance(body["rules_fired"], int)

    def test_reason_empty_rules(self, client):
        r = client.post(
            "/api/reason",
            json={"facts": ["Person(Alice)"], "rules": [], "mode": "forward"},
        )
        assert r.status_code in (200, 422)
        if r.status_code == 200:
            assert r.json()["rules_fired"] == 0


# ---------------------------------------------------------------------------
# Enrichment — Extract
# ---------------------------------------------------------------------------

class TestEnrichExtract:
    def test_extract_returns_structure(self, client):
        r = client.post(
            "/api/enrich/extract",
            json={"text": "Alice works at Acme Corp in New York."},
        )
        # 200 when spacy/transformers available; 422 otherwise
        assert r.status_code in (200, 422), (
            f"Unexpected status {r.status_code}: {r.text}"
        )
        if r.status_code == 200:
            body = r.json()
            assert "entities" in body and "relations" in body
            assert isinstance(body["entities"], list)
            assert isinstance(body["relations"], list)

    def test_extract_empty_text(self, client):
        r = client.post("/api/enrich/extract", json={"text": ""})
        assert r.status_code in (200, 422)


# ---------------------------------------------------------------------------
# Enrichment — Link Prediction
# ---------------------------------------------------------------------------

class TestLinkPrediction:
    def test_predict_links_known_node(self, client):
        r = client.post(
            "/api/enrich/links",
            json={"node_id": "python", "top_n": 5},
        )
        # 200 when KG extras available; 422 otherwise
        assert r.status_code in (200, 422), (
            f"Unexpected status {r.status_code}: {r.text}"
        )
        if r.status_code == 200:
            body = r.json()
            assert body["node_id"] == "python"
            assert isinstance(body["predictions"], list)
            # Must not include existing neighbours
            neighbour_ids = {"ml", "web_dev"}
            for pred in body["predictions"]:
                assert pred.get("target") not in neighbour_ids

    def test_predict_links_unknown_node(self, client):
        r = client.post(
            "/api/enrich/links",
            json={"node_id": "does_not_exist", "top_n": 5},
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Enrichment — Deduplication
# ---------------------------------------------------------------------------

class TestDedup:
    def test_dedup_returns_structure(self, client):
        r = client.post("/api/enrich/dedup", json={"threshold": 0.8})
        assert r.status_code in (200, 422), (
            f"Unexpected status {r.status_code}: {r.text}"
        )
        if r.status_code == 200:
            body = r.json()
            assert "duplicates" in body and "total_flagged" in body
            assert isinstance(body["duplicates"], list)
            assert body["total_flagged"] == len(body["duplicates"])


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

class TestAnnotations:
    def test_create_and_list(self, client):
        r = client.post(
            "/api/annotations",
            json={"node_id": "python", "content": "Great language!", "tags": ["fav"]},
        )
        assert r.status_code == 201
        ann = r.json()
        assert ann["node_id"] == "python"
        assert ann["content"] == "Great language!"
        assert "annotation_id" in ann and ann["annotation_id"]
        assert "created_at" in ann
        ann_id = ann["annotation_id"]

        # List all
        r = client.get("/api/annotations")
        assert r.status_code == 200
        assert any(a["annotation_id"] == ann_id for a in r.json())

        # List filtered by node
        r = client.get("/api/annotations?node_id=python")
        assert r.status_code == 200
        assert all(a["node_id"] == "python" for a in r.json())

        # Delete
        r = client.delete(f"/api/annotations/{ann_id}")
        assert r.status_code == 204

        # Verify gone
        r = client.get("/api/annotations")
        assert all(a["annotation_id"] != ann_id for a in r.json())

    def test_create_annotation_bad_node(self, client):
        r = client.post(
            "/api/annotations",
            json={"node_id": "nonexistent", "content": "oops"},
        )
        assert r.status_code == 404

    def test_delete_annotation_not_found(self, client):
        r = client.delete("/api/annotations/no_such_id")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_json(self, client):
        r = client.post("/api/export", json={"format": "json"})
        assert r.status_code == 200
        ct = r.headers.get("content-type", "")
        assert "json" in ct.lower()
        # Payload must be valid JSON with entities and relationships
        data = r.json()
        assert "entities" in data and "relationships" in data
        assert len(data["entities"]) >= 5

    def test_export_json_subset(self, client):
        r = client.post("/api/export", json={"format": "json", "node_ids": ["python", "ml"]})
        assert r.status_code == 200
        data = r.json()
        ids = [e["id"] for e in data["entities"]]
        assert set(ids) == {"python", "ml"}

    def test_export_unsupported(self, client):
        r = client.post("/api/export", json={"format": "pdf"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

class TestImport:
    def test_import_json(self, client):
        payload = json.dumps({
            "nodes": [
                {"id": "imported_node", "type": "test", "properties": {"content": "hello"}}
            ],
            "edges": [],
        })
        r = client.post(
            "/api/import",
            files={"file": ("import.json", payload, "application/json")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["nodes_added"] >= 1

        # Verify the node is now in the graph
        r2 = client.get("/api/graph/node/imported_node")
        assert r2.status_code == 200
        assert r2.json()["id"] == "imported_node"

    def test_import_with_edges(self, client):
        payload = json.dumps({
            "nodes": [
                {"id": "import_src", "type": "test", "properties": {"content": "src"}},
                {"id": "import_tgt", "type": "test", "properties": {"content": "tgt"}},
            ],
            "edges": [
                {"source": "import_src", "target": "import_tgt", "type": "links_to"}
            ],
        })
        r = client.post(
            "/api/import",
            files={"file": ("import2.json", payload, "application/json")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["nodes_added"] >= 2

    def test_import_unsupported_format(self, client):
        r = client.post(
            "/api/import",
            files={"file": ("data.csv", b"a,b,c", "text/csv")},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "unsupported"
