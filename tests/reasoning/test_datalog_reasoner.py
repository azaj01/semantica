"""
Test suite for the DatalogReasoner module.
"""

import pytest
from typing import List, Dict, Any

from semantica.reasoning.datalog_reasoner import DatalogReasoner, DatalogFact

# fixtures and mocks

@pytest.fixture
def reasoner():
    """Provides a fresh DatalogReasoner instance for each test."""
    return DatalogReasoner()

class MockContextGraph:
    """A simple mock to simulate Semantica's ContextGraph for testing."""
    def __init__(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
        self._nodes = nodes
        self._edges = edges

    def nodes(self):
        return self._nodes

    def edges(self):
        return self._edges

# Test suite

class TestBasicFacts:
    def test_add_string_fact(self, reasoner):
        reasoner.add_fact("parent(tom, bob)")
        assert len(reasoner._all_facts) == 1
        fact = list(reasoner._all_facts)[0]
        assert fact.predicate == "parent"
        assert fact.args == ("tom", "bob")

    def test_add_dict_fact(self, reasoner):
        reasoner.add_fact({"subject": "bob", "predicate": "parent", "object": "ann"})
        assert len(reasoner._all_facts) == 1
        fact = list(reasoner._all_facts)[0]
        assert fact.predicate == "parent"
        assert fact.args == ("bob", "ann")

    def test_duplicate_fact_ignored(self, reasoner):
        reasoner.add_fact("parent(tom, bob)")
        reasoner.add_fact("parent(tom, bob)")
        assert len(reasoner._all_facts) == 1


class TestRules:
    def test_single_rule(self, reasoner):
        reasoner.add_fact("parent(tom, bob)")
        reasoner.add_rule("ancestor(X, Y) :- parent(X, Y).")
        
        derived = reasoner.derive_all()
        assert "ancestor(tom, bob)" in derived

    def test_recursive_ancestor(self, reasoner):
        reasoner.add_fact("parent(tom, bob)")
        reasoner.add_fact("parent(bob, ann)")
        
        reasoner.add_rule("ancestor(X, Y) :- parent(X, Y).")
        reasoner.add_rule("ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).")
        
        derived = reasoner.derive_all()
        assert "ancestor(tom, bob)" in derived
        assert "ancestor(bob, ann)" in derived
        assert "ancestor(tom, ann)" in derived  

    def test_multi_hop_three_levels(self, reasoner):
        reasoner.add_fact("edge(1, 2)")
        reasoner.add_fact("edge(2, 3)")
        reasoner.add_fact("edge(3, 4)")
        
        reasoner.add_rule("reachable(X, Y) :- edge(X, Y).")
        reasoner.add_rule("reachable(X, Y) :- edge(X, Z), reachable(Z, Y).")
        
        derived = reasoner.derive_all()
        assert "reachable(1, 4)" in derived

    def test_two_body_atoms(self, reasoner):
        reasoner.add_fact("parent(tom, bob)")
        reasoner.add_fact("parent(bob, ann)")
        

        reasoner.add_rule("grandparent(X, Y) :- parent(X, Z), parent(Z, Y).")
        
        derived = reasoner.derive_all()
        assert "grandparent(tom, ann)" in derived
        assert "grandparent(tom, bob)" not in derived


class TestQuery:
    def test_variable_binding(self, reasoner):
        reasoner.add_fact("parent(tom, bob)")
        reasoner.add_fact("parent(tom, alex)")
        reasoner.add_rule("ancestor(X, Y) :- parent(X, Y).")
        
        results = reasoner.query("ancestor(tom, ?Y)")
        y_bindings = sorted([res["Y"] for res in results])
        assert y_bindings == ["alex", "bob"]

    def test_pre_bound_variable(self, reasoner):
        reasoner.add_fact("parent(tom, bob)")
        reasoner.add_rule("ancestor(X, Y) :- parent(X, Y).")
        

        results_bob = reasoner.query("ancestor(tom, ?Y)", bindings={"Y": "bob"})
        assert len(results_bob) == 1
        assert results_bob[0]["Y"] == "bob"
        
        results_ann = reasoner.query("ancestor(tom, ?Y)", bindings={"Y": "ann"})
        assert len(results_ann) == 0

    def test_no_match_returns_empty(self, reasoner):
        reasoner.add_fact("parent(tom, bob)")
        results = reasoner.query("parent(sarah, ?Y)")
        assert results == []


class TestContextGraphIntegration:
    def test_load_from_graph(self, reasoner):
        graph = MockContextGraph(
            nodes=[{"id": "microsoft", "type": "company"}],
            edges=[{"source": "microsoft", "target": "openai", "type": "invested_in"}]
        )
        
        added = reasoner.load_from_graph(graph)
        assert added == 2 
        
        assert DatalogFact("company", ("microsoft",)) in reasoner._all_facts
        assert DatalogFact("invested_in", ("microsoft", "openai")) in reasoner._all_facts

    def test_edge_becomes_fact(self, reasoner):
        graph = MockContextGraph(
            nodes=[],
            edges=[{"source_id": "a", "target_id": "b", "relation": "connected_to"}]
        )
        reasoner.load_from_graph(graph)
        assert DatalogFact("connected_to", ("a", "b")) in reasoner._all_facts

    def test_derive_after_load(self, reasoner):
        graph = MockContextGraph(
            nodes=[],
            edges=[
                {"source": "node_a", "target": "node_b", "type": "linked"},
                {"source": "node_b", "target": "node_c", "type": "linked"}
            ]
        )
        reasoner.load_from_graph(graph)
        reasoner.add_rule("path(X, Y) :- linked(X, Y).")
        reasoner.add_rule("path(X, Y) :- linked(X, Z), path(Z, Y).")
        
        derived = reasoner.derive_all()
        assert "path(node_a, node_c)" in derived


class TestEdgeCases:
    def test_empty_program(self, reasoner):
        derived = reasoner.derive_all()
        assert derived == []

    def test_derive_all_idempotent(self, reasoner):
        reasoner.add_fact("parent(tom, bob)")
        reasoner.add_rule("ancestor(X, Y) :- parent(X, Y).")
        
        first_run = len(reasoner.derive_all())
        second_run = len(reasoner.derive_all())
        
        assert first_run == second_run
        assert first_run == 2  

    def test_clear_resets_state(self, reasoner):
        reasoner.add_fact("parent(tom, bob)")
        reasoner.add_rule("ancestor(X, Y) :- parent(X, Y).")
        reasoner.derive_all()
        
        reasoner.clear()
        
        assert len(reasoner._all_facts) == 0
        assert len(reasoner._rules) == 0
        assert len(reasoner._delta_new) == 0
        assert len(reasoner._delta_old) == 0
        assert len(reasoner._fact_index) == 0