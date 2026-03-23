import unittest
import warnings
from unittest.mock import MagicMock, patch
import sys
import os
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from semantica.kg.graph_builder import GraphBuilder
from semantica.kg.graph_analyzer import GraphAnalyzer
from semantica.kg.temporal_model import TemporalBound, deserialize_relationship_temporal_fields
from semantica.utils.exceptions import TemporalValidationError

class TestGraphBuilder(unittest.TestCase):
    def setUp(self):
        # Patch where it is defined since it is imported inside __init__
        self.mock_tracker_patcher = patch("semantica.utils.progress_tracker.get_progress_tracker")
        self.mock_get_tracker = self.mock_tracker_patcher.start()
        self.mock_tracker = MagicMock()
        self.mock_get_tracker.return_value = self.mock_tracker

        self.mock_resolver_patcher = patch("semantica.kg.entity_resolver.EntityResolver")
        self.mock_resolver_cls = self.mock_resolver_patcher.start()
        
        self.mock_conflict_patcher = patch("semantica.conflicts.conflict_detector.ConflictDetector")
        self.mock_conflict_cls = self.mock_conflict_patcher.start()

    def tearDown(self):
        self.mock_tracker_patcher.stop()
        self.mock_resolver_patcher.stop()
        self.mock_conflict_patcher.stop()

    def test_initialization_defaults(self):
        """Test initialization with default parameters"""
        builder = GraphBuilder()
        self.assertFalse(builder.merge_entities)
        self.assertTrue(builder.resolve_conflicts)
        self.assertFalse(builder.enable_temporal)
        # Should initialize resolver and conflict detector by default
        self.assertIsNone(builder.entity_resolver)
        self.assertIsNotNone(builder.conflict_detector)

    def test_initialization_disabled_features(self):
        """Test initialization with features disabled"""
        builder = GraphBuilder(merge_entities=False, resolve_conflicts=False)
        self.assertFalse(builder.merge_entities)
        self.assertFalse(builder.resolve_conflicts)
        self.assertIsNone(builder.entity_resolver)
        self.assertIsNone(builder.conflict_detector)

    def test_build_simple(self):
        """Test building a simple graph"""
        builder = GraphBuilder(merge_entities=False, resolve_conflicts=False)
        
        sources = [
            {
                "entities": [{"id": "1", "name": "A"}, {"id": "2", "name": "B"}],
                "relationships": [{"source": "1", "target": "2", "type": "rel"}]
            }
        ]
        
        # We need to mock what happens inside build. 
        # The current implementation of build seems to just extract and return lists 
        # (based on the truncated read I did earlier, it seemed to just extend lists)
        # Let's see if it does more processing. 
        # Assuming it returns a dict with entities and relationships.
        
        graph = builder.build(sources)
        
        self.assertIn("entities", graph)
        self.assertIn("relationships", graph)
        self.assertEqual(len(graph["entities"]), 2)
        self.assertEqual(len(graph["relationships"]), 1)
        self.assertIn("metadata", graph)

    def test_build_format_handling(self):
        """Test building from different source formats"""
        builder = GraphBuilder(merge_entities=False, resolve_conflicts=False)
        
        # Single dict source
        source_dict = {
            "entities": [{"id": "1"}],
            "relationships": []
        }
        graph1 = builder.build(source_dict)
        self.assertEqual(len(graph1["entities"]), 1)
        
        # List of dicts
        source_list = [
            {"entities": [{"id": "1"}]},
            {"entities": [{"id": "2"}]}
        ]
        graph2 = builder.build(source_list)
        self.assertEqual(len(graph2["entities"]), 2)

    def test_build_with_external_relationship_ids(self):
        builder = GraphBuilder(merge_entities=False, resolve_conflicts=False)

        entities = [
            {"id": "1", "name": "A"},
            {"id": "2", "name": "B"},
        ]
        relationships = [
            {"source_id": "1", "target_id": "2", "type": "rel"},
        ]

        source = {
            "entities": entities,
            "relationships": relationships,
        }

        graph = builder.build(source)

        self.assertEqual(len(graph["entities"]), 2)
        self.assertEqual(len(graph["relationships"]), 1)
        rel = graph["relationships"][0]
        self.assertEqual(rel.get("source"), "1")
        self.assertEqual(rel.get("target"), "2")

    def test_build_with_conflict_resolution(self):
        """Test building with conflict resolution enabled"""
        builder = GraphBuilder(resolve_conflicts=True)
        
        # Mock conflict detector methods
        self.mock_conflict_cls.return_value.detect_conflicts.return_value = ["conflict1"]
        self.mock_conflict_cls.return_value.resolve_conflicts.return_value = {"resolved_count": 1}
        
        sources = [{"entities": [{"id": "1", "name": "A"}], "relationships": []}]
        graph = builder.build(sources)
        
        # Verify conflict detector was called
        self.mock_conflict_cls.return_value.detect_conflicts.assert_called_once()
        self.mock_conflict_cls.return_value.resolve_conflicts.assert_called_once()

    def test_build_single_source(self):
        builder = GraphBuilder(merge_entities=False, resolve_conflicts=False)
        source = {
            "entities": [{"id": "1", "name": "A"}],
            "relationships": [{"source_id": "1", "target_id": "1", "type": "self"}],
        }
        graph = builder.build_single_source(source)
        self.assertEqual(len(graph["entities"]), 1)
        self.assertEqual(len(graph["relationships"]), 1)

    def test_build_with_explicit_relationships_argument(self):
        builder = GraphBuilder(merge_entities=False, resolve_conflicts=False)

        entities = [
            {"id": "1", "name": "A"},
            {"id": "2", "name": "B"},
        ]
        relationships = [
            {"source_id": "1", "target_id": "2", "type": "rel"},
        ]

        graph = builder.build(entities, relationships=relationships)

        self.assertEqual(len(graph["entities"]), 2)
        self.assertEqual(len(graph["relationships"]), 1)
        rel = graph["relationships"][0]
        self.assertEqual(rel.get("source"), "1")
        self.assertEqual(rel.get("target"), "2")

    def test_build_warns_when_all_relationships_dropped(self):
        builder = GraphBuilder(merge_entities=False, resolve_conflicts=False)
        source = {
            "entities": [],
            "relationships": [{"foo": "x"}, {"bar": "y"}],
        }

        with patch.object(builder.logger, "warning") as mock_warning:
            graph = builder.build(source)

        self.assertEqual(len(graph["relationships"]), 0)
        mock_warning.assert_called()
        args, _ = mock_warning.call_args
        self.assertIn("All relationships were dropped", args[0])

class TestGraphAnalyzer(unittest.TestCase):
    def setUp(self):
        self.mock_tracker_patcher = patch("semantica.kg.graph_analyzer.get_progress_tracker")
        self.mock_get_tracker = self.mock_tracker_patcher.start()
        self.mock_get_tracker.return_value = MagicMock()

        self.mock_centrality_patcher = patch("semantica.kg.graph_analyzer.CentralityCalculator")
        self.mock_centrality_cls = self.mock_centrality_patcher.start()
        self.mock_centrality = self.mock_centrality_cls.return_value

        self.mock_community_patcher = patch("semantica.kg.graph_analyzer.CommunityDetector")
        self.mock_community_cls = self.mock_community_patcher.start()
        self.mock_community = self.mock_community_cls.return_value

        self.mock_connectivity_patcher = patch("semantica.kg.graph_analyzer.ConnectivityAnalyzer")
        self.mock_connectivity_cls = self.mock_connectivity_patcher.start()
        self.mock_connectivity = self.mock_connectivity_cls.return_value

    def tearDown(self):
        self.mock_tracker_patcher.stop()
        self.mock_centrality_patcher.stop()
        self.mock_community_patcher.stop()
        self.mock_connectivity_patcher.stop()

    def test_initialization(self):
        """Test analyzer initialization"""
        analyzer = GraphAnalyzer()
        self.mock_centrality_cls.assert_called_once()
        self.mock_community_cls.assert_called_once()
        self.mock_connectivity_cls.assert_called_once()

    def test_analyze_graph(self):
        """Test comprehensive analysis"""
        analyzer = GraphAnalyzer()
        graph = {"entities": [], "relationships": []}
        
        # Setup mock returns
        self.mock_centrality.calculate_all_centrality.return_value = {"degree": {}}
        self.mock_community.detect_communities.return_value = []
        self.mock_connectivity.analyze_connectivity.return_value = {"components": 1}
        
        # We need to mock compute_metrics if it's called
        # Based on code read, it is called.
        # But compute_metrics is a method of GraphAnalyzer, we can mock it on the instance
        # OR we can let it run if it doesn't have complex dependencies.
        # The code for compute_metrics wasn't fully read, let's assume it might fail if dependencies are missing.
        # Let's mock it for now to isolate delegation logic.
        
        with patch.object(analyzer, 'compute_metrics') as mock_metrics:
            mock_metrics.return_value = {"nodes": 0}
            
            results = analyzer.analyze_graph(graph)
            
            self.assertIn("centrality", results)
            self.assertIn("communities", results)
            self.assertIn("connectivity", results)
            self.assertIn("metrics", results)
            
            self.mock_centrality.calculate_all_centrality.assert_called_once()
            self.mock_community.detect_communities.assert_called_once()
            self.mock_connectivity.analyze_connectivity.assert_called_once()
            mock_metrics.assert_called_once()

class TestTemporalGraphQuery(unittest.TestCase):
    def setUp(self):
        self.mock_tracker_patcher = patch("semantica.utils.progress_tracker.get_progress_tracker")
        self.mock_get_tracker = self.mock_tracker_patcher.start()
        self.mock_get_tracker.return_value = MagicMock()
        
        # Patch TemporalPatternDetector if needed, or let it run since it's simple
        # It's better to let it run to test integration within the module if it has no external deps
        
        from semantica.kg.temporal_query import TemporalGraphQuery
        self.query_engine = TemporalGraphQuery()

    def tearDown(self):
        self.mock_tracker_patcher.stop()

    def test_query_at_time(self):
        """Test querying graph at specific time"""
        graph = {
            "entities": [{"id": "1"}, {"id": "2"}],
            "relationships": [
                {
                    "source": "1", "target": "2", "type": "rel1",
                    "valid_from": "2023-01-01", "valid_until": "2023-12-31"
                },
                {
                    "source": "2", "target": "1", "type": "rel2",
                    "valid_from": "2024-01-01", "valid_until": "2024-12-31"
                }
            ]
        }
        
        # Query in 2023
        result_2023 = self.query_engine.query_at_time(graph, "", "2023-06-01")
        self.assertEqual(len(result_2023["relationships"]), 1)
        self.assertEqual(result_2023["relationships"][0]["type"], "rel1")
        
        # Query in 2024
        result_2024 = self.query_engine.query_at_time(graph, "", "2024-06-01")
        self.assertEqual(len(result_2024["relationships"]), 1)
        self.assertEqual(result_2024["relationships"][0]["type"], "rel2")
        
        # Query in 2025 (no matches)
        result_2025 = self.query_engine.query_at_time(graph, "", "2025-06-01")
        self.assertEqual(len(result_2025["relationships"]), 0)

    def test_query_time_range(self):
        """Test querying graph within time range"""
        graph = {
            "relationships": [
                {
                    "source": "1", "target": "2",
                    "valid_from": "2023-01-01", "valid_until": "2023-06-30"
                }
            ]
        }
        
        # Range overlaps
        result = self.query_engine.query_time_range(graph, "", "2023-02-01", "2023-08-01")
        self.assertEqual(len(result["relationships"]), 1)
        
        # Range does not overlap (after)
        result = self.query_engine.query_time_range(graph, "", "2023-07-01", "2023-08-01")
        self.assertEqual(len(result["relationships"]), 0)

    def test_find_temporal_paths(self):
        """Test finding paths with temporal constraints"""
        graph = {
            "relationships": [
                {"source": "A", "target": "B", "valid_from": "2023-01-01"},
                {"source": "B", "target": "C", "valid_from": "2023-01-01"}
            ]
        }
        
        # Find path A -> C valid in 2023
        result = self.query_engine.find_temporal_paths(
            graph, "A", "C", start_time="2023-02-01", end_time="2023-12-31"
        )
        self.assertEqual(result["num_paths"], 1)
        self.assertEqual(len(result["paths"][0]["path"]), 3) # A, B, C

    def test_parse_time_normalizes_equivalent_dates_and_utc(self):
        parsed_a = self.query_engine._parse_time("2024-1-1")
        parsed_b = self.query_engine._parse_time("2024-01-01")
        parsed_c = self.query_engine._parse_time("2024-06-15T10:00:00+05:30")

        self.assertEqual(parsed_a, parsed_b)
        self.assertEqual(parsed_c, datetime(2024, 6, 15, 4, 30, tzinfo=timezone.utc))

    def test_parse_time_invalid_raises_temporal_validation_error(self):
        with self.assertRaises(TemporalValidationError):
            self.query_engine._parse_time("not-a-date")

    def test_query_at_time_supports_open_bound_and_canonical_none_without_warning(self):
        graph = {
            "relationships": [
                {
                    "source": "1",
                    "target": "2",
                    "type": "current",
                    "valid_from": "2024-01-01",
                    "valid_until": TemporalBound.OPEN,
                },
                {
                    "source": "2",
                    "target": "3",
                    "type": "deprecated-none",
                    "valid_from": "2024-01-01",
                    "valid_until": None,
                },
            ]
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self.query_engine.query_at_time(graph, "", "2026-01-01")

        self.assertEqual(len(result["relationships"]), 2)
        self.assertFalse(any(item.category is DeprecationWarning for item in caught))

    def test_query_at_time_supports_transaction_axis(self):
        graph = {
            "relationships": [
                {
                    "source": "1",
                    "target": "2",
                    "type": "known-later",
                    "valid_from": "2019-01-01",
                    "valid_until": TemporalBound.OPEN,
                    "recorded_at": "2021-01-01",
                    "superseded_at": TemporalBound.OPEN,
                }
            ]
        }

        result_2020 = self.query_engine.query_at_time(
            graph, "", "2020-06-01", time_axis="transaction"
        )
        result_2021 = self.query_engine.query_at_time(
            graph, "", "2021-06-01", time_axis="transaction"
        )

        self.assertEqual(len(result_2020["relationships"]), 0)
        self.assertEqual(len(result_2021["relationships"]), 1)

    def test_analyze_evolution_accepts_open_valid_until(self):
        graph = {
            "relationships": [
                {
                    "source": "1",
                    "target": "2",
                    "type": "current",
                    "valid_from": "2024-01-01",
                    "valid_until": TemporalBound.OPEN,
                }
            ]
        }

        result = self.query_engine.analyze_evolution(
            graph,
            start_time="2025-01-01",
            end_time="2025-12-31",
        )

        self.assertEqual(result["num_relationships"], 1)

    def test_query_at_time_legacy_transaction_axis_uses_valid_from_when_recorded_missing(self):
        graph = {
            "relationships": [
                {
                    "source": "1",
                    "target": "2",
                    "type": "legacy",
                    "valid_from": "2024-01-01",
                    "valid_until": TemporalBound.OPEN,
                }
            ]
        }

        result = self.query_engine.query_at_time(
            graph, "", "2025-06-01", time_axis="transaction"
        )

        self.assertEqual(len(result["relationships"]), 1)

    def test_null_valid_until_deserializes_to_open(self):
        relationship = deserialize_relationship_temporal_fields(
            {"source": "1", "target": "2", "type": "rel", "valid_from": "2024-01-01", "valid_until": None}
        )
        self.assertIs(relationship["valid_until"], TemporalBound.OPEN)

if __name__ == "__main__":
    unittest.main()
