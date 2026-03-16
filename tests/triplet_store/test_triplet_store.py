import unittest
from unittest.mock import MagicMock, patch
from semantica.triplet_store.triplet_store import TripletStore
from semantica.triplet_store.query_engine import QueryEngine
from semantica.semantic_extract.triplet_extractor import Triplet

class TestTripletStore(unittest.TestCase):

    def setUp(self):
        self.mock_logger = MagicMock()
        self.mock_tracker = MagicMock()
        
        self.logger_patcher = patch('semantica.triplet_store.triplet_store.get_logger', return_value=self.mock_logger)
        self.tracker_patcher = patch('semantica.triplet_store.triplet_store.get_progress_tracker', return_value=self.mock_tracker)
        
        self.logger_patcher.start()
        self.tracker_patcher.start()

    def tearDown(self):
        self.logger_patcher.stop()
        self.tracker_patcher.stop()

    @patch('semantica.triplet_store.blazegraph_store.BlazegraphStore')
    def test_triplet_store_init(self, mock_blazegraph_store):
        store = TripletStore(backend="blazegraph", endpoint="http://localhost:9999")
        self.assertEqual(store.backend_type, "blazegraph")
        self.assertEqual(store.endpoint, "http://localhost:9999")
        mock_blazegraph_store.assert_called_once()

    @patch('semantica.triplet_store.blazegraph_store.BlazegraphStore')
    def test_add_triplet(self, mock_blazegraph_store):
        # Setup mock backend
        mock_backend_instance = MagicMock()
        mock_blazegraph_store.return_value = mock_backend_instance
        mock_backend_instance.add_triplet.return_value = {"status": "success"}
        
        store = TripletStore(backend="blazegraph")
        triplet = Triplet(subject="s", predicate="p", object="o")
        
        result = store.add_triplet(triplet)
        
        self.assertEqual(result, {"status": "success"})
        mock_backend_instance.add_triplet.assert_called_once_with(triplet)

    @patch('semantica.triplet_store.blazegraph_store.BlazegraphStore')
    def test_add_triplets(self, mock_blazegraph_store):
        # Setup mock backend and bulk loader
        mock_backend_instance = MagicMock()
        mock_blazegraph_store.return_value = mock_backend_instance
        
        store = TripletStore(backend="blazegraph")
        
        # Mock bulk loader
        mock_loader = MagicMock()
        store.bulk_loader = mock_loader
        mock_progress = MagicMock()
        mock_progress.metadata = {"success": True}
        mock_progress.total_triplets = 2
        mock_progress.loaded_triplets = 2
        mock_progress.failed_triplets = 0
        mock_progress.total_batches = 1
        mock_loader.load_triplets.return_value = mock_progress
        
        triplets = [
            Triplet(subject="s1", predicate="p1", object="o1"),
            Triplet(subject="s2", predicate="p2", object="o2")
        ]
        
        result = store.add_triplets(triplets, batch_size=2)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["total"], 2)
        mock_loader.load_triplets.assert_called_once()

    @patch('semantica.triplet_store.blazegraph_store.BlazegraphStore')
    def test_get_triplets(self, mock_blazegraph_store):
        mock_backend_instance = MagicMock()
        mock_blazegraph_store.return_value = mock_backend_instance
        expected_triplets = [Triplet(subject="s", predicate="p", object="o")]
        mock_backend_instance.get_triplets.return_value = expected_triplets
        
        store = TripletStore(backend="blazegraph")
        result = store.get_triplets(subject="s")
        
        self.assertEqual(result, expected_triplets)
        mock_backend_instance.get_triplets.assert_called_once_with(subject="s", predicate=None, object=None)

    @patch('semantica.triplet_store.blazegraph_store.BlazegraphStore')
    def test_delete_triplet(self, mock_blazegraph_store):
        mock_backend_instance = MagicMock()
        mock_blazegraph_store.return_value = mock_backend_instance
        mock_backend_instance.delete_triplet.return_value = {"success": True}
        
        store = TripletStore(backend="blazegraph")
        triplet = Triplet(subject="s", predicate="p", object="o")
        
        result = store.delete_triplet(triplet)
        
        self.assertTrue(result["success"])
        mock_backend_instance.delete_triplet.assert_called_once_with(triplet)
    
    def test_query_engine_build_values_clause(self):
        engine = QueryEngine()
        uris = ["http://ex.org/1", "http://ex.org/2"]
        
        clause = engine.build_values_clause("subject", uris)
        self.assertEqual(clause, "VALUES ?subject { <http://ex.org/1> <http://ex.org/2> }")
        
        empty_clause = engine.build_values_clause("subject", [])
        self.assertEqual(empty_clause, "")

    def test_query_engine_expand_entity_uri_disabled(self):
        engine = QueryEngine()
        mock_backend = MagicMock()
        
        result = engine.expand_entity_uri("http://ex.org/1", mock_backend, use_alignments=False)
        
        # Should return only the original URI and NOT query the store
        self.assertEqual(result, ["http://ex.org/1"])
        mock_backend.execute_sparql.assert_not_called()

    def test_query_engine_expand_entity_uri_enabled(self):
        engine = QueryEngine()
        mock_backend = MagicMock()
        
        # Mock the backend returning an aligned URI
        mock_backend.execute_sparql.return_value = {
            "bindings": [{"aligned": {"value": "http://ex.org/aligned_entity"}}]
        }
        
        result = engine.expand_entity_uri("http://ex.org/original", mock_backend, use_alignments=True)
        
        self.assertIn("http://ex.org/original", result)
        self.assertIn("http://ex.org/aligned_entity", result)
        self.assertEqual(len(result), 2)
        mock_backend.execute_sparql.assert_called_once()
        
    def test_end_to_end_cross_ontology_uri_flow(self):
        """
        Full end-to-end: real expand_entity_uri queries a mock backend,
        then build_values_clause injects the results into a SPARQL template.
        """
        engine = QueryEngine()
        mock_backend = MagicMock()
        mock_backend.execute_sparql.return_value = {
            "bindings": [{"aligned": {"value": "http://aligned.org/2"}}]
        }

        original_uri = "http://ex.org/1"
        expanded = engine.expand_entity_uri(original_uri, store_backend=mock_backend, use_alignments=True)
        values_clause = engine.build_values_clause("subject", expanded)

        sparql_query = f"""
            SELECT ?instance ?name WHERE {{
                {values_clause}
                ?instance a ?subject .
                ?instance <http://schema.org/name> ?name .
            }}
        """

        self.assertIn("http://ex.org/1", sparql_query)
        self.assertIn("http://aligned.org/2", sparql_query)
        self.assertIn("VALUES ?subject", sparql_query)
        mock_backend.execute_sparql.assert_called_once()
