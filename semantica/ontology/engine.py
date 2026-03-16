from typing import Any, Dict, List, Optional

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker
from .ontology_generator import OntologyGenerator
from .class_inferrer import ClassInferrer
from .property_generator import PropertyGenerator
from .owl_generator import OWLGenerator
from .ontology_evaluator import OntologyEvaluator
from .ontology_validator import OntologyValidator
from .llm_generator import LLMOntologyGenerator
from ..semantic_extract.triplet_extractor import Triplet


class OntologyEngine:
    def __init__(self, **config):
        self.logger = get_logger("ontology_engine")
        self.progress = get_progress_tracker()
        self.config = config

        self.generator = OntologyGenerator(**config)
        self.inferrer = ClassInferrer(**config)
        self.propgen = PropertyGenerator(**config)
        self.owl = OWLGenerator(**config)
        self.evaluator = OntologyEvaluator(**config)
        self.validator = OntologyValidator(**config)
        self.llm = LLMOntologyGenerator(**config)
        self.store = config.get("store")

    def from_data(self, data: Dict[str, Any], **options) -> Dict[str, Any]:
        tracking_id = self.progress.start_tracking(
            module="ontology",
            submodule="OntologyEngine",
            message="Generating ontology from data",
        )
        try:
            ontology = self.generator.generate_ontology(data, **options)
            self.progress.update_tracking(tracking_id, message="Ontology generated")
            return ontology
        except Exception as e:
            self.progress.update_tracking(tracking_id, message="Generation failed")
            raise

    def from_text(self, text: str, provider: Optional[str] = None, model: Optional[str] = None, **options) -> Dict[str, Any]:
        if provider:
            self.llm.set_provider(provider, model=model)
        return self.llm.generate_ontology_from_text(text, **options)

    def infer_classes(self, entities: List[Dict[str, Any]], **options) -> List[Dict[str, Any]]:
        return self.inferrer.infer_classes(entities, **options)

    def infer_properties(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        classes: List[Dict[str, Any]],
        **options,
    ) -> List[Dict[str, Any]]:
        return self.propgen.infer_properties(entities, relationships, classes, **options)
    
    def _sanitize_uri(self, uri: str) -> str:
        """Prevent SPARQL injection by percent-encoding dangerous characters."""
        if not isinstance(uri, str):
            return ""
        return uri.replace("<", "%3C").replace(">", "%3E")
    
    def create_alignment(self, source_uri: str, target_uri: str, predicate: str, **options) -> None:
        """
        Creates an alignment between two ontology entities and stores it.
        """
        if not self.store:
            raise ProcessingError("TripletStore instance not configured in OntologyEngine.")

        if not predicate.startswith(("http://", "https://")):
            raise ProcessingError(
                f"predicate must be a full URI (e.g. 'http://www.w3.org/2002/07/owl#equivalentClass'), "
                f"not a CURIE: '{predicate}'"
            )

        tracking_id = self.progress.start_tracking(
            module="ontology",
            submodule="OntologyEngine",
            message=f"Creating alignment: {source_uri} -> {target_uri}"
        )
        try:
            triplet = Triplet(subject=source_uri, predicate=predicate, object=target_uri)
            self.store.add_triplet(triplet, **options)
            
            self.progress.stop_tracking(tracking_id, status="completed", message="Alignment created")
        except Exception as e:
            self.progress.stop_tracking(tracking_id, status="failed", message=str(e))
            self.logger.error(f"Failed to create alignment: {e}")
            raise ProcessingError(f"Alignment creation failed: {e}")

    def get_alignments(self, entity_uri: str, **options) -> List[Dict[str, Any]]:
        """
        Retrieves all alignments for a specific entity URI (bidirectional).
        """
        if not self.store:
            raise ProcessingError("TripletStore instance not configured in OntologyEngine.")
        
        safe_uri = self._sanitize_uri(entity_uri)

        query = f"""
        SELECT ?s ?p ?o WHERE {{
            {{ <{safe_uri}> ?p ?o . BIND(<{safe_uri}> AS ?s) }}
            UNION
            {{ ?s ?p <{safe_uri}> . BIND(<{safe_uri}> AS ?o) }}
            
            FILTER (?p IN (
                <http://www.w3.org/2002/07/owl#equivalentClass>,
                <http://www.w3.org/2002/07/owl#equivalentProperty>,
                <http://www.w3.org/2002/07/owl#sameAs>,
                <http://www.w3.org/2004/02/skos/core#exactMatch>,
                <http://www.w3.org/2004/02/skos/core#closeMatch>,
                <http://www.w3.org/2004/02/skos/core#broadMatch>,
                <http://www.w3.org/2004/02/skos/core#narrowMatch>,
                <http://www.w3.org/2004/02/skos/core#relatedMatch>
            ))
        }}
        """
        try:
            results = self.store.execute_query(query, **options)

            alignments = []
            if hasattr(results, 'bindings'):
                for b in results.bindings:
                    alignments.append({
                        "source": b.get("s", {}).get("value") if isinstance(b.get("s"), dict) else b.get("s"),
                        "predicate": b.get("p", {}).get("value") if isinstance(b.get("p"), dict) else b.get("p"),
                        "target": b.get("o", {}).get("value") if isinstance(b.get("o"), dict) else b.get("o")
                    })
            return alignments
        except Exception as e:
            self.logger.error(f"Failed to get alignments for {entity_uri}: {e}")
            raise ProcessingError(f"Failed to get alignments: {e}")

    def list_alignments(self, ontology_uri: Optional[str] = None, **options) -> List[Dict[str, Any]]:
        """
        Lists all alignments, optionally filtered by an ontology URI.
        """
        if not self.store:
            raise ProcessingError("TripletStore instance not configured in OntologyEngine.")

        filter_clause = ""
        if ontology_uri:
            # Sanitize characters that could break out of the SPARQL string literal or WHERE block
            safe_ontology_uri = (
                ontology_uri
                .replace("\\", "%5C")
                .replace('"', '%22')
                .replace("{", "%7B")
                .replace("}", "%7D")
            )
            filter_clause = f'FILTER(STRSTARTS(STR(?s), "{safe_ontology_uri}") || STRSTARTS(STR(?o), "{safe_ontology_uri}"))'

        query = f"""
        SELECT ?s ?p ?o WHERE {{
            ?s ?p ?o .
            FILTER (?p IN (
                <http://www.w3.org/2002/07/owl#equivalentClass>,
                <http://www.w3.org/2002/07/owl#equivalentProperty>,
                <http://www.w3.org/2002/07/owl#sameAs>,
                <http://www.w3.org/2004/02/skos/core#exactMatch>,
                <http://www.w3.org/2004/02/skos/core#closeMatch>,
                <http://www.w3.org/2004/02/skos/core#broadMatch>,
                <http://www.w3.org/2004/02/skos/core#narrowMatch>,
                <http://www.w3.org/2004/02/skos/core#relatedMatch>
            ))
            {filter_clause}
        }}
        """
        try:
            results = self.store.execute_query(query, **options)
            
            alignments = []
            if hasattr(results, 'bindings'):
                for b in results.bindings:
                    alignments.append({
                        "source": b.get("s", {}).get("value") if isinstance(b.get("s"), dict) else b.get("s"),
                        "predicate": b.get("p", {}).get("value") if isinstance(b.get("p"), dict) else b.get("p"),
                        "target": b.get("o", {}).get("value") if isinstance(b.get("o"), dict) else b.get("o")
                    })
            return alignments
        except Exception as e:
            self.logger.error(f"Failed to list alignments: {e}")
            raise ProcessingError(f"Failed to list alignments: {e}")
            
    def evaluate(self, ontology: Dict[str, Any], **options):
        return self.evaluator.evaluate_ontology(ontology, **options)

    def validate(self, ontology: Dict[str, Any], **options):
        return self.validator.validate(ontology, **options)

    def to_owl(self, ontology: Dict[str, Any], format: str = "turtle", **options):
        return self.owl.generate_owl(ontology, format=format, **options)

    def export_owl(self, ontology: Dict[str, Any], path: str, format: str = "turtle"):
        return self.owl.export_owl(ontology, path, format=format)