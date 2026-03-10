# Semantica v0.3.0 — Release Notes

**Released:** 2026-03-10
**PyPI:** `pip install semantica`
**Tag:** [v0.3.0](https://github.com/Hawksight-AI/semantica/releases/tag/v0.3.0)
**Classification:** Production/Stable

> First stable, full public release of Semantica. Covers everything shipped across three release stages: 0.3.0-alpha (2026-02-19), 0.3.0-beta (2026-03-07), and 0.3.0 stable (2026-03-10).

---

## Contributors

| Contributor | Role |
|------------|------|
| [@KaifAhmad1](https://github.com/KaifAhmad1) | Lead maintainer — context graph, decision intelligence, KG algorithms, semantic extraction, pipeline, provenance, bug fixes, release management |
| [@ZohaibHassan16](https://github.com/ZohaibHassan16) | Deduplication v2 suite (candidate generation, two-stage scoring, semantic dedup), incremental/delta processing, benchmark suite |
| [@Sameer6305](https://github.com/Sameer6305) | Apache AGE backend, PgVector store, Snowflake connector, Apache Arrow export |
| [@tibisabau](https://github.com/tibisabau) | ArangoDB AQL export, Apache Parquet export |
| [@d4ndr4d3](https://github.com/d4ndr4d3) | ResourceScheduler deadlock fix |

---

## v0.3.0 — Stable (2026-03-10)

### Context Graph Feature Completeness

**Temporal Validity Windows** (by @KaifAhmad1)

Nodes and edges now carry first-class `valid_from` / `valid_until` ISO datetime fields. These are stored directly on `ContextNode` and `ContextEdge` dataclasses — not in metadata — and survive full serialisation round-trips through `save_to_file()` / `load_from_file()` and `to_dict()` / `from_dict()`.

- `ContextNode.is_active(at_time=None)` and `ContextEdge.is_active(at_time=None)` — returns `True` if the node/edge is live at the given time (defaults to now). Handles both tz-aware and tz-naive datetime inputs correctly.
- `ContextGraph.find_active_nodes(node_type=None, at_time=None)` — filters the entire graph and returns only nodes within their validity window.
- `add_node(valid_from=..., valid_until=...)` and `add_edge(valid_from=..., valid_until=...)` — pass validity fields directly in the call signature.
- Bug fix: `is_active()` previously crashed with `TypeError` when passed a tz-aware `datetime` (e.g. `datetime.now(timezone.utc)`). Fixed by normalising all inputs to tz-naive UTC via a new `_parse_iso_dt()` helper.
- Bug fix: validity fields were silently lost in `add_nodes()`, `add_edges()`, `to_dict()`, and `from_dict()`. All four paths now correctly preserve and restore them.

**Weighted Multi-Hop BFS** (by @KaifAhmad1)

`ContextGraph.get_neighbors(node_id, hops=1, relationship_types=None, min_weight=0.0)` now accepts a `min_weight` threshold. Any edge with weight below the threshold is skipped during BFS traversal, allowing callers to confine multi-hop queries to high-confidence causal links. Default `0.0` is fully backward-compatible.

**Cross-Graph Navigation** (by @KaifAhmad1)

Separate `ContextGraph` instances can now be linked and navigated between — hierarchically, like separate knowledge domains that reference each other.

- `link_graph(other_graph, source_node_id, target_node_id, link_type="CROSS_GRAPH") -> str` — creates a navigable bridge and returns a `link_id`. Records a dedicated `"cross_graph_link"` typed marker node internally (not a phantom `"entity"`) and a marker edge.
- `navigate_to(link_id) -> (other_graph, target_node_id)` — jumps to the target graph and entry node for a given link.
- `graph_id` field — each `ContextGraph` now carries a stable UUID so instances can identify each other across save/load.
- `save_to_file()` — now writes a `links` section alongside nodes and edges, containing `link_id`, `source_node_id`, `target_node_id`, and `other_graph_id` for every cross-graph link.
- `load_from_file()` — restores `graph_id` and populates `_unresolved_links` from the `links` section.
- `resolve_links(registry: Dict[str, ContextGraph]) -> int` — reconnects unresolved links post-load. Pass `{graph_id: graph_instance}` for each linked graph; returns the count of successfully resolved links. `navigate_to()` raises a clear `KeyError` with a `resolve_links()` hint if called before resolution.
- Bug fix: the previous implementation auto-created the synthetic marker target as an `"entity"` node (phantom pollution). Fixed by explicitly pre-creating a `"cross_graph_link"` typed `ContextNode` before the marker edge.
- 14 new tests in `tests/context/test_cross_graph_navigation.py` covering all scenarios including full save/load round-trips with partial registry resolution.

**Other Fixes** (by @KaifAhmad1)

- `PipelineBuilder.add_step()` return type annotation corrected from `"PipelineBuilder"` to `"PipelineStep"` — the implementation was already correct; only the annotation and docstring were stale.
- `test_hybrid_search_performance` timing computation fixed — now accumulates a true `search_times` list instead of reusing the last loop iteration's `start_time`; threshold relaxed to `< 5.0s` for real `sentence-transformers` (384-dim) latency on development machines.

**Test Coverage Added**

- 14 cross-graph navigation tests (`tests/context/test_cross_graph_navigation.py`)
- **Total: 335 context tests, 886+ tests across all modules — 0 failures**

---

## v0.3.0-beta — Beta (2026-03-07)

### Semantic Extraction Fixes

**Multi-Founder LLM Extraction & Reasoner Inference Fix** (PR #354, by @KaifAhmad1)

- `_parse_relation_result` in `methods.py` — unmatched subjects/objects now produce a synthetic `UNKNOWN` entity instead of silently dropping the relation. All co-founders returned by the LLM are preserved in the output.
- Duplicate relation fix — an orphaned legacy block that appended every relation twice has been removed.
- `extraction_method` parameter added — typed extraction paths now correctly record `"llm_typed"` in relation metadata instead of `"llm"`.
- `_match_pattern` in `reasoner.py` rewritten — splits patterns on `?var` placeholders first, then escapes only literal segments. Pre-bound variables resolve to exact literals, repeated variables use backreferences, non-greedy `.+?` prevents over-consumption of separators.
- Added `tests/reasoning/test_reasoner.py` (4 tests) and `tests/semantic_extract/test_relation_extractor.py` (6 tests).

**TTL Export Alias Fix** (PR #355, by @KaifAhmad1)

- `RDFExporter` now accepts `"ttl"`, `"nt"`, `"xml"`, `"rdf"`, and `"json-ld"` as format aliases in `export_to_rdf()`. Aliases resolve before format validation — zero public API changes.
- Added `tests/export/test_rdf_exporter.py` (8 tests).

### Incremental / Delta Processing

**Native Delta Computation** (PR #349, by @ZohaibHassan16, reviewed and fixed by @KaifAhmad1)

- Native SPARQL-based diff between graph snapshots — only changed triples flow through the pipeline.
- `delta_mode` configuration in `PipelineBuilder` for near-real-time workloads.
- Version snapshot management with graph URI tracking and metadata storage.
- `prune_versions()` for automatic snapshot retention cleanup.
- Bug fixes: corrected SPARQL variable order, fixed class references, resolved duplicate dictionary keys.

### Deduplication v2

**Candidate Generation v2** (PR #338, by @ZohaibHassan16)

- New opt-in strategies: `blocking_v2` and `hybrid_v2`, replacing O(N²) pair enumeration.
- Multi-key blocking with normalised token prefixes, type-aware keys, and optional phonetic (Soundex) blocking.
- Deterministic `max_candidates_per_entity` budgeting with stable sorting.
- **63.6% faster** in worst-case scenarios (0.259s → 0.094s for 100 entities).

**Two-Stage Scoring Prefilter** (PR #339, by @ZohaibHassan16)

- Fast gates for type mismatch, name-length ratio, and token overlap eliminate expensive semantic scoring for obvious non-matches.
- Configurable thresholds: `min_length_ratio`, `min_token_overlap_ratio`, `required_shared_token`.
- **18–25% faster** batch processing with prefilter enabled (`prefilter_enabled=False` by default).

**Semantic Relationship Deduplication v2** (PR #340, by @ZohaibHassan16, fixes by @KaifAhmad1)

- Canonicalisation engine with predicate synonym mapping (`works_for` → `employed_by`).
- O(1) hash matching for exact canonical signatures.
- Weighted scoring: 60% predicate + 40% object composition with explainable `semantic_match_score`.
- **6.98x faster** than legacy mode (83ms vs 579ms).
- `dedup_triplets()` infinite recursion bug fixed; function is now a first-class API in `methods.py`.

**Deduplication v2 Migration Guide** (PR #344, by @ZohaibHassan16, fixes by @KaifAhmad1)

- Comprehensive `MIGRATION_V2.md` documenting all v2 strategies with code examples.
- Full backward compatibility maintained — legacy mode remains the default.

### Export Formats

**ArangoDB AQL Export** (PR #342, by @tibisabau)

- Full AQL INSERT statement generation for vertices and edges.
- Configurable collection names with validation and sanitisation; batch processing (default: 1000).
- `export_arango()` convenience function; `.aql` auto-detection in the unified exporter.
- 17 tests, 100% pass rate.

**Apache Parquet Export** (PR #343, by @tibisabau)

- Columnar storage format with configurable compression: snappy, gzip, brotli, zstd, lz4, none.
- Explicit Apache Arrow schemas with type safety; field normalisation for varied naming conventions.
- `export_parquet()` convenience function; `.parquet` auto-detection.
- Analytics-ready for pandas, Spark, Snowflake, BigQuery, Databricks.
- 25 tests, 100% pass rate.

### Bug Fixes & Test Suite Stabilisation

**Test Suite Fixes** (by @KaifAhmad1)

Context module:
- `retrieve_decision_precedents` — gated entity extraction on `use_hybrid_search=True` correctly.
- `_extract_entities_from_query` — now uses `word[0].isupper()` to capture camelCase identifiers like `CreditCard`.
- Added missing `expand_context` (BFS traversal) and `_get_decision_query` methods.
- Fixed `hybrid_retrieval`, `dynamic_context_traversal`, and `multi_hop_context_assembly` for correct single-pass BFS.
- Fixed `_retrieve_from_vector` fallback to `result["metadata"]["content"]` to prevent empty content and negative re-ranking scores.

KG module:
- `calculate_pagerank` — added `alpha`/`max_iter` aliases; return format changed to `{"centrality": scores, "rankings": sorted_list}`.
- `community_detector._to_networkx` — now returns a NetworkX graph directly when one is passed (previously lost all edges).
- Added 9 domain-specific tracking methods to `AlgorithmTrackerWithProvenance`.
- Created `provenance_tracker.py` with `ProvenanceTracker` (`track_entity`, `get_all_sources`, `clear`).

Pipeline module:
- Retry loop fixed — now correctly iterates to `max_retries`.
- `RecoveryAction` dataclass and `handle_failure(error, policy, retry_count)` added with LINEAR, EXPONENTIAL, and FIXED strategies.
- `add_step()` fixed to return the created `PipelineStep`.
- `validate` added as alias for `validate_pipeline` in `PipelineValidator`.

Other:
- Fixed `NameError` for missing `Type` import in `utils/helpers.py`.
- Vector store performance threshold relaxed (100ms → 500ms per decision for development machines).
- Windows cp1252 encoding fix in test files (emoji → ASCII).
- `ProvenanceTracker` added to `semantica/kg/__init__.py` exports.

**Results: ~840 tests passing, 36 skipped (external services), 0 failed**

---

## v0.3.0-alpha — Alpha (2026-02-19)

### Context & Decision Intelligence

**Context Engineering Enhancement** (PR #307, by @KaifAhmad1)

The foundational 0.3.0 feature — complete overhaul of the context module for production-grade decision intelligence:

- Full decision lifecycle: `record_decision()` → `add_decision()` → `add_causal_relationship()` → `trace_decision_chain()` → `analyze_decision_impact()` → `analyze_decision_influence()` → `find_similar_decisions()`
- `AgentContext` unified wrapper with granular feature flags: `decision_tracking`, `kg_algorithms`, `graph_expansion`; methods: `store()`, `retrieve()`, `get_conversation_history()`, `get_statistics()`, `capture_cross_system_inputs()`
- `AgentMemory` with working, conversation, and long-term memory tiers
- `PolicyEngine` with versioned policy nodes, compliance checking (`check_decision_rules()`), and `PolicyException` model
- Hybrid precedent search combining vector, structural, and category similarity with configurable weights
- Decision influence analysis via centrality measures and causal chain tracking
- GraphStore validation preventing runtime failures; secure logging
- 9 critical bug fixes across logging, security, audit trails, API compatibility, Cypher queries, centrality access, validation

**Context Decision Tracking Fixes** (PR #315, by @KaifAhmad1)

- Fixed empty/None decision ID handling in `add_decision()`
- Fixed None metadata handling preventing `TypeError`
- Fixed causal chain depth logic and node exclusion
- Fixed nonexistent node handling in `add_causal_relationship()`
- Fixed precedent search direction in `find_precedents()`
- Added missing `properties` field in `to_dict()`; added `from_dict()` method
- Fixed UUID generation across all decision models
- All 71 context tests passing

### Knowledge Graph Algorithms

**Improved Graph Algorithms** (PR #292, by @KaifAhmad1)

- 30+ graph algorithms across 7 categories
- Node embeddings: Node2Vec, DeepWalk, Word2Vec via `NodeEmbedder`
- Similarity: cosine, Euclidean, Manhattan, Correlation via `SimilarityCalculator`
- Path finding: Dijkstra, A*, BFS, K-shortest paths via `PathFinder`
- Link prediction: preferential attachment, Jaccard, Adamic-Adar via `LinkPredictor`
- Centrality: degree, betweenness, closeness, PageRank via `CentralityAnalyzer`
- Community detection: Louvain, Leiden, label propagation via `CommunityDetector`
- Connectivity: components, bridges, density via `ConnectivityAnalyzer`
- `GraphBuilderWithProvenance` and `AlgorithmTrackerWithProvenance` with full execution metadata

**Improved Vector Store for Decision Tracking** (PR #293, by @KaifAhmad1)

- `DecisionEmbeddingPipeline` with semantic and structural embeddings
- `HybridSimilarityCalculator` with configurable weights (semantic: 0.7, structural: 0.3)
- `ContextRetriever` with multi-hop reasoning
- Convenience API: `quick_decision()`, `find_precedents()`, `explain()`, `similar_to()`, `batch_decisions()`, `filter_decisions()`
- 34+ tests; performance: 0.028s per decision, 0.031s search, ~0.8KB memory per decision

### Graph Database Backends

**Apache AGE Backend Security Fixes** (PR #311, by @Sameer6305, fixes by @KaifAhmad1)

- `AgeStore` class with `GraphStore` API compatibility (openCypher via SQL on PostgreSQL)
- SQL injection vulnerabilities fixed with input validation
- psycopg2-binary dependency and migration guide added
- Fixed parameter replacement and test mock leakage

**PgVector Store Support** (PR #303, by @Sameer6305, @KaifAhmad1)

- Native PostgreSQL vector storage using the pgvector extension
- Multiple distance metrics: cosine, L2/Euclidean, inner product
- HNSW and IVFFlat indexing for approximate nearest neighbour search
- JSONB metadata storage with flexible filtering; batch operations
- Connection pooling with psycopg3/psycopg2 fallback
- SQL injection protection via `psycopg_sql.SQL()`; idempotent index and table management
- 36+ tests with Docker integration

### Infrastructure

**ResourceScheduler Deadlock Fix** (PR #299, #301, by @d4ndr4d3, @KaifAhmad1)

- Replaced `threading.Lock()` with `threading.RLock()` to fix nested lock acquisition deadlock in `allocate_resources()`
- Added `ValidationError` when no resources can be allocated
- Progress tracking updates moved outside lock scope
- 6 regression tests for deadlock prevention

**Security Configuration** (by @KaifAhmad1)

- Dependabot bi-weekly security updates with manual review
- Automated security scans (Bandit, Safety, Semgrep) on schedule
- Security-critical package grouping; zero auto-merge policy

---

## Summary by the Numbers

| Metric | Value |
|--------|-------|
| Total tests passing | **886+** |
| Test failures | **0** |
| Context tests | 335 |
| KG tests | ~430 |
| Semantic extraction tests | 70 (9 skipped — external LLM APIs) |
| Reasoning tests | 19 |
| Real-world scenario tests | 85 |
| PyPI classifier | Production/Stable |
| Python support | 3.8 – 3.12 |

---

## Upgrade

```bash
pip install --upgrade semantica
```

No breaking changes. All new parameters have safe defaults and all new methods are additive.

See [CHANGELOG.md](CHANGELOG.md) for the full line-by-line diff.
