# Repository Structure & Component Inventory

## VERITAS-MM 2.0: Modular Software Engineering Directory Map

---

```
multimodal-misinfo-submission/
├── app.py                                   # Primary Streamlit multi-tab web application (Production UI)
├── requirements.txt                         # Certified production dependencies
├── README.md                                # Recruiter-friendly, academic project portfolio README
├── .gitignore                               # Comprehensive git protection rules (DB, keys, caches)
├── PUBLIC_RELEASE_CHECKLIST.md              # Pre-publication verification and security checklist
│
├── configs/                                 # Centralized system configurations
│   ├── config.yaml                          # Thresholds (tau=0.28, Delta=0.07), paths, timeouts
│   └── news_sources.json                    # Registry of 9 authorized national and global news feeds
│
├── src/                                     # Production application source code (23 modules)
│   ├── __init__.py                          # Package root
│   ├── pipeline.py                          # Master PipelineService orchestrating Tri-Pillar verification
│   ├── decomposition.py                     # spaCy linguistic dependency parser & proposition extractor
│   ├── retrieval.py                         # Multi-tier evidence retrieval (Google News, Wiki, BM25)
│   ├── verification.py                      # DeBERTa-v3 directional NLI cross-encoder & epistemic adjudicator
│   ├── cross_modal.py                       # OpenCLIP ViT-B/32 unit-hypersphere alignment & selective reject
│   ├── rhetoric_classifier.py               # 1.58M Gated Multimodal Fusion Adapter inference engine
│   ├── database.py                          # SQLite 3 ACID datastore (WAL mode, articles, verifications)
│   ├── news_ingestion.py                    # RSS/Atom newsfeed parser and GDELT DOC 2.0 API client
│   ├── scheduler.py                         # Background daemon for periodic ingestion & mutex lease locks
│   ├── schemas.py                           # Strongly typed Pydantic models & validation schemas
│   ├── text_cleaning.py                     # Anchor-bounded headline sanitization & Unicode normalization
│   └── ocr.py                               # Lead image text extraction fallback via EasyOCR
│
├── models/                                  # Trained neural network artifacts
│   ├── rhetoric_adapter_gated.pt            # Approved 1.58M Gated Multimodal Fusion Adapter weights (6.3 MB)
│   └── rhetoric_adapter_gated_config.json   # Hyperparameter configuration & architectural metadata
│
├── data/                                    # Local persistence & evidence corpora
│   ├── news_intelligence.db                 # SQLite database (articles, verifications, registry, locks)
│   └── evidence_corpus.jsonl                # High-IDF BM25 fallback reference evidence corpus
│
├── screenshots/                             # Production workstation screenshots (11 UI views)
│   ├── Screenshot 2026-11-02 202814.png     # Live News Intelligence Header & Feed Ingestion Controls
│   ├── Screenshot 2026-11-02 202829.png     # Sidebar Configuration, Threshold Controls & Active Feeds
│   ├── Screenshot 2026-11-02 202854.png     # System Architecture & Tri-Pillar Decoupling Engine Diagram
│   ├── Screenshot 2026-11-02 202907.png     # Evidence Explorer & Searchable SQLite Store Filter View
│   ├── Screenshot 2026-11-02 202915.png     # Benchmark Evaluation Dashboard (NewsCLIPpings Metrics)
│   ├── Screenshot 2026-11-02 202921.png     # NewsCLIPpings Full Test Confusion Matrix (tau=0.28)
│   ├── Screenshot 2026-11-02 202937.png     # Adversarial Generator Robustness Breakdown (Gens 0-3)
│   ├── Screenshot 2026-11-02 202952.png     # Claim Investigation Workspace (Pristine Concordance)
│   ├── Screenshot 2026-11-02 203016.png     # Claim Investigation Workspace (Falsified Context Mismatch)
│   ├── Screenshot 2026-11-02 203027.png     # Selective Classification Status Banner (High Margin Accept)
│   └── Screenshot 2026-11-02 203039.png     # Rhetorical Framing & Modality Salience Weight Breakdown
│
├── docs/                                    # Technical, academic and portfolio documentation
│   ├── portfolio/                           # Recruiter-facing deep dives & specifications
│   │   ├── ARCHITECTURE.md                  # Comprehensive 4-layer architecture & data flow
│   │   ├── DATASETS_AND_METHODOLOGY.md      # NewsCLIPpings, VLDBench, syndication graph, sanitization
│   │   ├── EXPERIMENTAL_RESULTS.md          # Full empirical tables, confusion matrices, CIs, ablation
│   │   ├── DATABASE_DESIGN.md               # SQLite 3 schema, indexes, concurrency tuning, queries
│   │   ├── INSTALLATION_GUIDE.md            # Windows PowerShell step-by-step setup and test guide
│   │   ├── LIMITATIONS_AND_FUTURE_WORK.md   # Failure modes, Gen 0 blindspot, selective coverage trade
│   │   ├── REFERENCES.md                    # Academic literature & BibTeX citations
│   │   ├── PROJECT_STRUCTURE.md             # This file: structural component inventory
│   │   ├── MODEL_CARD.md                    # Standardized ML model cards (DeBERTa, OpenCLIP, Adapter)
│   │   └── figures/                         # High-resolution (300 DPI) publication-quality charts
│   │       ├── system_architecture.png      # 4-Layer architectural diagram
│   │       ├── tri_pillar_engine.png        # Tri-Pillar decoupled verification workflow
│   │       ├── rag_nli_workflow.png         # Pillar 1 RAG NLI pipeline
│   │       ├── cross_modal_alignment.png    # Pillar 2 OpenCLIP hypersphere geometry & score densities
│   │       ├── selective_classification.png # Chow's reject option ambiguity rejection band
│   │       ├── gated_adapter_architecture.png# Pillar 3 1.58M Gated Multimodal Fusion Adapter
│   │       ├── end_to_end_workflow.png      # Operational sequence workflow
│   │       ├── newsclippings_performance.png# Full test confusion matrix & generator breakdown
│   │       ├── risk_coverage_curve.png      # Selective classification accuracy & risk curves
│   │       ├── vldbench_comparison.png      # VLDBench architecture comparison & shift robustness
│   │       ├── research_lifecycle.png       # 9-Phase chronological research timeline
│   │       └── sqlite_database_er_diagram.png# Complete database entity-relationship diagram
│   └── assets/                              # Documentation source assets
│
├── tests/                                   # Automated testing suite (122 tests passed)
│   ├── test_submission_pipeline.py          # End-to-end pipeline integration verification
│   ├── test_benchmark_infrastructure.py     # NewsCLIPpings evaluation harness tests
│   ├── test_universal_verification.py       # Fact-checking RAG & proposition extraction tests
│   ├── test_rhetoric_integration.py         # Rhetoric adapter loading, salience & inference tests
│   ├── test_news_intelligence.py            # RSS ingestion, database transactions & deduplication
│   ├── test_final_app_smoke.py              # Streamlit UI tab rendering & smoke tests
│   └── verify_all_metrics.py                # Telemetry validator asserting exact JSON metrics
│
└── outputs/                                 # Verified empirical benchmark results
    ├── benchmark_results/                   # NewsCLIPpings task outputs (tau=0.28, selective metrics)
    └── performance_reconciliation.json      # Profiling logs resolving local vs. web latency
```
