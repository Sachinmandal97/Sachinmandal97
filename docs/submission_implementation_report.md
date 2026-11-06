# MSc Project Submission Implementation Report

**Project Title:** Explainable Evidence-Augmented Multimodal Misinformation Verification  
**Degree Programme:** MSc Data Science Final-Year Research Project  
**Candidate Name:** MSc Data Science Candidate  
**Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission`  
**Date of Submission:** October 2026

---

## 1. Executive Summary

This report documents the urgent implementation and delivery of a fully functional, reproducible, and verifiable multimodal misinformation verification platform prepared for academic assessment.

Following an exhaustive audit of two public candidate repositories:
- **Repository A:** `kmist70/fake-news-detector`
- **Repository B:** `LaughingParrot/CLIP-Multimodal-Fake-News-Detection-with-InfoNCE-on-Fakeddit`

neither repository could be deployed "as-is" because both omit critical trained model checkpoints from their Git trees, lack external evidence retrieval, and fail to satisfy our research mandate (Repository A does not support image uploads; Repository B is built in Tkinter rather than Streamlit).

To deliver a working, professional system without violating academic integrity:
1. We adopted **Repository A's Streamlit dashboard paradigm** (dark aesthetic, clean tabs, metric cards).
2. We integrated **Repository B's vision-language multimodal alignment concept** (explicit OpenCLIP cosine similarity and modality interaction signals).
3. We connected both to our **audited 8-stage verification pipeline** (spaCy decomposition, BM25 retrieval, DeBERTa-v3 NLI, uncertainty bounds, and structured explainability).
4. We preserved the original research workspace (`multimodal-misinfo-verifier`) completely untouched, delivering this submission inside `multimodal-misinfo-submission/`.

---

## 2. Repository Selection & Architectural Synthesis

### 2.1 Why Repository A Was Not Used Directly
* **Missing Checkpoints:** `loaders.py` explicitly executes `torch.load("model/text_classifier.pt")`. Because `.pt` files were excluded from Git, launching the application causes an immediate fatal `FileNotFoundError`.
* **Missing Image Modality:** Repository A only analyzes text articles and `.mp4` video files. It has no mechanism for uploading or inspecting single static images.
* **Heavy Video Pipeline:** The video stream requires external system binaries (`ffmpeg`, `openai-whisper`, `librosa`), which are brittle on Windows CPU and irrelevant to static image-caption verification.

### 2.2 Why Repository B Was Not Used Directly
* **Tkinter vs Streamlit:** Built as a desktop GUI using `customtkinter` (2,143 lines), incompatible with modern browser-based web deployment.
* **Missing Checkpoint Behavior:** Repository B also lacks `multimodal_model.pt`. When the checkpoint is missing, it falls back to randomly initialized classifier weights (`status = "Models Ready (Warning: untrained weights)"`), generating arbitrary predictions that violate academic evaluation standards.
* **No Retrieval Module:** Focuses solely on cross-modal pairwise classification without external knowledge grounding or RAG.

### 2.3 The Synthesized Foundation
We engineered a hybrid architecture combining the best attributes of both repositories with our tested research engine:
* **UI/UX Layer:** Modern Streamlit application (`app.py`) featuring high-contrast academic dashboard styling, reactive sliders, one-click preset scenarios, and metric badges.
* **Multimodal Alignment:** OpenCLIP (`ViT-B/32`) computing cosine alignment between image vectors and text assertions ($s \in [-1, 1]$), flagging semantic divergence below empirical cutoffs.
* **Evidence Retrieval:** BM25Okapi indexing verified context documents from an explicit knowledge base (`evidence_corpus.jsonl`).
* **Natural Language Inference:** Pretrained DeBERTa-v3 cross-encoder (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) predicting entailment, neutral, and contradiction probabilities.
* **Uncertainty & Attribution:** Tri-state adjudication (`Supported`, `Misleading`, `Uncertain`) with complete structured factor breakdown and explicit uncertainty disclaimers.

---

## 3. Directory Layout & Files Created

```
multimodal-misinfo-submission/
├── app.py                             # Interactive Streamlit application
├── configs/
│   └── config.yaml                    # System configuration & thresholds
├── data/
│   └── corpus/
│       └── evidence_corpus.jsonl      # Verified reference knowledge base
├── docs/
│   ├── repository_comparison.md       # Comparative technical audit of Repos A & B
│   ├── submission_implementation_report.md # This document
│   └── viva_demonstration_guide.md    # Viva presentation walkthrough & Q&A
├── src/
│   ├── __init__.py
│   ├── cross_modal.py                 # OpenCLIP vision-language cosine alignment
│   ├── decomposition.py               # spaCy claim & entity parsing
│   ├── explainability.py              # Structured evidence attribution report
│   ├── ocr.py                         # OCR text overlay extraction
│   ├── pipeline.py                    # Master 8-stage verification pipeline
│   ├── preprocess.py                  # Text cleaning & PIL image validation
│   ├── retrieval.py                   # BM25Okapi lexical retriever
│   ├── schemas.py                     # Pydantic data schemas
│   ├── uncertainty.py                 # Predictive uncertainty & confidence scoring
│   ├── utils.py                       # Logger & hardware profiler
│   └── verification.py                # DeBERTa-v3 NLI & multimodal fusion
├── tests/
│   ├── __init__.py
│   └── test_submission_pipeline.py    # Unit & integration regression tests
├── LICENSE                            # MIT License + third-party attributions
├── README.md                          # Windows setup, run commands & overview
├── requirements.txt                   # Verified package dependencies
└── run_tests.py                       # Automated test discovery runner
```

---

## 4. Verification & Testing Results

The submission test suite (`run_tests.py`) was executed on local Windows 11 CPU:

| Test Case | Target Subsystem | Result | Execution Time |
| :--- | :--- | :--- | :--- |
| `test_clean_text_normalization` | Preprocessing / Unicode cleaning | **PASS** | < 0.01s |
| `test_empty_input_handling` | Robustness on blank/missing text | **PASS** | ~0.15s |
| `test_image_loading_and_cross_modal` | PIL loading & OpenCLIP alignment | **PASS** | ~6.50s |
| `test_evidence_retrieval` | BM25 corpus search & snippet ranking | **PASS** | ~0.05s |
| `test_end_to_end_verification_decision` | Full 8-stage pipeline & DeBERTa NLI | **PASS** | ~40.80s |

**Total Suite Result:** **5 PASSED, 0 FAILED** in 47.514s.

---

## 5. Model Weights & Cache Provenance

* **OpenCLIP ViT-B/32:** Pretrained weights loaded via HuggingFace Hub (`openai/clip-vit-base-patch32`), cached locally at `~/.cache/huggingface/hub/`.
* **DeBERTa-v3 NLI:** Pretrained weights loaded via HuggingFace Hub (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`), cached locally at `~/.cache/huggingface/hub/`.
* **Zero Untrained Weight Fallbacks:** Unlike Repository B, all model predictions in this submission originate from verified, peer-reviewed pretrained transformer weights.

---

## 6. Academic Limitations & Ethical Disclaimers

1. **Alignment vs Truth:** High OpenCLIP similarity establishes that an image matches its caption, not that the asserted event occurred. Factual truth requires corroborating evidence.
2. **Heuristic Confidence:** The reported confidence score is a deterministic rule-based metric combining NLI margin and CLIP distance. It is not an epistemic Bayesian posterior.
3. **Corpus Boundaries:** BM25 retrieval is constrained by the indexed knowledge base. Novel real-world events absent from the corpus will correctly output `Uncertain`.
