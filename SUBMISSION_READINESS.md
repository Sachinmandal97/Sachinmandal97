# MSc Data Science Research Project — Submission Readiness Document

**Project Title:** Explainable Multimodal Vision-Language Framework for AI-Generated Fake News and Misinformation Detection  
**Candidate:** Candidate (MSc Data Science)  
**Date of Audit:** October 13, 2026  
**Submission Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission`  
**Protected Baseline Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-verifier` (*Untouched, timestamp 7/10/2026 8:20:55 PM*)  
**Overall Readiness Verdict:** **SUBMISSION READY (Software & Task A Benchmark Frozen)**

---

## 1. Executive Summary

This document certifies that the software implementation, empirical benchmarking, and verification audit for the MSc Data Science dissertation project are **complete, fully reproducible, and frozen**.

* **Task A Empirical Benchmark (NewsCLIPpings, $N=7,264$):** Evaluated on the official, untouched test split. Calibrated threshold ($\tau=0.28$) achieves **63.53% Accuracy** (+8.85% over baseline) and **62.75% Macro-F1** (+18.60% over baseline), with **ROC-AUC = 0.7197**. Edwards continuity-corrected McNemar test confirms the gain is statistically significant ($\chi^2 = 196.3621, p = 1.30 \times 10^{-44}$).
* **Application Workflow & Smoke Tests:** 7/7 critical application workflows passed (multimodal input, text-only input, uncertain evidence, conflicting evidence, network fallback, malformed input, and preset visual isolation).
* **Automated Regression Test Suite:** **62 / 62 tests passed (100% pass rate)** on Python 3.13.5 across all 8 test modules.
* **Research Integrity & Factify 2 Status:** Factify 2 is formally documented as an **access-blocked experiment** due to external CodaLab/Google Forms authentication barriers. No synthetic or mock data was fabricated or presented as benchmark findings.
* **Workspace Preservation:** The protected original baseline workspace has remained completely untouched throughout all phases.

---

## 2. Inventory of Primary Deliverables

| Deliverable Asset | Location / Relative Path | Description & Verification Status |
| :--- | :--- | :--- |
| **Streamlit Web Application** | `app.py` | Interactive verification dashboard with multi-stage RAG, live search, and visual attribution. |
| **Core Verification Modules** | `src/*.py` (15 modules) | Preprocessing, OCR, OpenCLIP, spaCy decomposition, BM25 retrieval, DeBERTa NLI, epistemic fusion, and uncertainty. |
| **Benchmark Harness & Adapters** | `evaluate_benchmark.py`, `src/benchmark_adapters.py`, `src/evaluation_harness.py` | Standalone evaluation engine supporting NewsCLIPpings (Task A) and Factify 2 (Tasks B & C) with 4 ablation switches. |
| **Automated Test Suites** | `tests/*.py` (8 modules, 62 tests) | Comprehensive unit, integration, and smoke test suites covering all edge cases. |
| **Raw Benchmark Datasets** | `data/raw/newsclippings/` (8 files, ~418 MB) | Bit-for-bit SHA-256 authentic NewsCLIPpings validation and test annotations, metadata, and OpenCLIP embeddings. |
| **Saved Benchmark Outputs** | `outputs/benchmark_results/` (6 run directories) | Immutable per-sample JSONL predictions and metric JSON summaries for test ($N=7,264$), test ($N=200$), and validation runs. |
| **Final Evaluation Report** | `final_research_evaluation_report.md` | Formal 10-section academic research report with full mathematical formulations and tables. |
| **Verification Checklist** | `FINAL_VERIFICATION_CHECKLIST.md` | Detailed audit checklist verifying every statistic, metric, and test outcome. |
| **System Documentation** | `README.md` | Installation, execution, architecture, and limitation reference guide. |

---

## 3. Verified Test Suite Outcomes

All 62 tests executed and passed on Python 3.13.5 (PyTest 9.1.1):

| Test Module | Test Count | Pass Rate | Scope Verified |
| :--- | :---: | :---: | :--- |
| `tests/test_final_app_smoke.py` | 7 | **7/7 (100%)** | Image-plus-claim, text-only, uncertain evidence, conflicting evidence, network fallback, malformed inputs, preset isolation |
| `tests/test_benchmark_infrastructure.py` | 9 | **9/9 (100%)** | Wilson score CIs, Mann-Whitney ROC-AUC, stratified sampling, schema validation, genuine data loaders |
| `tests/test_phase_p0_fixes.py` | 8 | **8/8 (100%)** | Root URL rejection, unresolved wrapper handling, long claim tokenization, text-only decoupling |
| `tests/test_evidence_hotfix.py` | 8 | **8/8 (100%)** | Live fetch throttling, snippet preservation, content extraction guards |
| `tests/test_submission_pipeline.py` | 1 | **1/1 (100%)** | End-to-end pipeline initialization and offline fallback execution |
| `tests/test_nli_diagnostic.py` | 9 | **9/9 (100%)** | DeBERTa NLI premise scoring, contradiction precedence, batch inference |
| `tests/test_universal_fact_checking.py` | 10 | **10/10 (100%)** | Multi-category claim decomposition, epistemic verdict aggregation, truth-assumption neutrality |
| `tests/test_universal_verification.py` | 10 | **10/10 (100%)** | Provider parsing, query TTL caching, irrelevant evidence filtering, decoupled fusion |
| **TOTAL** | **62** | **62/62 (100%)** | **Complete submission codebase verified with 0 failures** |

---

## 4. Key Experimental Findings (NewsCLIPpings Benchmark)

All metrics were independently recalculated directly from the 7,264 per-sample prediction records stored in `outputs/benchmark_results/`:

### 4.1 Test Split Benchmark Performance ($N=7,264$):
* **Baseline Threshold ($\tau=0.22$):** Accuracy: **54.68%**, Macro-F1: **44.15%**, Balanced Accuracy: **54.68%**, 95% Wilson CI: `[53.53%, 55.82%]`.
* **Calibrated Threshold ($\tau=0.28$):** Accuracy: **63.53%**, Macro-F1: **62.75%**, Balanced Accuracy: **63.53%**, 95% Wilson CI: `[62.42%, 64.63%]`.
* **Empirical Gain ($\Delta$):** **+8.85% Accuracy**, **+18.60% Macro-F1**.
* **ROC-AUC (Rank-Order Metric):** **0.7197** on full test split ($N=7,264$) vs. **0.7286** on validation split ($N=200$). Generalization gap is $< 0.009$.

### 4.2 McNemar's Paired Significance Test:
* **Discordant Pairs ($b + c$):** 2,099 ($b=728$, $c=1,371$).
* **Edwards Continuity-Corrected McNemar Statistic:**
  $$\chi^2 = \frac{(|728 - 1,371| - 1)^2}{2,099} = \frac{642^2}{2,099} = \mathbf{196.3621} \quad (\text{df} = 1)$$
* **Asymptotic $p$-Value:** $p = \mathbf{1.30 \times 10^{-44}}$
* **Exact Binomial Two-Sided $p$-Value:** $p = \mathbf{2.65 \times 10^{-45}}$
* **Conclusion:** The +8.85% accuracy gain of $\tau=0.28$ over $\tau=0.22$ is statistically significant ($p \ll 0.001$).

### 4.3 Adversarial Generator Disaggregation ($N=1,816$ each at $\tau=0.28$):
* **Gen 1 (`semantics_clip_text_text`):** **81.44% Accuracy** (Pristine: 89.1%, Falsified: 73.8%). CLIP excels at flagging entity/semantic mismatches.
* **Gen 2 (`scene_resnet_place`):** **67.79% Accuracy** (Pristine: 72.0%, Falsified: 63.6%). Moderate robustness to scene swaps.
* **Gen 3 (`person_sbert_text_text`):** **61.07% Accuracy** (Pristine: 82.3%, Falsified: 39.9%). Struggles with person identity swaps when semantic context matches.
* **Gen 0 (`semantics_clip_text_image`):** **43.83% Accuracy** (Pristine: 64.9%, Falsified: 22.8%). Fails when adversary explicitly optimizes CLIP similarity on false pairs.
* **Scientific Insight:** Gen 0 illustrates the theoretical limit of unimodal contrastive representations and motivates the necessity of evidence-grounded external retrieval.

---

## 5. Step-by-Step Reproduction Instructions

### Step 1: Environment Setup
```powershell
cd C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2: Run Application Dashboard
```powershell
streamlit run app.py
```

### Step 3: Run Full Automated Test Suite
```powershell
pytest -v
```

### Step 4: Run Independent Metric Verification
```powershell
python tests/verify_all_metrics.py
```

### Step 5: Reproduce Benchmark Evaluations
```powershell
# Full Test Calibrated (tau=0.28)
python evaluate_benchmark.py --dataset newsclippings --task task_a --split test --threshold-cm 0.28 --output-dir outputs/benchmark_results/newsclippings_test_full_tau28

# Adversarial Generator Breakdown
python tests/analyze_test_generators.py
```

---

## 6. Known Limitations & Research Boundaries

1. **Pixel-Level Generative Image Detection (Explicit Non-Claim):** The system detects out-of-context multimodal news pairings (cheapfakes) via OpenCLIP semantic alignment. It does **not** perform pixel-level generative synthesis artifact detection (Midjourney/DALL-E/GAN deepfakes).
2. **Factify 2 Benchmark Blocker:** Multi-premise factual verification (Task B) and 5-class multimodal epistemic adjudication (Task C) remain access-blocked due to CodaLab credential gates. No synthetic data was fabricated.
3. **Pipeline Ablation Scope:** Architectural ablations (`no_decomp`, `no_rel_filter`, `text_only`, `forced_binary`) are reserved for multi-premise evidence retrieval datasets and are not methodologically applicable to single image-caption pairs without retrieval.
4. **Latency Characterization:** The 1.33-second benchmark throughput ($N=7,264$) measures precomputed 512-d vector cosine similarity indexing ($< 0.2$ ms/sample). End-to-end pixel Vision Transformer inference requires 150–300 ms per sample on CPU.
5. **Prevalence Construct:** The 50/50 balance in NewsCLIPpings is an experimental benchmark convention, not representative of open web feeds where false context media is a low-prevalence anomaly ($< 5\%$).

---

## 7. Manual Actions Required Before Dissertation Submission

1. **Incorporate Results into Dissertation Text (Student Action):**
   - Transcribe Section 4.1 (Results Table), Section 4.2 (McNemar's Test), and Section 4.3 (Adversarial Breakdown) into Chapter 4 (Results) and Chapter 5 (Discussion) of the MSc dissertation.
   - Use the clear domain distinction between cheapfake detection and pixel-level AI image detection in Chapter 1 (Introduction) and Chapter 3 (Methodology).
2. **Factify 2 Ingestion (Optional — Only if requested by University Supervisor):**
   - If multi-premise claim verification is required, log into CodaLab Competition 8275, download `val.json`, place in `data/raw/factify2/val.json`, and run:
     ```powershell
     python evaluate_benchmark.py --dataset factify2 --task task_b --split val
     ```
3. **Archive Submission Workspace:**
   - Package `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission` (excluding `.git` or cache directories if size-constrained) for submission to the university submission portal.

---

## 8. Final Sign-Off

* **Software Status:** **Complete, verified, and frozen.**
* **Task A Benchmark:** **Complete, verified, and auditable.**
* **Protected Baseline:** **Untouched (`7/10/2026 8:20:55 PM`).**
* **Development Phase:** **Terminated. No further development required.**
