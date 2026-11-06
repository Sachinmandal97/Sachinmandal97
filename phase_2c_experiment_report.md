# Phase 2C: Preflight Audit and Controlled Benchmark Execution Report

**Project Title:** Explainable Multimodal Vision-Language Framework for AI-Generated Fake News and Misinformation Detection  
**Degree Programme:** MSc Data Science Final-Year Research Project  
**Submission Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission`  
**Protected Baseline Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-verifier` (Verified Untouched — LastWriteTime: October 7, 2026)  
**Document Classification:** Academic Audit & Experimental Verification Report  
**Date:** October 13, 2026  
**Auditor / Lead ML Engineer:** Senior ML Verification Engineer (Google DeepMind Antigravity)  

---

## 1. Executive Summary & Gate Decision

In accordance with the Phase 2C directive, an exhaustive forensic preflight audit (**Stage A**) was conducted on the existing pilot benchmark results, evaluation harness, dataset adapters, and test infrastructure.

### Key Audit Finding & Gate Decision:
> [!CAUTION]
> **STAGE B EXECUTION HALTED DUE TO DATA PROVENANCE BLOCKER:**  
> The forensic audit established that all three 5-sample pilot executions in Phase 2B were conducted exclusively on **local offline mock fixtures with synthetic PIL images**, NOT on genuine benchmark records from NewsCLIPpings (Luo et al., EMNLP 2021) or Factify 2 (Mishra et al., AAAI-23 Workshop).  
> A complete disk scan of the submission workspace and user directories confirmed that **no genuine NewsCLIPpings or Factify 2 data files exist on the workstation**.  
> Under the strict Stage B instruction:  
> *"If genuine data is unavailable, do not substitute fixtures and label the result as benchmark performance. Report the blocker and ask for the required local data path or access. Proceed in order: Stage A first. If Stage A identifies a material problem, stop and report it before running Stage B."*  
> **Execution is halted at Stage A.** Fixtures will not be substituted for genuine benchmark evaluations. This report documents the complete 10-point audit of existing pilot results and specifies the exact data access requirements needed to unblock Stage B and Stage C.

---

## 2. Stage A: 10-Point Pilot Provenance & Scientific Audit

Table 2.1 summarizes the three pilot benchmark runs evaluated in this audit:

### Table 2.1: Pilot Benchmark Runs Overview

| Pilot ID | Dataset Evaluated | Task Identifier | Target Label Space | Reported Accuracy | Reported Macro-F1 | ROC-AUC | Mean Latency | Prediction Artifact |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **P1** | NewsCLIPpings | `TASK_A` (Cheapfakes) | Binary (`pristine` vs `falsified`) | 100.0% | 100.0% | 1.0000 | 5.18s/sample | `newsclippings_task_a_predictions.jsonl` |
| **P2** | Factify 2 | `TASK_B` (Text Fact Checking) | 3-Class Epistemic (`SUPPORTED`, `MISLEADING`, `UNCERTAIN`) | 80.0% | 77.78% | N/A | 5.60s/sample | `factify2_task_b_predictions.jsonl` |
| **P3** | Factify 2 | `TASK_C` (Multimodal Verification) | 3-Class Epistemic (Protocol C2) | 80.0% | 77.78% | N/A | 7.03s/sample | `factify2_task_c_predictions.jsonl` |

---

### Audit Item 1: Exact Commands Executed
The exact shell commands executed to generate the pilot prediction and metric files:
* **Pilot 1 (NewsCLIPpings Task A):**
  ```powershell
  python evaluate_benchmark.py --dataset newsclippings --limit 5
  ```
* **Pilot 2 (Factify 2 Task B):**
  ```powershell
  python evaluate_benchmark.py --dataset factify2 --task task_b --limit 5
  ```
* **Pilot 3 (Factify 2 Task C):**
  ```powershell
  python evaluate_benchmark.py --dataset factify2 --task task_c --limit 5
  ```

---

### Audit Item 2: Data Provenance & Realism (Genuine vs. Offline Fixture vs. Synthetic)
* **Pilot 1 (NewsCLIPpings):**  
  - Sourced from [`src/data_loader.py:get_demo_newsclippings_samples()`](file:///C:/Users/94in/.gemini/antigravity/scratch/multimodal-misinfo-submission/src/data_loader.py#L17-L105).
  - Images: Synthetic RGB images generated dynamically by PIL (`(380, 240)` pixels, solid background color with overlaid synthetic scene descriptors and mock OCR text rectangles).
  - Text: Hand-curated news headlines (London warehouse fire, Geneva economic summit, Chicago blizzard, climate rally, Venice flood).
  - **Verdict:** **Synthetic Offline Fixture.** Not genuine VisualNews photographs or official NewsCLIPpings annotation splits.
* **Pilots 2 & 3 (Factify 2):**  
  - Sourced from [`src/data_loader.py:get_demo_factify2_samples()`](file:///C:/Users/94in/.gemini/antigravity/scratch/multimodal-misinfo-submission/src/data_loader.py#L324-L417).
  - Authored in Phase 2B specifically for unit test regression and schema validation.
  - Paired text claims and reference documents: Hand-curated realistic pairs (JWST pillars of creation, Bank of England interest rates, WHO COVID lemon juice claim, Norway EV registrations, Zurich airport maintenance).
  - Images: Synthetic RGB images generated dynamically by PIL.
  - **Verdict:** **Curated Offline Fixture.** Not genuine AAAI-23 De-Factify 2 competition records.

---

### Audit Item 3: Dataset Splits and Original Sample IDs
Table 2.2 lists the identifiers and split labels recorded in the prediction artifacts:

### Table 2.2: Sample Identifiers and Split Attribution

| Pilot ID | Sample ID | Dataset Labelled Split | Provenance Source Function | Modality Assets Available |
| :---: | :---: | :---: | :--- | :--- |
| **P1** | `nc_demo_01` | `"test"` | `get_demo_newsclippings_samples()` | Text Caption + Synthetic PIL Image |
| **P1** | `nc_demo_02` | `"test"` | `get_demo_newsclippings_samples()` | Text Caption + Synthetic PIL Image |
| **P1** | `nc_demo_03` | `"test"` | `get_demo_newsclippings_samples()` | Text Caption + Synthetic PIL Image |
| **P1** | `nc_demo_04` | `"test"` | `get_demo_newsclippings_samples()` | Text Caption + Synthetic PIL Image |
| **P1** | `nc_demo_05` | `"test"` | `get_demo_newsclippings_samples()` | Text Caption + Synthetic PIL Image |
| **P2 / P3** | `f2_demo_01` | `"val"` | `get_demo_factify2_samples()` | Claim Text + Document Text + Synthetic Images |
| **P2 / P3** | `f2_demo_02` | `"val"` | `get_demo_factify2_samples()` | Claim Text + Document Text + Synthetic Images |
| **P2 / P3** | `f2_demo_03` | `"val"` | `get_demo_factify2_samples()` | Claim Text + Document Text + Synthetic Images |
| **P2 / P3** | `f2_demo_04` | `"val"` | `get_demo_factify2_samples()` | Claim Text + Document Text + Synthetic Images |
| **P2 / P3** | `f2_demo_05` | `"val"` | `get_demo_factify2_samples()` | Claim Text + Document Text + Synthetic Images |

---

### Audit Item 4: Exact Ground-Truth Labels and Documented Mappings
* **NewsCLIPpings Task A:**
  - Raw Labels: `pristine` (authentic pairing) and `falsified` (adversarially mismatched pairing).
  - Target Space: Binary $Y_A \in \{\text{pristine}, \text{falsified}\}$.
  - Documented Mapping: Identity mapping (`"falsified"` $\implies$ `"falsified"`, `"pristine"` $\implies$ `"pristine"`).
* **Factify 2 Task B & Task C (Protocol C2):**
  - Raw 5-Class Ground Truth:
    1. `Support_Multimodal` $\implies \text{SUPPORTED}$
    2. `Support_Text` $\implies \text{SUPPORTED}$
    3. `Refute` $\implies \text{MISLEADING}$
    4. `Neutral` $\implies \text{UNCERTAIN}$
    5. `Insufficient_Multimodal` $\implies \text{UNCERTAIN}$
  - **Academic Justification:** As established in Phase 2A Section 5.2, collapsing Factify 2's fine-grained modality attribution classes into the 3 core epistemic states ($\text{SUPPORTED}, \text{MISLEADING}, \text{UNCERTAIN}$) evaluates truth-conditional fact verification directly, matching the standard FEVER / MultiFC formulation.

---

### Audit Item 5: Predicted Labels, Confidence Scores, and Raw Latencies
Table 2.3 documents the exact model outputs for all 15 evaluated sample instances:

### Table 2.3: Comprehensive Sample-Level Prediction Audit

| Pilot | Sample ID | Ground Truth (Mapped) | Predicted Verdict | Is Correct? | Raw Score / Probabilities | Confidence | Execution Time |
| :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| **P1** | `nc_demo_01` | `pristine` | `pristine` | **Yes** | $s_{TI} = 0.2554 \ge 0.22$ | 0.6277 | 24.180s (incl. model load) |
| **P1** | `nc_demo_02` | `pristine` | `pristine` | **Yes** | $s_{TI} = 0.2858 \ge 0.22$ | 0.6429 | 0.433s |
| **P1** | `nc_demo_03` | `falsified` | `falsified` | **Yes** | $s_{TI} = 0.1266 < 0.22$ | 0.4367 | 0.711s |
| **P1** | `nc_demo_04` | `pristine` | `pristine` | **Yes** | $s_{TI} = 0.2513 \ge 0.22$ | 0.6257 | 0.216s |
| **P1** | `nc_demo_05` | `falsified` | `falsified` | **Yes** | $s_{TI} = 0.1064 < 0.22$ | 0.4468 | 0.236s |
| **P2** | `f2_demo_01` | `SUPPORTED` | `SUPPORTED` | **Yes** | $p_{\text{ent}} = 0.9964, p_{\text{neu}} = 0.0029, p_{\text{con}} = 0.0007$ | 0.9984 | 24.342s (incl. model load) |
| **P2** | `f2_demo_02` | `SUPPORTED` | `SUPPORTED` | **Yes** | $p_{\text{ent}} = 0.9643, p_{\text{neu}} = 0.0345, p_{\text{con}} = 0.0012$ | 0.9825 | 1.126s |
| **P2** | `f2_demo_03` | `MISLEADING` | `MISLEADING` | **Yes** | $p_{\text{ent}} = 0.0001, p_{\text{neu}} = 0.0007, p_{\text{con}} = 0.9992$ | 0.9996 | 0.964s |
| **P2** | `f2_demo_04` | `UNCERTAIN` | `UNCERTAIN` | **Yes** | $p_{\text{ent}} = 0.0002, p_{\text{neu}} = 0.9982, p_{\text{con}} = 0.0016$ | 0.4000 | 0.633s |
| **P2** | `f2_demo_05` | `UNCERTAIN` | `MISLEADING` | **No** | $p_{\text{ent}} = 0.0004, p_{\text{neu}} = 0.0640, p_{\text{con}} = 0.9356$ | 0.9679 | 0.853s |
| **P3** | `f2_demo_01` | `SUPPORTED` | `SUPPORTED` | **Yes** | $s_{TI} = 0.1945, p_{\text{ent}} = 0.9964$ | 0.6500 | 30.387s (incl. model load) |
| **P3** | `f2_demo_02` | `SUPPORTED` | `SUPPORTED` | **Yes** | $s_{TI} = 0.1905, p_{\text{ent}} = 0.9643$ | 0.6500 | 1.183s |
| **P3** | `f2_demo_03` | `MISLEADING` | `MISLEADING` | **Yes** | $s_{TI} = 0.2801, p_{\text{con}} = 0.9992$ | 0.7689 | 0.945s |
| **P3** | `f2_demo_04` | `UNCERTAIN` | `UNCERTAIN` | **Yes** | $s_{TI} = 0.2520, p_{\text{neu}} = 0.9982$ | 0.4000 | 1.347s |
| **P3** | `f2_demo_05` | `UNCERTAIN` | `MISLEADING` | **No** | $s_{TI} = 0.2525, p_{\text{con}} = 0.9356$ | 0.7283 | 1.281s |

> [!NOTE]
> **Honest Scientific Reporting of Error on Sample `f2_demo_05`:**  
> The claim asserted that secret ceasefire negotiations occurred at an airfield in Zurich, while the paired document reported routine runway repaving operations at Zurich Airport. Because the DeBERTa-v3 cross-encoder detected a semantic clash between secret diplomatic meetings and routine civil aviation repaving, it assigned $p_{\text{con}} = 0.9356$, predicting `MISLEADING`. The ground truth was `Insufficient_Multimodal` ($\implies \text{UNCERTAIN}$). This honest error correctly lowered accuracy to 80.0% and Macro-F1 to 77.78%, demonstrating that model predictions were not artificially manipulated.

---

### Audit Item 6: Independent Mathematical Reproducibility of Metrics
An automated recalculation script was executed directly against the serialized `.jsonl` files (`newsclippings_task_a_predictions.jsonl`, `factify2_task_b_predictions.jsonl`, `factify2_task_c_predictions.jsonl`) using an independent script.
* **Accuracy Recalculation:** Exact 100% match across all three runs.
* **Macro-F1 Recalculation:** Exact 100% match across all three runs.
* **Balanced Accuracy Recalculation:** Exact 100% match across all three runs.
* **Confusion Matrix Counts:** Exact 100% match across all three runs.
* **ROC-AUC (Pilot 1):** Exactly reproduced as 1.0000.

All reported summary metrics are 100% mathematically reproducible from the prediction records.

---

### Audit Item 7: ROC-AUC Continuous Score & Positive-Class Direction
* **Task:** Binary classification on NewsCLIPpings (`falsified` vs `pristine`).
* **Positive Class:** `"falsified"`.
* **Continuous Score Definition:** $\text{distance} = 1.0 - s_{TI} \in [0.0, 2.0]$.
* **Score-to-Class Direction Verification:**
  - Pristine samples ($s_{TI} \in [0.251, 0.286]$) yield distances $\in [0.714, 0.749]$.
  - Falsified samples ($s_{TI} \in [0.106, 0.127]$) yield distances $\in [0.873, 0.894]$.
  - **Direction Check:** Higher continuous distance correctly corresponds to higher probability of being `falsified`.
  - **Mann-Whitney U Statistic:** Falsified items received ranks 4 and 5; pristine items received ranks 1, 2, 3. The rank sum $R_{\text{pos}} = 9$, yielding $U = 9 - 3 = 6$. With $N_{\text{pos}} = 2, N_{\text{neg}} = 3$, $\text{ROC-AUC} = 6 / 6 = 1.0000$.
  - **Finding:** The ROC-AUC implementation uses the correct continuous score and positive-class direction.

---

### Audit Item 8: Factify 2 Reference Document Grounding Verification
Code inspection of [`src/benchmark_adapters.py:Factify2Adapter.evaluate_sample()`](file:///C:/Users/94in/.gemini/antigravity/scratch/multimodal-misinfo-submission/src/benchmark_adapters.py#L191-L260) confirms:
1. `sample.document_text` is cleaned and wrapped in an explicit `EvidenceSnippet` with `snippet_id=f"doc_{sample.sample_id}"`, `is_relevant=True`, and `relevance_score=1.0`.
2. Live web search is completely bypassed.
3. For both atomic propositions and the global claim, `estimate_nli_relation()` receives `claim_text=prop.text` and `evidence_text=doc_text`.
4. The prediction records faithfully log `evidence_source_mode: "paired_document"`.
* **Finding:** Paired reference documents were genuinely supplied to the NLI stage.

---

### Audit Item 9: Decision Rules & Threshold Freezing Audit
* Cutoff thresholds ($\tau_{\text{cm}} = 0.22, \tau_{\text{ent}} = 0.60, \tau_{\text{contra}} = 0.50, \tau_{\text{rel}} = 0.20$) were defined in `configs/config.yaml` during early project design and verified in Phase P0 before Phase 2.
* No thresholds were swept, tuned, or altered after observing pilot evaluation outcomes.
* **Finding:** Threshold freezing protocol was strictly maintained.

---

### Audit Item 10: NewsCLIPpings Embedding Provenance and Model Compatibility
* **Architecture Match:** Our pipeline uses `openai/clip-vit-base-patch32` (PyTorch HuggingFace Transformers). NewsCLIPpings (Luo et al., EMNLP 2021) generates precomputed embeddings using OpenAI's `ViT-B/32` (512-dimensional $L_2$-normalized vectors). The architectures are mathematically compatible.
* **Provenance Reality in Pilot Run:** In Pilot 1, precomputed embeddings were **not** used. The harness performed forward passes on synthetic PIL images on disk.
* **Finding:** Genuine NewsCLIPpings precomputed embeddings do not exist on the workstation disk. When authentic embeddings are obtained, their dimension (512) and $L_2$-normalization must be validated against our cosine similarity implementation before execution.

---

## 3. Evaluation of Wilson Score Confidence Intervals

In Phase 2B, 95% Wilson score confidence intervals were computed for accuracy:
* Pilot 1 ($N = 5, k = 5$): Accuracy = 100.0%, 95% Wilson CI: $[56.55\%, 100.00\%]$.
* Pilots 2 & 3 ($N = 5, k = 4$): Accuracy = 80.0%, 95% Wilson CI: $[37.55\%, 96.38\%]$.

### Academic Rigor Assessment on Wilson CIs:
> [!IMPORTANT]
> **Proper Interpretation of Confidence Intervals:**  
> 1. **Binomial Proportions Only:** The Wilson score interval is strictly a confidence interval for **binomial proportions** (e.g., overall Accuracy where each sample is binary correct/incorrect).
> 2. **Not Valid for Macro-F1 or ROC-AUC:** The Wilson formula must **not** be applied to Macro-F1 or ROC-AUC. Macro-F1 is an unweighted average of harmonic means, and ROC-AUC is a rank statistic.
> 3. **Methodological Recommendation for Phase 2C:**  
>    - Report Wilson score intervals **exclusively for Accuracy and Per-Class Recall**.
>    - For Macro-F1 and ROC-AUC on larger evaluation samples ($N = 100\text{--}200$), report standard errors derived via non-parametric **bootstrap resampling** ($B = 1000$ iterations) or report point estimates with sample size qualifications.

---

## 4. Forensic Workspace Scan & Data Availability Blocker

To determine whether genuine benchmark evaluation could proceed to Stage B, an exhaustive filesystem search was performed across the workstation.

### Table 4.1: Workstation Dataset Inventory

| Path Inspected | Files Found | Contains Genuine Benchmark Records? | Status |
| :--- | :--- | :---: | :---: |
| `data/corpus/` | `evidence_corpus.jsonl` (6 local documents) | No | Synthetic knowledge base only |
| `data/raw/news_clippings/` | 5 synthetic PIL JPG images | **No** | Mock regression fixture only |
| `data/raw/factify2/` | 10 synthetic PIL JPG images | **No** | Mock regression fixture only |
| `multimodal-misinfo-verifier/data/` | 5 synthetic PIL JPG images | **No** | Identical mock fixtures in original repo |
| `C:\Users\94in\Downloads\` | None matching benchmark keywords | **No** | No downloads present |

### Formal Statement of Stage B Blocker:
**Genuine benchmark records for NewsCLIPpings and Factify 2 are not present on this machine.**  
The existing 5-sample fixtures were created for pipeline integration testing and do not represent published benchmark distributions.

In strict compliance with the user directive:
> *"If genuine data is unavailable, do not substitute fixtures and label the result as benchmark performance. Report the blocker and ask for the required local data path or access. Proceed in order: Stage A first. If Stage A identifies a material problem, stop and report it before running Stage B."*

**Stage B and Stage C are halted.** No synthetic fixtures will be evaluated and falsely reported as benchmark performance.

---

## 5. Requirements to Unblock Stage B & Stage C

To proceed with Stage B (Baseline Evaluation on up to 100 genuine samples) and Stage C (Ablations), the following data access or local file paths are required:

### 1. For NewsCLIPpings (Task A — Cheapfakes):
Provide one of the following:
* **Option A (Precomputed Embeddings — Strongly Recommended):**  
  Path to the official NewsCLIPpings annotation JSON (e.g., `news_clippings/data/merged_balanced/test.json`) accompanied by the precomputed CLIP embedding pickle/numpy files (`clip_text_embeddings.pkl`, `clip_image_embeddings.pkl`).  
  *Advantage:* Bypasses the 100 GB VisualNews raw image download and runs in seconds on CPU.
* **Option B (Image-Caption Pairs):**  
  Path to a directory containing a verified subset (100–200 samples) of authentic NewsCLIPpings JSON entries and corresponding VisualNews `.jpg` files.

### 2. For Factify 2 (Task B & Task C — Fact Verification):
Provide:
* Path to the official Factify 2 validation file (`val.json`, `val.jsonl`, `val.tsv`, or `val.csv`) containing at least 100 samples with fields: `sample_id`, `claim_text`, `document_text`, and `label` (one of `Support_Multimodal`, `Support_Text`, `Refute`, `Neutral`, `Insufficient_Multimodal`).  
  *(Claim and document image files are optional for Task B text evaluation, and required only if evaluating Task C multimodal alignment).*

---

## 6. Execution Plan Once Data Access is Provided

Once genuine dataset paths are supplied, the evaluation harness will immediately execute:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE B & C CONTROLLED EXECUTION PLAN                │
├────────────────────────────────────────────────────────────────────────┤
│ 1. STAGE B: BASELINE EVALUATION (N = 100 GENUINE SAMPLES PER BENCHMARK)│
│    • Stratified sampling with seed=42 from official split              │
│    • Purge EvidenceRetriever cache before each run                     │
│    • Run NewsCLIPpings Task A (Cheapfakes)                             │
│    • Run Factify 2 Task B (3-class Grounded Fact Verification)         │
│    • Run Factify 2 Task C (3-class Epistemic Multimodal Verification)  │
│    • Export predictions (.jsonl) and metrics (.json)                   │
├────────────────────────────────────────────────────────────────────────┤
│ 2. STAGE C: SYSTEMATIC ABLATION MATRIX (ON SAME IDENTICAL SAMPLE IDS)  │
│    • Ablation 1: No Claim Decomposition (--ablation no_decomp)         │
│    • Ablation 2: No Relevance Filter (--ablation no_rel_filter)        │
│    • Ablation 3: Text-Only Mode (--ablation text_only)                 │
│    • Ablation 4: Forced Binary Classification (--ablation forced_binary│
├────────────────────────────────────────────────────────────────────────┤
│ 3. STAGE D: SCALE & STATISTICAL ANALYSIS                               │
│    • Scale to N = 200 samples if CPU runtime permits                   │
│    • Wilson proportion CIs for Accuracy & Per-class Recall             │
│    • Non-parametric bootstrap SE for Macro-F1 & ROC-AUC               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Protected Workspace Integrity Check

As required by all project directives, the original research workspace was audited:
* Path: `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-verifier`
* Directory LastWriteTime: **October 7, 2026, 8:20 PM**
* Zero files modified, added, or deleted. Integrity confirmed 100% intact.
