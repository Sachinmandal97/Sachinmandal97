# Phase 2C — Controlled Benchmark Results and Empirical Calibration Report

**Project Title:** Explainable Multimodal Vision-Language Framework for AI-Generated Fake News and Misinformation Detection  
**Module:** MSc Data Science Research Project  
**Date:** October 13, 2026  
**Status:** Completed (NewsCLIPpings Task A Evaluated on 100 & 200 Samples; Factify 2 Access Gated)  
**Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission`  
**Protected Baseline Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-verifier` (Untouched)  

---

## 1. Executive Summary

This report documents the first controlled, fully reproducible empirical evaluation of the Multimodal Misinformation Framework on **genuine benchmark data** (NewsCLIPpings, Luo et al., EMNLP 2021). All experimental runs were performed strictly in the submission workspace using deterministic sampling on the authentic validation corpus.

### Key Findings
1. **Validation-Set Evaluation Scope:** These evaluations were executed on the **validation split** (`val.json`), not the final test set. They serve as an exploratory calibration baseline to assess cross-modal alignment behaviors and establish decision thresholds for subsequent untouched test evaluation.
2. **Empirical Discriminative Power (ROC-AUC):** The continuous OpenCLIP ViT-B/32 cosine similarity score ($s_{TI}$) exhibits strong discriminative separation between pristine and falsified news pairs, achieving an empirical ROC-AUC of **0.7314** on 100 samples and **0.7286** on 200 samples.
3. **Threshold Calibration Insight:** The *a priori* uncalibrated baseline threshold ($\tau_{\text{cm}} = 0.22$) produced high pristine recall (94–96%) but conservative falsified detection (15–18% recall), yielding an overall accuracy of 56.00% ($N=100$) and 55.50% ($N=200$). Systematic threshold sensitivity analysis demonstrates that calibrating the threshold to $\tau^* \in [0.28, 0.29]$ balances precision and recall, raising balanced accuracy to **67.50%** and Macro-F1 to **67.46%**.
4. **Sample-Size Stability:** Performance remained consistent between $N=100$ and $N=200$ (accuracy shifted by only 0.5%, ROC-AUC by 0.0028), demonstrating sample stability under deterministic stratified sampling.
5. **Ablation Applicability:** None of the four full-pipeline ablations (`no_decomp`, `no_rel_filter`, `text_only`, `forced_binary`) are methodologically applicable to NewsCLIPpings Task A due to the benchmark's isolated visual-semantic cheapfake formulation. These ablations remain reserved for Factify 2.

---

## 2. Dataset Provenance and Sampling Protocol

### 2.1 Benchmark Provenance
* **Dataset:** NewsCLIPpings (Luo et al., Findings of EMNLP 2021).
* **Split Used:** Validation Split (`data/raw/newsclippings/merged_balanced/val.json`, 7,024 samples).
* **Corpus Balancing:** Exactly 3,512 pristine pairs and 3,512 falsified pairs across four adversarial mismatch generators (`semantics_clip_text_image`, `semantics_clip_text_text`, `scene_resnet_place`, `person_sbert_text_text`).
* **Embeddings:** Official 512-dimensional precomputed OpenCLIP ViT-B/32 text and image vectors provided directly by the benchmark authors.

### 2.2 Deterministic Stratified Sampling
Sampling was conducted using the verified `stratified_sample` utility in `src/evaluation_harness.py`:
* **Fixed Pseudo-Random Seed:** `seed = 42` (Mersenne Twister `random.Random(42)`).
* **Class Balancing:** Equal quotas drawn per stratum ($N/2$ pristine, $N/2$ falsified).
* **Determinism:** Bit-for-bit reproducible sample ordering across repeated executions.
* **Sample ID Verification:**
  - 100-Sample First 5 IDs: `nc_val_192915_192915`, `nc_val_672278_672278`, `nc_val_1265767_1265767`, `nc_val_1576873_1576873`, `nc_val_1117864_1117864`.
  - 100-Sample Last 5 IDs: `nc_val_1159647_985345`, `nc_val_737680_1474386`, `nc_val_266214_318870`, `nc_val_169451_169451`, `nc_val_583966_523489`.

---

## 3. Baseline Task A Evaluation: 100 Genuine Samples

### 3.1 Execution Command
```bash
python evaluate_benchmark.py \
  --dataset newsclippings \
  --task task_a \
  --split val \
  --limit 100 \
  --seed 42 \
  --threshold-cm 0.22 \
  --output-dir outputs/benchmark_results/newsclippings_val_100
```

### 3.2 Quantitative Results ($N=100$, $\tau_{\text{cm}} = 0.22$)
* **Timestamp (UTC):** `2026-10-13T16:06:13.380711+00:00`
* **Sample Count:** 100 requested, 100 evaluated, 0 failed (Coverage: 100.0%, Error Rate: 0.0%).
* **Runtime:** 0.99 seconds total latency (mean 0.010 s/sample).
* **Overall Accuracy:** **56.00%** (95% Wilson Confidence Interval: `[46.23%, 65.33%]`, Standard Error: `0.0496`).
* **Balanced Accuracy:** **56.00%**.
* **Macro-Precision:** **64.20%**.
* **Macro-Recall:** **56.00%**.
* **Macro-F1 Score:** **48.57%**.
* **ROC-AUC (Continuous $s_{TI}$ Ranking):** **0.7314**.

### 3.3 Per-Class Performance and Confusion Matrix
| Class | Support | Precision | Recall | F1-Score | True Positive | False Positive | False Negative | True Negative |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pristine** | 50 | 53.41% | 94.00% | 68.12% | 47 | 41 | 3 | 9 |
| **Falsified** | 50 | 75.00% | 18.00% | 29.03% | 9 | 3 | 41 | 47 |

**Confusion Matrix (Raw Counts):**
```
True \ Pred      pristine    falsified
pristine               47            3
falsified              41            9
```

---

## 4. Score Direction and Threshold Provenance

### 4.1 Independent Direction Definition
The mathematical mapping between the continuous similarity score $s_{TI} = \cos(\mathbf{e}_T, \mathbf{e}_I)$ and the binary decision is defined *a priori*:
$$\hat{y} = \begin{cases} \text{pristine (0)}, & \text{if } s_{TI} \ge \tau_{\text{cm}} \\ \text{falsified (1)}, & \text{if } s_{TI} < \tau_{\text{cm}} \end{cases}$$
The score direction is mathematically sound and aligned with OpenCLIP representation space: semantically aligned text-image pairs exhibit higher cosine similarity than mismatched adversarial pairs.

### 4.2 Independent Threshold Origin & Domain Shift
The initial baseline threshold $\tau_{\text{cm}} = 0.22$ was established *a priori* in `configs/pipeline_config.yaml` based on general image-text verification heuristics, completely independent of the benchmark evaluation data.

Inspection of the empirical score distributions reveals why $\tau_{\text{cm}} = 0.22$ produced asymmetric recall:
* **Pristine Pairs ($N=50$):** Mean $s_{TI} = \mathbf{0.3040}$ (Median: `0.3036`, Min: `0.1838`, Max: `0.3843`).
* **Falsified Pairs ($N=50$):** Mean $s_{TI} = \mathbf{0.2687}$ (Median: `0.2780`, Min: `0.1692`, Max: `0.3619`).

Because the NewsCLIPpings dataset generates adversarial cheapfakes using nearest-neighbor semantic retrievers (e.g., CLIP text-to-image and SBERT entity matches), the falsified pairs exhibit relatively high semantic similarity ($\mu = 0.2687$). At $\tau = 0.22$, 41 out of 50 falsified pairs exceeded the threshold and were falsely classified as pristine.

---

## 5. Threshold Sensitivity Analysis

To investigate the decision boundary without tuning on untouched test data, a threshold sensitivity sweep was conducted on the 100 validation samples across candidate cutoffs $\tau \in [0.18, 0.32]$:

| Threshold $\tau$ | Accuracy | Balanced Accuracy | Macro-F1 | Pristine Recall | Falsified Recall | Qualitative State |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.18** | 51.00% | 51.00% | 35.52% | 100.00% | 2.00% | Degenerate pristine classifier |
| **0.20** | 52.00% | 52.00% | 39.12% | 98.00% | 6.00% | High false-negative rate |
| **0.22 (Baseline)** | **56.00%** | **56.00%** | **48.57%** | **94.00%** | **18.00%** | *A priori* uncalibrated baseline |
| **0.24** | 60.00% | 60.00% | 54.77% | 94.00% | 26.00% | Improving falsified sensitivity |
| **0.26** | 67.00% | 67.00% | 64.41% | 94.00% | 40.00% | Significant gain in falsified recall |
| **0.28** | **67.00%** | **67.00%** | **66.60%** | **78.00%** | **56.00%** | **Optimal F1 & Balanced Accuracy** |
| **0.29** | **66.00%** | **66.00%** | **65.99%** | **68.00%** | **64.00%** | **Most balanced per-class recall** |
| **0.30** | 65.00% | 65.00% | 64.83% | 58.00% | 72.00% | Falsified-biased classifier |
| **0.32** | 57.00% | 57.00% | 52.43% | 26.00% | 88.00% | Degenerate falsified classifier |

> [!NOTE]
> **Methodological Clarification:** This sensitivity analysis is an empirical hyperparameter calibration of the visual-semantic comparator on validation data. It is **not** an ablation of the multi-stage misinformation verification pipeline.

---

## 6. Pipeline Ablation Applicability Audit

The research specification outlines four core ablations for the framework:
1. `no_decomp`: Claim decomposition disabled (claim evaluated as a single monolith).
2. `no_rel_filter`: Evidence relevance filtering disabled.
3. `text_only`: Cross-modal image features disabled.
4. `forced_binary`: Three-state epistemic verdicts collapsed into forced binary decisions.

### 6.1 Audit on NewsCLIPpings Task A
| Ablation Candidate | Applicable to Task A? | Scientific Justification |
| :--- | :---: | :--- |
| **`no_decomp`** | **NO** | Task A evaluates single image-caption pairs without multi-proposition text decomposition. |
| **`no_rel_filter`** | **NO** | Task A does not perform external document retrieval; no evidence retrieval filters are engaged. |
| **`text_only`** | **NO** | Cheapfakes consist of authentic news text paired with out-of-context images. Text-only analysis carries zero falsification signal on this task. |
| **`forced_binary`** | **NO** | Task A is natively a binary classification task (`pristine` vs `falsified`); no 3-state epistemic mapping exists to collapse. |

### 6.2 Remaining Ablations Reserved for Factify 2
All four full-pipeline ablations are reserved for **Factify 2 (Task B & Task C)** or other multi-claim, evidence-grounded benchmarks where claim decomposition, multi-document retrieval, NLI premise filtering, and three-state epistemic aggregation are actively deployed.

---

## 7. Scaling Evaluation: 200 Genuine Samples

To evaluate sample-size stability under deterministic stratified sampling, Task A was executed on $N=200$ validation samples using identical configuration parameters.

### 7.1 Execution Command
```bash
python evaluate_benchmark.py \
  --dataset newsclippings \
  --task task_a \
  --split val \
  --limit 200 \
  --seed 42 \
  --threshold-cm 0.22 \
  --output-dir outputs/benchmark_results/newsclippings_val_200
```

### 7.2 Comparative Performance Breakdown ($N=100$ vs $N=200$)
| Evaluation Metric | 100-Sample Validation Run | 200-Sample Validation Run | Absolute Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Total Requested / Evaluated** | 100 / 100 | 200 / 200 | +100 |
| **Pristine : Falsified Support** | 50 : 50 (50.0%) | 100 : 100 (50.0%) | Exact 1:1 balance |
| **Coverage Rate / Errors** | 100.0% / 0 | 100.0% / 0 | 0.0% |
| **Total Latency (Mean/Sample)** | 0.99 s (0.010 s) | 0.89 s (0.004 s) | -0.10 s |
| **Overall Accuracy** | **56.00%** | **55.50%** | **-0.50%** |
| **95% Wilson Confidence Interval** | `[46.23%, 65.33%]` | `[48.57%, 62.22%]` | Narrower CI bounds |
| **Standard Error ($SE$)** | 0.0496 | 0.0351 | -0.0145 (reduced error) |
| **Balanced Accuracy** | 56.00% | 55.50% | -0.50% |
| **Macro-Precision** | 64.20% | 65.99% | +1.79% |
| **Macro-Recall** | 56.00% | 55.50% | -0.50% |
| **Macro-F1 Score** | 48.57% | 46.77% | -1.80% |
| **ROC-AUC (Empirical Binary)** | **0.7314** | **0.7286** | **-0.0028** |

### 7.3 Confusion Matrix Comparison
* **100 Samples:** Pristine: 47 TP, 3 FN; Falsified: 9 TP, 41 FN.
* **200 Samples:** Pristine: 96 TP, 4 FN; Falsified: 15 TP, 85 FN.

### 7.4 Threshold Sweep at $N=200$
A parallel threshold sensitivity sweep on the 200 validation samples confirmed the stability of the calibrated threshold:
* At $\tau = 0.22$ (Baseline): Accuracy = 55.50%, Macro-F1 = 46.77%.
* At $\tau = 0.28$: Accuracy = **67.50%**, Macro-F1 = **66.98%** (Pristine Rec: 80%, Falsified Rec: 55%).
* At $\tau = 0.29$: Accuracy = **67.50%**, Macro-F1 = **67.46%** (Pristine Rec: 71%, Falsified Rec: 64%).

The optimal operating point ($\tau^* = 0.28 - 0.29$) is identical across both sample sizes, confirming that decision boundary shift—not sample variance—governs baseline accuracy.

---

## 8. Factify 2 Access Blocker Documentation

* **Access Gate Status:** Factify 2 data is hosted on CodaLab Competitions (ID: 8275) and requires user credential sign-in and affirmative completion of official registration Google Forms.
* **Status:** Because no local authorized files exist in `data/raw/factify2/`, Factify 2 was not executed in this phase.
* **Resolution Path:** As documented in `phase_2c_data_acquisition_report.md`, the user may manually download `val.json` from CodaLab and place it in `data/raw/factify2/val.json`. The existing Factify 2 adapter is implemented and verified to parse and run it immediately upon arrival.

---

## 9. Error Isolation, Runtime Latency, and Limitations

1. **Zero Runtime Errors:** Across both the 100-sample and 200-sample runs, zero exceptions occurred (`error_rate = 0.0%`).
2. **Sub-Second Throughput:** Leveraging precomputed OpenCLIP embeddings, full 200-sample evaluation completed in **0.89 seconds** on CPU ($< 5\text{ ms}$ per sample), eliminating the computational bottleneck of raw image rasterization.
3. **Current Limitations:**
   - Single-task evaluation: Only Task A (cheapfakes) is empirically evaluated on genuine data to date.
   - Validation-set boundary: The 67.50% calibrated performance reflects validation tuning and must be verified on an untouched test split in final reporting.

---

## 10. Concrete Next Actions

1. **Document Empirical Results:** Transfer the validated Task A metrics, Wilson confidence intervals, and ROC-AUC figures into the MSc dissertation results chapter.
2. **Execute Untouched Test-Set Evaluation:** Run Task A on the untouched `data/raw/newsclippings/merged_balanced/test.json` split using both the baseline $\tau = 0.22$ and calibrated $\tau = 0.28$ to record definitive test-set performance.
3. **Resolve Factify 2 Data (If Desired):** If multi-proposition multimodal fact-checking results are required, obtain the CodaLab `val.json` to execute Task B and Task C benchmarks. If external data remains unavailable, formalize the NewsCLIPpings evaluation as the primary multimodal empirical benchmark.
