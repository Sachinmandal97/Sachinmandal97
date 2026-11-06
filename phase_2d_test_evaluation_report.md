# Phase 2D — Untouched NewsCLIPpings Test Evaluation Report

**Project Title:** Explainable Multimodal Vision-Language Framework for AI-Generated Fake News and Misinformation Detection  
**Module:** MSc Data Science Research Project  
**Date:** October 13, 2026  
**Status:** Completed (Full Untouched NewsCLIPpings Test Split Evaluated at Fixed Thresholds $\tau=0.22$ and $\tau=0.28$)  
**Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission`  
**Protected Baseline Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-verifier` (Untouched)  

---

## 1. Executive Summary

This report delivers the final empirical test-set evaluation for Task A (out-of-context multimodal cheapfake detection) on the official, untouched **NewsCLIPpings test split** (Luo et al., Findings of EMNLP 2021). 

Following the validation-set calibration conducted in Phase 2C, two pre-specified decision thresholds were evaluated on the test set **without any test-set tuning or parameter optimization**:
1. **$\tau = 0.22$**: The *a priori*, uncalibrated baseline heuristic inherited from the original application configuration.
2. **$\tau = 0.28$**: The optimal decision threshold independently selected based on validation-set balanced accuracy and macro-F1.

### Key Empirical Findings
* **Definitive Test-Set Volume:** Evaluated the complete official test split consisting of **7,264 genuine news samples** (3,632 pristine pairs, 3,632 falsified pairs; exact 1:1 balance). A 200-sample test subset (`seed=42`) was also evaluated to enable direct sample-size comparisons with Phase 2C validation runs.
* **Generalization of Calibrated Threshold:** On the full 7,264-sample test split, the validation-calibrated threshold ($\tau = 0.28$) improved overall accuracy from **54.68%** to **63.53%** (+8.85 percentage points) and Macro-F1 from **44.15%** to **62.75%** (+18.60 percentage points) compared to the uncalibrated $\tau = 0.22$ baseline.
* **Empirical Discriminative Separability (ROC-AUC):** The continuous OpenCLIP ViT-B/32 cosine similarity score ($s_{TI}$) achieved a full-test ROC-AUC of **0.7197**, closely mirroring the validation-set ROC-AUC of **0.7286**.
* **Adversarial Generator Disparity:** At $\tau = 0.28$, the framework achieved **81.44% accuracy** on `semantics_clip_text_text` mismatches and **67.79% accuracy** on `scene_resnet_place` mismatches, but struggled on `semantics_clip_text_image` (43.83%), validating published findings that adversaries optimizing directly against CLIP representation space create strong evasive perturbations.
* **Sub-Second Throughput:** Leveraging official precomputed 512-d CLIP vectors, inference across all 7,264 test records executed in **1.33 seconds** on CPU ($< 0.2\text{ ms}$ per sample), with zero failures (`error_rate = 0.0%`).

---

## 2. Test Data Provenance, Verification, and Cryptographic Checksums

In accordance with Phase 2D Step 1, all required official test split assets were acquired from the official public HuggingFace repository (`g-luo/news-clippings`, Apache-2.0 license) without downloading the ~100 GB VisualNews raster image archive.

### 2.1 File Inventory and Checksums
| File Description | Relative Workspace Path | File Size | SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| **NewsCLIPpings Test Split** | `data/raw/newsclippings/merged_balanced/test.json` | 788,016 B | `616E06AF39FD8AFA88D1920E455E4D6CA69E013C9B2FA4DADDA593FB0E813AEA` |
| **NewsCLIPpings Metadata Test** | `data/raw/newsclippings/metadata/test.json` | 37,369,991 B | `1B2BC8438135AC8C5E4B78289F6FEC6D6956D92F30850945C057F3733A998BB5` |
| **CLIP Text Embeddings Test** | `data/raw/newsclippings/embeddings/clip_text_embeddings/clip_text_embeddings_test.pkl` | 86,368,983 B | `7E4C96F98952B57D3D2EBD83476DB3F1BB5202AC54DEDCDB94E0A04DEFCFA438` |
| **CLIP Image Embeddings Test** | `data/raw/newsclippings/embeddings/clip_image_embeddings/clip_image_embeddings_test.pkl` | 105,640,860 B | `408AA7AD2F7319A18B51412985D65535D1945F4155742AA4889536CB90514570` |

### 2.2 Test Corpus Integrity & Balancing
* **Total Annotations:** 7,264 records.
* **Pristine Pairs (Label 0):** 3,632 (50.00%).
* **Falsified Pairs (Label 1):** 3,632 (50.00%).
* **Precomputed Embedding Coverage:** 7,264 / 7,264 text vectors (100.0%) and 7,264 / 7,264 image vectors (100.0%). Zero missing embeddings.
* **Vector Specifications:** 512-dimensional `float32` vectors, unit $L_2$-normalized ($||\mathbf{e}||_2 = 1.000000$).
* **Generator Distribution:** Exactly 1,816 samples (25.0%) across each of the four official mismatch generation algorithms (908 pristine, 908 falsified per generator).

> [!WARNING]
> **Prevalence Notice:** The 50/50 balance in NewsCLIPpings is an artificial benchmark design to facilitate unskewed evaluation of sensitivity and specificity. It **must not** be interpreted as reflecting real-world misinformation prevalence in open news streams, where falsified media is typically a low-prevalence anomaly ($< 5\%$).

---

## 3. Official Full Test Split Evaluation ($N = 7,264$)

Evaluation was executed across all 7,264 untouched test records at both fixed thresholds. Neither threshold was calibrated, adjusted, or re-tuned on this test set.

### 3.1 Execution Commands
```bash
# Baseline Uncalibrated Heuristic (tau = 0.22)
python evaluate_benchmark.py \
  --dataset newsclippings \
  --task task_a \
  --split test \
  --threshold-cm 0.22 \
  --output-dir outputs/benchmark_results/newsclippings_test_full_tau22

# Validation-Calibrated Decision Threshold (tau = 0.28)
python evaluate_benchmark.py \
  --dataset newsclippings \
  --task task_a \
  --split test \
  --threshold-cm 0.28 \
  --output-dir outputs/benchmark_results/newsclippings_test_full_tau28
```

### 3.2 Full Test Quantitative Results ($N = 7,264$)
| Metric | Baseline Threshold ($\tau = 0.22$) | Validation-Calibrated ($\tau = 0.28$) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Total Evaluated / Failures** | 7,264 / 0 (100% coverage) | 7,264 / 0 (100% coverage) | 0 |
| **Total Latency (Mean/Sample)** | 1.42 s (0.0002 s/sample) | 1.33 s (0.0002 s/sample) | -0.09 s |
| **Overall Accuracy** | **54.68%** | **63.53%** | **+8.85%** |
| **95% Wilson Confidence Interval** | `[53.53%, 55.82%]` | `[62.42%, 64.63%]` | Disjoint CIs ($p < 0.001$) |
| **Standard Error ($SE$)** | 0.0058 | 0.0056 | -0.0002 |
| **Balanced Accuracy** | 54.68% | 63.53% | +8.85% |
| **Macro-Precision** | 69.04% | 64.78% | -4.26% |
| **Macro-Recall** | 54.68% | 63.53% | +8.85% |
| **Macro-F1 Score** | **44.15%** | **62.75%** | **+18.60%** |
| **Weighted-F1 Score** | 44.15% | 62.75% | +18.60% |
| **ROC-AUC (Empirical Binary)** | **0.7197** | **0.7197** | Threshold-invariant |

### 3.3 Full Test Confusion Matrices
* **Baseline ($\tau = 0.22$):**
  ```
  True \ Pred        pristine    falsified
  pristine               3563           69    (Recall: 98.10%, Precision: 52.51%, F1: 68.40%)
  falsified              3223          409    (Recall: 11.26%, Precision: 85.56%, F1: 19.90%)
  ```
* **Validation-Calibrated ($\tau = 0.28$):**
  ```
  True \ Pred        pristine    falsified
  pristine               2835          797    (Recall: 78.06%, Precision: 60.49%, F1: 68.16%)
  falsified              1852         1780    (Recall: 49.01%, Precision: 69.07%, F1: 57.34%)
  ```

---

## 4. Test Performance Breakdown by Adversarial Generator

The 7,264 test records are partitioned into four equal generator subsets ($N=1,816$ each). Disaggregating performance by generator reveals where cross-modal consistency succeeds and where it encounters fundamental limitations:

| Generator Name | Alignment Mechanism | Accuracy ($\tau = 0.22$) | Accuracy ($\tau = 0.28$) | Pristine Recall ($\tau=0.28$) | Falsified Recall ($\tau=0.28$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Gen 1: `semantics_clip_text_text`** | Text-to-text semantic nearest neighbor | 55.12% | **81.44%** | 89.10% | **73.79%** |
| **Gen 2: `scene_resnet_place`** | Places365 scene context match | 60.35% | **67.79%** | 72.03% | **63.55%** |
| **Gen 3: `person_sbert_text_text`** | Person entity overlap + SBERT similarity | 54.52% | **61.07%** | 82.27% | **39.87%** |
| **Gen 0: `semantics_clip_text_image`** | Unconstrained CLIP text-to-image NN | 48.73% | **43.83%** | 68.83% | **18.83%** |

### Methodological Interpretation:
1. **High Efficacy on Semantic and Scene Shifts:** For text-to-text mismatches (Gen 1) and scene substitutions (Gen 2), calibrating to $\tau=0.28$ achieves strong classification accuracy (81.44% and 67.79%), detecting semantic divergence between image content and caption claims.
2. **The Adversarial Vulnerability of Shared Representation Spaces:** In Gen 0, the benchmark generator selected images by directly maximizing CLIP cosine similarity against the caption. Because the adversary optimizes the exact objective used by the zero-shot detector, the falsified pairs exhibit artificially elevated similarity ($s_{TI} \ge 0.28$), depressing falsified recall to 18.83%. This directly replicates the findings of Luo et al. (EMNLP 2021) and demonstrates the necessity of multimodal entity-grounded reasoning beyond pure embedding similarity.

---

## 5. Controlled 200-Sample Test Subset Evaluation

To ensure direct methodological comparability with the 200-sample validation run reported in Phase 2C, a 200-sample test subset was drawn using identical deterministic stratified sampling (`seed=42`, `balanced=True`):

### 5.1 Execution Commands
```bash
python evaluate_benchmark.py --dataset newsclippings --task task_a --split test --limit 200 --seed 42 --threshold-cm 0.22 --output-dir outputs/benchmark_results/newsclippings_test_200_tau22
python evaluate_benchmark.py --dataset newsclippings --task task_a --split test --limit 200 --seed 42 --threshold-cm 0.28 --output-dir outputs/benchmark_results/newsclippings_test_200_tau28
```

### 5.2 Comparative Analysis (Validation vs. Test Subset vs. Full Test)
| Evaluation Split / Sample Set | Threshold $\tau$ | Accuracy | Macro-F1 | Pristine Recall | Falsified Recall | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Subset ($N=200$, Phase 2C)** | 0.22 | 55.50% | 46.77% | 96.00% | 15.00% | 0.7286 |
| **Validation Subset ($N=200$, Phase 2C)** | 0.28 | **67.50%** | **66.98%** | 80.00% | 55.00% | 0.7286 |
| **Test Subset ($N=200$, Phase 2D)** | 0.22 | 52.00% | 41.11% | 95.00% | 9.00% | 0.6585 |
| **Test Subset ($N=200$, Phase 2D)** | 0.28 | **56.50%** | **55.43%** | 72.00% | 41.00% | 0.6585 |
| **Full Test Split ($N=7,264$, Phase 2D)** | 0.22 | 54.68% | 44.15% | 98.10% | 11.26% | 0.7197 |
| **Full Test Split ($N=7,264$, Phase 2D)** | 0.28 | **63.53%** | **62.75%** | 78.06% | 49.01% | **0.7197** |

### Insights on Sample Scaling:
* **Subset Sampling Variance:** The 200-sample test subset exhibited slightly lower ROC-AUC (0.6585) due to sample draw variability within smaller quotas ($N=25$ per generator).
* **Law of Large Numbers on Full Split:** On the complete 7,264-sample test split, the full ROC-AUC converged to **0.7197**, demonstrating that the framework's discriminative capability generalizes consistently from validation (0.7286) to test (0.7197) with a divergence of less than 0.009.

---

## 6. Methodological Distinctions and Scientific Rigor

To maintain scientific integrity in the MSc dissertation, four distinct evaluation modes implemented across this project must be strictly delineated:

```
                      METHODOLOGICAL EVALUATION HIERARCHY
                      
1. Synthetic Fixtures (P0 / Pilot 2B):
   └── Unit test stubs (5 samples) verifying code execution and pipeline schema.
       NOT empirical scientific evidence.

2. Live-Web Smoke Tests (P0.1):
   └── Qualitative end-to-end demonstrations with live HTTP/RSS retrieval.
       Tests operational integrity, NOT benchmark metrics.

3. Validation Tuning (Phase 2C):
   └── Exploratory analysis on val.json to diagnose boundary shifts and calibrate tau*=0.28.
       Used strictly for hyperparameter selection.

4. Untouched Benchmark Test (Phase 2D - THIS REPORT):
   └── Final, frozen evaluation on test.json across 7,264 genuine news items.
       Primary empirical claim for dissertation Task A results.
```

### Clarification on What This Evaluates
> [!IMPORTANT]
> This evaluation assesses **cross-modal visual-semantic consistency (cheapfake detection)** using OpenCLIP embeddings. It **does not** assess whether an image was AI-generated (e.g. by Midjourney or Stable Diffusion). Cheapfakes consist of authentic, real-world news photographs repurposed with fabricated or mismatched captions.

---

## 7. Artifact & Reproducibility Audit Trail

All prediction logs, confusion matrices, and metric summaries are preserved in JSONL and JSON format in the submission workspace:
* **Full Test ($\tau = 0.22$) Predictions:** [`outputs/benchmark_results/newsclippings_test_full_tau22/newsclippings_task_a_predictions.jsonl`](file:///C:/Users/94in/.gemini/antigravity/scratch/multimodal-misinfo-submission/outputs/benchmark_results/newsclippings_test_full_tau22/newsclippings_task_a_predictions.jsonl)
* **Full Test ($\tau = 0.22$) Metrics:** [`outputs/benchmark_results/newsclippings_test_full_tau22/newsclippings_task_a_metrics.json`](file:///C:/Users/94in/.gemini/antigravity/scratch/multimodal-misinfo-submission/outputs/benchmark_results/newsclippings_test_full_tau22/newsclippings_task_a_metrics.json)
* **Full Test ($\tau = 0.28$) Predictions:** [`outputs/benchmark_results/newsclippings_test_full_tau28/newsclippings_task_a_predictions.jsonl`](file:///C:/Users/94in/.gemini/antigravity/scratch/multimodal-misinfo-submission/outputs/benchmark_results/newsclippings_test_full_tau28/newsclippings_task_a_predictions.jsonl)
* **Full Test ($\tau = 0.28$) Metrics:** [`outputs/benchmark_results/newsclippings_test_full_tau28/newsclippings_task_a_metrics.json`](file:///C:/Users/94in/.gemini/antigravity/scratch/multimodal-misinfo-submission/outputs/benchmark_results/newsclippings_test_full_tau28/newsclippings_task_a_metrics.json)
* **200-Sample Test ($\tau = 0.22$) Metrics:** [`outputs/benchmark_results/newsclippings_test_200_tau22/newsclippings_task_a_metrics.json`](file:///C:/Users/94in/.gemini/antigravity/scratch/multimodal-misinfo-submission/outputs/benchmark_results/newsclippings_test_200_tau22/newsclippings_task_a_metrics.json)
* **200-Sample Test ($\tau = 0.28$) Metrics:** [`outputs/benchmark_results/newsclippings_test_200_tau28/newsclippings_task_a_metrics.json`](file:///C:/Users/94in/.gemini/antigravity/scratch/multimodal-misinfo-submission/outputs/benchmark_results/newsclippings_test_200_tau28/newsclippings_task_a_metrics.json)
