# VERITAS-MM 2.0: Official Consolidated Empirical Evaluation Results

**Academic Program**: Master of Science in Data Science  
**Institution**: Somaiya Vidyavihar University  
**Academic Year**: 2026–2027  
**Evaluation Scope**: Official Certified Results for Academic Project Documentation  
**Primary Repository**: `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission`  
**Protected Baseline Repository**: `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-verifier` (Untouched, Timestamp: `7/10/2026 8:20:55 PM`)

---

## 1. Executive Metrics Overview

This document records the exact, verified empirical evaluation metrics for the **VERITAS-MM 2.0** multimodal misinformation verification framework across all evaluated benchmarks and software suites.

```
+-------------------------------------------------------------------------------------------------------+
| VERITAS-MM 2.0 EVALUATION SUMMARY DASHBOARD                                                           |
+-------------------------------------------------------------------------------------------------------+
| 1. NEWSCLIPPINGS TASK A (OUT-OF-CONTEXT CHEAPFAKES, N=7,264 TEST RECORDS)                             |
|    - Full Benchmark Baseline (tau=0.28, 100% Coverage):                                              |
|      Nominal Accuracy: 63.53%  |  Balanced Accuracy: 63.53%  |  Macro-F1: 62.75%  |  ROC-AUC: 0.7197   |
|      95% Wilson CI: [62.42%, 64.63%]  |  McNemar Test: chi^2 = 196.3621, p = 1.30e-44                |
|    - Selective Classification Regime (tau=0.28, Delta*=0.07, 10.85% Coverage, N_acc=788):            |
|      Accepted-Subset Accuracy: 87.44%  |  Balanced Accuracy: 87.10%  |  Macro-F1: 87.24%              |
|      Selective Error Risk: 12.56% (65.6% Relative Risk Reduction vs. 100% Coverage)                   |
|      95% Wilson CI: [84.94%, 89.57%] (Empirical point estimate; not a guarantee strictly > 85%)     |
|      Pristine Class: Prec: 87.33%, Rec: 90.34%, F1: 88.81% (N=435, TP=393, FP=57)                    |
|      Falsified Class: Prec: 87.57%, Rec: 83.85%, F1: 85.67% (N=353, TP=296, FP=42)                  |
|      Deferred Review: 6,476 records (89.15%) deferred for subsequent human review                    |
|                                                                                                       |
| 2. VLDBENCH MULTIMODAL RHETORICAL FRAMING (31,339 RECORDS TOTAL, 4,285 CLUSTERS)                      |
|    - Selected Model: Option 3 Gated Multimodal Fusion Adapter (1,582,594 Parameters)                  |
|      Protocol A Test (N=1,094): Macro-F1: 64.54% (95% CI: [61.73%, 67.20%])                           |
|      Balanced Accuracy: 64.80%  |  Test Accuracy: 64.72%  |  ROC-AUC: 0.7086                         |
|      Protocol B Strict Temporal (Post-July 2024, 129 Groups Purged): Macro-F1: 62.43%                 |
|      Publisher-Held-Out Evaluation (Unseen Newsrooms): Macro-F1: 61.42%                               |
|      Learned Modality Salience: Text Channel w_T = 38.19%  |  Visual Channel w_I = 61.81%             |
|                                                                                                       |
| 3. SOFTWARE INTEGRITY & REGRESSION SUITES                                                             |
|    - Master Regression Runner (run_tests.py): 122 / 122 Tests PASSED (100% Pass Rate)                 |
|    - Rhetoric Integration Suite (tests/test_rhetoric_integration.py): 8 / 8 Tests PASSED              |
|    - SQLite Production Intelligence Store (data/news_intelligence.db): 325 Verified Records Intact    |
+-------------------------------------------------------------------------------------------------------+
```

---

## 2. Benchmark 1: NewsCLIPpings Task A (Cross-Modal Mismatch Detection)

### 2.1 Dataset Provenance & Split Sizes
- **Source**: Luo et al., "NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media," Findings of EMNLP 2021.
- **License**: Apache License 2.0.
- **Splits**:
  - Full Validation Split (`data/raw/newsclippings/merged_balanced/val.json`): $N = 7,024$ records (3,512 pristine, 3,512 falsified).
  - Full Test Split (`data/raw/newsclippings/merged_balanced/test.json`): $N = 7,264$ records (3,632 pristine, 3,632 falsified).
- **Validation Sample Size Resolution**:
  - Phase 2C Exploratory Diagnostic: $N = 200$ samples (`--limit 200 --seed 42`, `outputs/benchmark_results/newsclippings_val_200/newsclippings_task_a_metrics.json`, `total_samples: 200`, `total_available_in_split: 7024`). Used under unbatched CPU image decoding limits to select baseline decision boundary $\tau = 0.28$.
  - Selective Classification Risk-Coverage Optimization: $N = 7,024$ records (`val.json`, `scratch/selective_classification_audit.json`). Used with precomputed OpenCLIP embeddings to locate uncertainty margin $\Delta^* = 0.07$.
  - Zero test records were accessed during either parameter tuning pass.

### 2.2 Baseline vs. Calibrated Threshold Performance ($N = 7,264$ Test Records)

| Metric | Heuristic Baseline ($\tau = 0.22$) | Calibrated Operating Point ($\tau = 0.28$) | Absolute Delta | Relative Gain |
|:---|:---:|:---:|:---:|:---:|
| **Nominal Accuracy** | 54.68% | **63.53%** | +8.85% | +16.19% |
| **95% Wilson Confidence Interval** | [53.53%, 55.82%] | **[62.42%, 64.63%]** | — | — |
| **Balanced Accuracy** | 54.68% | **63.53%** | +8.85% | +16.19% |
| **Macro-Precision** | 69.04% | **64.78%** | -4.26% | -6.17% |
| **Macro-Recall** | 54.68% | **63.53%** | +8.85% | +16.19% |
| **Macro-F1 Score** | 44.15% | **62.75%** | **+18.60%** | **+42.13%** |
| **Pristine Class Specificity** | 98.10% (3,563 / 3,632) | **78.06%** (2,835 / 3,632) | -20.04% | Re-balanced |
| **Falsified Class Sensitivity** | 11.26% (409 / 3,632) | **49.01%** (1,780 / 3,632) | **+37.75%** | **+335.2%** |
| **ROC-AUC (Rank Order)** | 0.7197 | **0.7197** | 0.00% | Invariant |
| **Total Evaluation Latency** | 1.42 s (10.3 ms/sample) | **1.33 s** (9.7 ms/sample) | -0.09 s | Optimized |

### 2.3 Statistical Hypothesis Testing (Edwards Continuity-Corrected McNemar Test)
- Discordant Pairs on Identical 7,264 Test Records:
  - $b = 728$ ($\tau = 0.22$ correct, $\tau = 0.28$ incorrect).
  - $c = 1,371$ ($\tau = 0.28$ correct, $\tau = 0.22$ incorrect).
  - Discordant Total: $b + c = 2,099$.
- Test Statistic:
  $$\chi^2 = \frac{(|728 - 1,371| - 1)^2}{2,099} = \frac{642^2}{2,099} = \mathbf{196.3621} \quad (\text{df} = 1)$$
- Asymptotic $p$-value: $p = \mathbf{1.30 \times 10^{-44}}$.
- Exact Binomial Two-Sided $p$-value: $p = \mathbf{2.65 \times 10^{-45}}$.
- *Conclusion*: Statistically significant beyond any reasonable doubt ($p \ll 0.001$).

### 2.4 Adversarial Generator Robustness Breakdown ($N = 1,816$ per Generator Strategy)

| Generator Category | Adversarial Perturbation Technique | Baseline Acc ($\tau=0.22$) | Calibrated Acc ($\tau=0.28$) | Pristine Rec ($\tau=0.28$) | Falsified Rec ($\tau=0.28$) |
|:---|:---|:---:|:---:|:---:|:---:|
| **Gen 1: `semantics_clip_text_text`** | Text nearest neighbor in CLIP space | 55.12% | **81.44%** | 89.10% | **73.79%** |
| **Gen 2: `scene_resnet_place`** | Scene feature substitution via Places365 | 60.35% | **67.79%** | 72.03% | **63.55%** |
| **Gen 3: `person_sbert_text_text`** | Person entity matching via SBERT | 54.52% | **61.07%** | 82.27% | **39.87%** |
| **Gen 0: `semantics_clip_text_image`** | Unconstrained CLIP text-image NN | 48.73% | **43.83%** | 64.87% | **22.80%** |

### 2.5 Full Selective Classification Sweep (Validation $N=7,024$ & Test $N=7,264$)

| Margin $\Delta$ | Val Cov $\Phi$ | Val Acc | Val Risk $\mathcal{R}$ | Test Cov $\Phi$ | Accepted Test $N$ | Accepted Test Acc | Accepted Test BalAcc | Accepted Test Macro-F1 | 95% Wilson CI (Test) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.00** | 100.0% | 63.51% | 36.49% | **100.0%** | **7,264** | **63.53%** | 63.53% | 62.75% | [62.42%, 64.63%] |
| 0.01 | 83.61% | 65.95% | 34.05% | 82.03% | 5,959 | 66.91% | 66.48% | 65.76% | [65.70%, 68.09%] |
| 0.02 | 66.20% | 69.40% | 30.60% | 66.42% | 4,825 | 70.24% | 69.29% | 68.95% | [68.93%, 71.51%] |
| 0.03 | 50.17% | 73.70% | 26.30% | 50.73% | 3,685 | 74.22% | 72.97% | 73.05% | [72.78%, 75.61%] |
| 0.04 | 36.77% | 77.82% | 22.18% | 36.78% | 2,672 | 78.52% | 77.07% | 77.48% | [76.92%, 80.03%] |
| 0.05 | 25.97% | 81.14% | 18.86% | 26.10% | 1,896 | 82.23% | 80.75% | 81.32% | [80.44%, 83.88%] |
| 0.06 | 17.75% | 84.28% | 15.72% | 16.97% | 1,233 | 85.24% | 84.32% | 84.69% | [83.15%, 87.11%] |
| **0.07\*** | **11.30%** | **86.40%** | **13.60%** | **10.85%** | **788** | **87.44%** | **87.10%** | **87.24%** | **[84.94%, 89.57%]** |
| 0.08 | 6.98% | 87.35% | 12.65% | 6.95% | 505 | 91.09% | 91.02% | 91.06% | [88.28%, 93.27%] |
| 0.09 | 4.02% | 90.43% | 9.57% | 4.35% | 316 | 93.04% | 93.04% | 93.04% | [89.68%, 95.36%] |
| 0.10 | 2.22% | 91.03% | 8.97% | 2.62% | 190 | 92.11% | 92.18% | 92.04% | [87.38%, 95.16%] |

*Operating Point $\Delta^* = 0.07$ Verification*:
- Total Accepted on Test: $N_{\text{acc}} = 788$ ($689$ correct, $99$ incorrect; Accuracy: **87.44%**, Selective Risk: **12.56%**).
- Total Deferred for Human Review: $N_{\text{rej}} = 6,476$ ($89.15\%$ rejection rate).
- Class 0 (Pristine): Support = 435, TP = 393, FP = 57, Precision = **87.33%**, Recall = **90.34%**, F1 = **88.81%**.
- Class 1 (Falsified): Support = 353, TP = 296, FP = 42, Precision = **87.57%**, Recall = **83.85%**, F1 = **85.67%**.

---

## 3. Benchmark 2: VLDBench Multimodal Rhetorical Manipulation

### 3.1 Architectural Comparison on Protocol A Test ($N = 1,094$)

| Model Family / Architecture | Trainable Parameters | Val Macro-F1 | Val BalAcc | Test Accuracy (95% CI) | Test Balanced Acc (95% CI) | Test Macro-F1 (95% CI) | Test ROC-AUC (95% CI) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Publisher-Only Baseline** | 0 | 0.4390 | 0.5244 | 0.5832 [0.5538, 0.6147] | 0.5305 [0.5161, 0.5460] | 0.4436 [0.4176, 0.4718] | 0.5697 [0.5369, 0.6031] |
| **Sanitized Text-Only (DeBERTa-v3)** | 0.20M | 0.5754 | 0.5759 | 0.5576 [0.5284, 0.5868] | 0.5499 [0.5218, 0.5779] | 0.5500 [0.5215, 0.5778] | 0.5581 [0.5249, 0.5908] |
| **Image-Only (OpenCLIP ViT-B/32)** | 0.20M | 0.6620 | 0.6711 | 0.6435 [0.6141, 0.6719] | 0.6483 [0.6200, 0.6763] | 0.6430 [0.6138, 0.6718] | 0.7056 [0.6766, 0.7349] |
| **Option 1: Late Concatenation** | 0.86M | 0.6474 | 0.6565 | 0.6325 [0.6031, 0.6615] | 0.6381 [0.6088, 0.6664] | 0.6322 [0.6022, 0.6611] | 0.7053 [0.6745, 0.7371] |
| **Option 3: Gated Multimodal Fusion\*** | **1.58M** | **0.6606** | **0.6662** | **0.6472** [0.6196, 0.6740] | **0.6480** [0.6204, 0.6745] | **0.6454** [0.6173, 0.6720] | **0.7086** [0.6793, 0.7387] |

\* *Selected Primary Adapter Model for Production Deployment.*

### 3.2 Generalization Across Robustness Evaluation Protocols

| Evaluation Protocol | Evaluation Objective | Records ($N$) | Option 3 Macro-F1 | Option 1 Macro-F1 | Text-Only Macro-F1 |
|:---|:---|:---:|:---:|:---:|:---:|
| **Protocol A (Spatial Group-Aware)** | Bipartite syndication clustering (70/15/15) | 1,094 | **64.54%** | 63.22% | 55.00% |
| **Protocol B Strict (Temporal Purged)** | Temporal shift post-July 1, 2024 (129 groups purged) | 1,225 | **62.43%** | 60.88% | 53.12% |
| **Publisher-Held-Out Evaluation** | Unseen newsrooms (evaluated after headline sanitization) | 980 | **61.42%** | 59.74% | 52.05% |

### 3.3 Learned Modality Gating Statistics ($N = 1,094$ Test Instances)
- Mean Sigmoid Gating Vector: $\mu_{\bar{g}} = \mathbf{0.3819} \pm 0.042$.
- Learned Modality Salience:
  - Textual Framing Channel Weight: $w_T = 38.19\%$.
  - Visual Context Channel Weight: $w_I = 61.81\%$.
- Text-Only Degradation Fallback: When `image = None`, adapter gracefully locks to $\bar{g} = 1.0$ (100% text salience, 0% visual salience) without runtime failure.

---

## 4. Software Regression & Integration Integrity Metrics

| Test Suite / Module | Script Path | Tests Executed | Passed | Failed | Execution Time |
|:---|:---|:---:|:---:|:---:|:---:|
| **Master Regression Runner** | `run_tests.py` | 122 | 122 | 0 | 53.30 s |
| **Rhetoric Integration Suite** | `tests/test_rhetoric_integration.py` | 8 | 8 | 0 | 50.40 s |
| **End-to-End Application Smoke** | `tests/test_final_app_smoke.py` | 7 | 7 | 0 | 12.18 s |
| **Total Automated Quality Tests** | — | **137** | **137** | **0** | **115.88 s** |

- **SQLite Database Integrity**: Verified active at `data/news_intelligence.db` with 325 articles stored, 0 corrupted records.
- **Protected Baseline**: `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-verifier` preserved completely untouched (`7/10/2026 8:20:55 PM`).
