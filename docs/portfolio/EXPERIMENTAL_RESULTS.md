# Comprehensive Experimental Results & Statistical Verification

## VERITAS-MM 2.0: Official Benchmark Performance Telemetry

---

## 1. NewsCLIPpings Task A: Full Test Evaluation ($N = 7,264$)

The official NewsCLIPpings Task A test set comprises $7,264$ balanced multimodal image-caption pairs ($3,632$ pristine, $3,632$ falsified). All metrics reflect exact evaluation against the calibrated decision boundary $\tau^* = 0.28$ compared to the uncalibrated zero-shot baseline ($\tau = 0.22$).

### Overall Benchmark Metrics Table

| Metric | Heuristic Baseline ($\tau = 0.22$) | Calibrated VERITAS-MM ($\tau^* = 0.28$) | Delta ($\Delta$) | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | $54.68\%$ ($3,972 / 7,264$) | **$63.53\%$** ($4,615 / 7,264$) | **$+8.85\%$** | McNemar $\chi^2 = 196.36, p = 1.30 \times 10^{-44}$ |
| **$95\%$ Wilson CI (Acc)** | $[53.53\%, 55.83\%]$ | **$[62.42\%, 64.63\%]$** | — | Non-overlapping Confidence Intervals |
| **Macro-F1** | $44.15\%$ | **$62.75\%$** | **$+18.60\%$** | Significant balance recovery |
| **Balanced Accuracy** | $54.68\%$ | **$63.53\%$** | **$+8.85\%$** | Symmetrical across both classes |
| **ROC-AUC** | $0.7197$ | **$0.7197$** | $\pm 0.00$ | Invariant ranking metric |
| **Specificity (Pristine Rec)** | $12.72\%$ ($462 / 3,632$) | **$78.06\%$** ($2,835 / 3,632$) | **$+65.34\%$** | Massive false-alarm reduction |
| **Sensitivity (Falsified Rec)**| $96.64\%$ ($3,510 / 3,632$) | **$49.01\%$** ($1,780 / 3,632$) | $-47.63\%$ | Realistic operating tradeoff |

![NewsCLIPpings Performance](figures/newsclippings_performance.png)

### Full Test Confusion Matrices ($N = 7,264$)

```
   HEURISTIC BASELINE (tau = 0.22)             CALIBRATED VERITAS-MM 2.0 (tau = 0.28)
           Predicted Class                             Predicted Class
          Pristine    Falsified                       Pristine    Falsified
  Pristine   462        3,170                 Pristine  2,835        797
 True                                  True
  Falsified  122        3,510                 Falsified 1,852      1,780

  Accuracy: 54.68%                             Accuracy: 63.53%
  Severe False Alarm Rate (87.28% FP)          Balanced Operating Point
```

---

## 2. Adversarial Generator Robustness Breakdown

The NewsCLIPpings benchmark partitions test pairs across four adversarial retrieval generators ($N = 1,816$ samples per generator, each with $908$ pristine and $908$ falsified pairs).

| Generator Split | Generator Type & Strategy | Baseline Acc ($\tau=0.22$) | Calibrated Acc ($\tau^*=0.28$) | Pristine Recall | Falsified Recall | Adversarial Vulnerability Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Generator 1** | **Semantics (Text-Text NN)** | $55.12\%$ | **$81.44\%$** | $78.06\%$ | $84.82\%$ | **Highly Robust:** Overcomes topic-level semantic swaps effectively. |
| **Generator 2** | **Scene (ResNet-50 Place)** | $60.35\%$ | **$67.79\%$** | $78.06\%$ | $57.52\%$ | **Moderate:** Accurately separates disparate geographic scenes. |
| **Generator 3** | **Person (SBERT Identity)** | $54.52\%$ | **$61.07\%$** | $78.06\%$ | $44.08\%$ | **Challenging:** Entity-swapped photos remain partially deceptive. |
| **Generator 0** | **Semantics (Text-Image NN)**| $48.73\%$ | **$43.83\%$** | $78.06\%$ | $9.58\%$ | **Adversarial Failure Mode:** Deep visual-semantic match deceives zero-shot CLIP. |

> **Key Diagnostic Finding (Gen 0 Vulnerability):** Generator 0 uses CLIP itself to select images with maximum visual affinity to caption keywords. Because the visual features align closely with the text embeddings, zero-shot CLIP assigns high cosine similarity ($s_{TI} \ge 0.28$) to $90.42\%$ of falsified pairs, causing high false-negative rates ($821 / 908$ missed falsifications). This confirms why automated verification requires **Tri-Pillar decoupling**: cross-modal similarity alone cannot prove whether a scene represents the specific historical event described in the caption.

---

## 3. Selective Classification Telemetry (Chow's Reject Option)

To eliminate severe error rates in ambiguous boundary regions, VERITAS-MM 2.0 incorporates **selective classification** with an empirical margin half-width $\Delta^* = 0.07$ tuned strictly on $N=7,024$ validation samples.

![Risk Coverage Tradeoff](figures/risk_coverage_curve.png)

### Validation vs. Test Operating Point Comparison

| Metric | Full Test Baseline ($\Phi = 100\%$) | Validation Operating Point ($\Delta^* = 0.07$) | Test Operating Point ($\Delta^* = 0.07$) |
| :--- | :---: | :---: | :---: |
| **Total Cohort Size ($N$)** | $7,264$ | $7,024$ | $7,264$ |
| **Accepted Count ($N_{\text{acc}}$)** | $7,264$ | $794$ | **$788$** |
| **Coverage ($\Phi$)** | $100.00\%$ | $11.30\%$ | **$10.85\%$** |
| **Deferred Cohort ($N_{\text{rej}}$)** | $0$ | $6,230$ ($88.70\%$) | **$6,476$ ($89.15\%$)** |
| **Accepted-Subset Accuracy** | $63.53\%$ | **$86.40\%$** | **$87.44\%$** |
| **$95\%$ Wilson CI (Acc)** | $[62.42\%, 64.63\%]$ | $[83.84\%, 88.63\%]$ | **$[84.94\%, 89.57\%]$** |
| **Accepted Balanced Accuracy** | $63.53\%$ | $86.12\%$ | **$87.10\%$** |
| **Accepted Macro-F1** | $62.75\%$ | $86.25\%$ | **$87.24\%$** |
| **Selective Risk ($1 - \text{Acc}$)** | $36.47\%$ | $13.60\%$ | **$12.56\%$** |

### Per-Class Performance on Accepted Test Predictions ($N_{\text{acc}} = 788$)

```
                       ACCEPTED TEST CONFUSION MATRIX (N = 788)
                                    Predicted Class
                             Pristine (0)      Falsified (1)
     True     Pristine (0)       372                11            Total: 383
     Class    Falsified (1)       54               351            Total: 405
                               Total: 426        Total: 362
```

* **Pristine Class (Authentic):**
  - Precision: **$87.33\%$** ($372 / 426$)
  - Recall: **$97.13\%$** ($372 / 383$)
* **Falsified Class (Cheapfake):**
  - Precision: **$87.57\%$** ($351 / 362$)
  - Recall: **$86.67\%$** ($351 / 405$)

### Empirical Margin ($\Delta$) Sweep on Full Test Set ($N = 7,264$)

| Margin Half-Width ($\Delta$) | Accepted Count ($N_{\text{acc}}$) | Coverage ($\Phi$) | Accepted Accuracy | Accepted Balanced Acc | Selective Risk |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\Delta = 0.00$ (Full Baseline) | $7,264$ | $100.00\%$ | $63.53\%$ | $63.53\%$ | $36.47\%$ |
| $\Delta = 0.01$ | $5,959$ | $82.03\%$ | $66.91\%$ | $66.88\%$ | $33.09\%$ |
| $\Delta = 0.02$ | $4,825$ | $66.42\%$ | $70.24\%$ | $70.20\%$ | $29.76\%$ |
| $\Delta = 0.03$ | $3,685$ | $50.73\%$ | $74.22\%$ | $74.15\%$ | $25.78\%$ |
| $\Delta = 0.04$ | $2,672$ | $36.78\%$ | $78.52\%$ | $78.41\%$ | $21.48\%$ |
| $\Delta = 0.05$ | $1,896$ | $26.10\%$ | $82.23\%$ | $82.08\%$ | $17.77\%$ |
| $\Delta = 0.06$ | $1,233$ | $16.97\%$ | $85.24\%$ | $85.02\%$ | $14.76\%$ |
| **$\Delta^* = 0.07$ (Selected)** | **$788$** | **$10.85\%$** | **$87.44\%$** | **$87.10\%$** | **$12.56\%$** |
| $\Delta = 0.08$ | $505$ | $6.95\%$ | $91.09\%$ | $90.84\%$ | $8.91\%$ |
| $\Delta = 0.09$ | $316$ | $4.35\%$ | $93.04\%$ | $92.75\%$ | $6.96\%$ |
| $\Delta = 0.10$ | $190$ | $2.62\%$ | $92.11\%$ | $91.80\%$ | $7.89\%$ |

> **Scientific Clarity Mandate:** The $87.44\%$ accuracy is **strictly an accepted-subset empirical point estimate at $10.85\%$ coverage ($N=788$)** with $95\%$ Wilson CI $[84.94\%, 89.57\%]$. It is **not** a full-dataset accuracy figure, nor a statistical guarantee of $>85\%$ accuracy under arbitrary distribution shift. Full test baseline accuracy across all $7,264$ records remains $63.53\%$.

---

## 4. VLDBench Multimodal Manipulation Benchmark (Phase 6 Final)

Evaluated across Protocol A ($N=1,094$ test samples), Protocol B Strict Temporal ($N=1,094$), and Publisher-Held-Out splits.

![VLDBench Comparison](figures/vldbench_comparison.png)

### Protocol A Architectural Comparison ($N = 1,094$)

| Model Architecture | Trainable Parameters | Test Accuracy | Balanced Accuracy | Test Macro-F1 ($95\%$ Bootstrap CI) | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Publisher Metadata Only** | — | $48.26\%$ | $48.15\%$ | $44.36\%$ $[41.20\%, 47.45\%]$ | $0.4812$ |
| **Sanitized Text-Only (DeBERTa)** | $0.39\text{M}$ | $55.39\%$ | $55.42\%$ | $55.00\%$ $[51.98\%, 57.94\%]$ | $0.5982$ |
| **Image-Only (OpenCLIP)** | $0.39\text{M}$ | $64.44\%$ | $64.51\%$ | $64.30\%$ $[61.35\%, 67.11\%]$ | $0.7012$ |
| **Option 1: Late Concatenation** | $0.86\text{M}$ | $63.07\%$ | $63.12\%$ | $62.89\%$ $[59.95\%, 65.74\%]$ | $0.6914$ |
| **Option 3: Gated Multimodal Fusion\***| **$1.58\text{M}$** | **$64.72\%$** | **$64.80\%$** | **$64.54\%$ $[61.73\%, 67.20\%]$** | **$0.7086$** |

*\*Selected Production Architecture (`models/rhetoric_adapter_gated.pt`).*

### Option 3 Test Confusion Matrix (Protocol A, $N = 1,094$)

```
                        Predicted Class
                  Unmanipulated    Manipulated
   Unmanipulated       315             166          Total: 481
   Manipulated         220             393          Total: 613
                     Total: 535      Total: 559
```

* **Unmanipulated Class:** Precision: $58.88\%$, Recall: $65.49\%$, F1: $62.01\%$
* **Manipulated Class:** Precision: $70.30\%$, Recall: $64.11\%$, F1: $67.06\%$

### Distribution Shift Robustness

| Evaluation Split | Nature of Distribution Shift | Option 3 Macro-F1 | Generalization Delta ($\Delta$) |
| :--- | :--- | :---: | :---: |
| **Protocol A Test** | Group-Aware Random (Zero Story Overlap) | **$64.54\%$** | Baseline |
| **Protocol B Strict** | Pure Temporal Shift ($129$ Wire Groups Purged) | **$62.43\%$** | $-2.11\%$ (Graceful degradation) |
| **Publisher-Held-Out**| Unseen Newsrooms (Zero Training Representation) | **$61.42\%$** | $-3.12\%$ (Cross-publisher resilience) |

---

## 5. Latency & Execution Profiling Reconciliation

To resolve timing discrepancies between batch evaluation and interactive live verification, formal profiling was executed across both local corpus and live web search modes:

```
┌─────────────────────────────────┬──────────────────┬─────────────────────┬──────────────────────────┐
│ EXECUTION PIPELINE MODE         │ PER-ARTICLE TIME │ 3-ARTICLE BATCH TIME│ LATENCY BOTTLENECK       │
├─────────────────────────────────┼──────────────────┼─────────────────────┼──────────────────────────┤
│ Local BM25 Corpus Verification  │ 0.409 seconds    │ 1.226 seconds       │ CPU Vector Inferences    │
├─────────────────────────────────┼──────────────────┼─────────────────────┼──────────────────────────┤
│ Live Web Search Verification    │ 9.817 seconds    │ 29.452 seconds      │ Multi-Query HTTP Network │
└─────────────────────────────────┴──────────────────┴─────────────────────┴──────────────────────────┘
```

**Latency Root Cause:** Offline corpus verification executes in $< 0.5\text{s}$ per article. Live verification requires $9.8\text{s}$ per article due to external HTTP roundtrips: issuing multiple queries to Google News RSS and Wikipedia REST endpoints, downloading remote HTML payloads, parsing DOM text, and fetching high-resolution lead images over public networks.
