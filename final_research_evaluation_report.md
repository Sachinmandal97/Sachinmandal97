# Final Research and Evaluation Report

**Degree Programme:** MSc in Data Science  
**Project Title:** Explainable Multimodal Vision-Language Framework for AI-Generated Fake News and Misinformation Detection  
**Module:** MSc Dissertation / Final Research Project  
**Date:** October 13, 2026  
**Status:** Complete & Audited (NewsCLIPpings Empirical Benchmark Complete; Factify 2 Gated)  
**Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission`  
**Protected Baseline Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-verifier` (Untouched)  

---

## 1. Project Context, Research Scope, and Core Definitions

This document consolidates the complete empirical evaluation, statistical verification, architectural diagnostics, and benchmark results for the MSc Data Science dissertation project.

### 1.1 Distinct Operational Scopes & Research Boundaries
To ensure academic integrity, the project strictly delineates three separate operational and methodological domains:

1. **NewsCLIPpings Image-Caption Consistency Evaluation (Task A — Cheapfakes):**
   - **What it is:** Empirical benchmark evaluation of out-of-context multimodal misinformation. Authentic news photographs paired with mismatched authentic captions.
   - **Mechanism:** Cosine similarity calculation between unit-normalized OpenCLIP ViT-B/32 image and text embeddings against calibrated decision thresholds ($\tau=0.22$ baseline, $\tau=0.28$ calibrated).
   - **Empirical Status:** **Completed, audited, and statistically verified** on the full untouched NewsCLIPpings test split ($N=7,264$, McNemar $\chi^2 = 196.3621$, $p \ll 0.001$).

2. **The Application's Evidence-Grounded Claim-Verification Workflow (Live RAG System):**
   - **What it is:** The interactive Streamlit software platform (`app.py`, `src/pipeline.py`) designed for real-world news verification.
   - **Mechanism:** Multi-stage retrieval-augmented verification comprising linguistic decomposition into atomic propositions via spaCy, live multi-query retrieval (Google News RSS & Wikipedia), entity-anchored topical relevance filtering, directional premise–hypothesis stance estimation via fine-tuned DeBERTa-v3 NLI, and transparent epistemic fusion (Supported, Partially Supported, Contradicted, Conflicting, Uncertain).
   - **Empirical Status:** **Completed and verified via end-to-end smoke tests and integration suites** (62/62 passing tests).

3. **Pixel-Level AI-Generated-Image Detection (Explicit Non-Claim & Technical Boundary):**
   - **What it is:** Forensic classification of synthetic, diffusion-generated, or GAN-manipulated pixel structures (e.g., Midjourney, DALL-E, Stable Diffusion artifacts, DeepFakes).
   - **Mandatory Academic Boundary:** **This capability is NOT implemented and MUST NOT be claimed as demonstrated.** OpenCLIP measures high-level vision-language semantic alignment; it does not perform low-level spatial or frequency-domain artifact analysis (such as DIRE, Spec, or Steganalysis). Visual mismatches flagged by this framework denote semantic discordance between image and text ("cheapfakes"), never proof of synthetic generative pixel manipulation.

### 1.2 Evaluation Hierarchy
To preserve scientific rigor in the dissertation results chapter, four distinct layers of evaluation executed across this project must remain clearly differentiated:
1. **Synthetic Fixtures (Phase P0 / Pilot 2B):** 5-sample offline unit test fixtures used solely to test pipeline error handling and JSON schema parsing. These are not empirical benchmark results.
2. **Live-Web Smoke Tests (Phase P0.1 & Phase 2E):** Qualitative and regression end-to-end demonstrations verifying live Google News RSS search, HTTP content fetching, DeBERTa NLI premise scoring, and edge-case resilience.
3. **Validation Calibration (Phase 2C):** Exploratory empirical diagnostics conducted on `val.json` to diagnose decision boundary shift and calibrate the optimal operating threshold ($\tau^* = 0.28$).
4. **Untouched Benchmark Evaluation (Phase 2D):** Frozen, un-tuned evaluation conducted across the complete official test split (`test.json`, $N=7,264$), serving as the primary empirical evidence for the dissertation.

---

## 2. Dataset Provenance, Split Sizes, and Cryptographic Integrity

All benchmark files are preserved in `data/raw/newsclippings/` within the submission workspace. Cryptographic SHA-256 hashes confirm bit-for-bit authenticity against the official repository (Luo et al., Findings of EMNLP 2021; Apache-2.0 license):

| Asset Name | Relative Workspace Path | Size (Bytes) | SHA-256 Checksum | Split Role |
| :--- | :--- | :---: | :--- | :---: |
| **`val.json` (Annotations)** | `data/raw/newsclippings/merged_balanced/val.json` | 762,083 B | `892AB89F9DD94302CAADFEEA3CDE43C24499E22F34A3220158B0589FF767FE52` | Validation |
| **`test.json` (Annotations)** | `data/raw/newsclippings/merged_balanced/test.json` | 788,016 B | `616E06AF39FD8AFA88D1920E455E4D6CA69E013C9B2FA4DADDA593FB0E813AEA` | Test |
| **`val.json` (Metadata)** | `data/raw/newsclippings/metadata/val.json` | 37,288,368 B | `D703C8836B0A6BFDDDD2F5967D4793CD003403573890AFD7282C47B8396D7D27` | Validation |
| **`test.json` (Metadata)** | `data/raw/newsclippings/metadata/test.json` | 37,369,991 B | `1B2BC8438135AC8C5E4B78289F6FEC6D6956D92F30850945C057F3733A998BB5` | Test |
| **`clip_text_embeddings_val.pkl`** | `data/raw/newsclippings/embeddings/clip_text_embeddings/clip_text_embeddings_val.pkl` | 86,261,352 B | `3B345116BFC30D5D7BC3179E4D17B41853E7DE0776F6A1F80B1A9BE0A023124F` | Validation |
| **`clip_image_embeddings_val.pkl`** | `data/raw/newsclippings/embeddings/clip_image_embeddings/clip_image_embeddings_val.pkl` | 105,641,474 B | `70B9C3EFEAC45D7A5A8A73A9025CDB62ABE40FB6D6809B9FC842BC299F4FD669` | Validation |
| **`clip_text_embeddings_test.pkl`** | `data/raw/newsclippings/embeddings/clip_text_embeddings/clip_text_embeddings_test.pkl` | 86,368,983 B | `7E4C96F98952B57D3D2EBD83476DB3F1BB5202AC54DEDCDB94E0A04DEFCFA438` | Test |
| **`clip_image_embeddings_test.pkl`** | `data/raw/newsclippings/embeddings/clip_image_embeddings/clip_image_embeddings_test.pkl` | 105,640,860 B | `408AA7AD2F7319A18B51412985D65535D1945F4155742AA4889536CB90514570` | Test |

### Split Characteristics & Prevalence Notice
* **Validation Split:** 7,024 samples (3,512 pristine, 3,512 falsified; 100% embedding coverage).
* **Test Split:** 7,264 samples (3,632 pristine, 3,632 falsified; 100% embedding coverage).
* **Prevalence Note:** The exact 50/50 balance in both splits is an artificial experimental condition to prevent evaluation bias. In open web environments, false context media represents a low-prevalence anomaly ($< 5\%$). Researchers must not infer that uncurated news streams exhibit 50% misinformation rates.

---

## 3. Threshold Calibration Rationale and Freezing Protocol

### 3.1 Mathematical Formulation of Score Direction
The continuous cross-modal consistency score $s_{TI}$ is defined as the cosine similarity between the $L_2$-normalized OpenCLIP ViT-B/32 text embedding $\mathbf{e}_T$ and image embedding $\mathbf{e}_I$:
$$s_{TI} = \frac{\mathbf{e}_T \cdot \mathbf{e}_I}{\|\mathbf{e}_T\|_2 \|\mathbf{e}_I\|_2} = \cos(\mathbf{e}_T, \mathbf{e}_I) \in [-1.0, 1.0]$$
The binary decision rule maps higher similarity to pristine pairings:
$$\hat{y} = \begin{cases} \text{pristine (0)}, & \text{if } s_{TI} \ge \tau_{\text{cm}} \\ \text{falsified (1)}, & \text{if } s_{TI} < \tau_{\text{cm}} \end{cases}$$

### 3.2 Diagnostic Findings on Validation Data
The baseline threshold $\tau_{\text{cm}} = 0.22$ was inherited *a priori* from the initial heuristic pipeline configuration. When evaluated on the NewsCLIPpings validation split, empirical analysis revealed a marked decision boundary shift:
* **Pristine Sample Mean ($N=50$):** $\mu_{\text{pristine}} = \mathbf{0.3040}$ (Median: `0.3036`).
* **Falsified Sample Mean ($N=50$):** $\mu_{\text{falsified}} = \mathbf{0.2687}$ (Median: `0.2780`).

Because adversarial cheapfakes are generated using semantic nearest neighbors (e.g., text-to-text semantic matching), the average falsified similarity ($0.2687$) sits substantially above the heuristic threshold ($0.22$). Consequently, at $\tau = 0.22$, 82–85% of falsified samples were falsely classified as pristine.

### 3.3 Freezing Threshold $\tau = 0.28$ Prior to Test Evaluation
In Phase 2C, a validation sensitivity sweep across $\tau \in [0.18, 0.32]$ demonstrated that balanced accuracy and Macro-F1 peaked at $\tau^* \in [0.28, 0.29]$:
* At $\tau = 0.28$: Balanced Accuracy = **67.50%**, Macro-F1 = **66.98%** (Pristine Recall: 80.0%, Falsified Recall: 55.0%).
* At $\tau = 0.29$: Balanced Accuracy = **67.50%**, Macro-F1 = **67.46%** (Pristine Recall: 71.0%, Falsified Recall: 64.0%).

Threshold $\tau = 0.28$ was selected as the calibrated operating point and **frozen in documentation prior to test evaluation**. Neither threshold was modified, tuned, or re-selected based on test results.

---

## 4. Consolidated Benchmark Evaluation Results

### 4.1 Exact Execution Commands
```bash
# 1. Validation Run (N=200, Baseline Heuristic tau=0.22)
python evaluate_benchmark.py --dataset newsclippings --task task_a --split val --limit 200 --seed 42 --threshold-cm 0.22 --output-dir outputs/benchmark_results/newsclippings_val_200

# 2. Test Subset Run (N=200, Baseline Heuristic tau=0.22)
python evaluate_benchmark.py --dataset newsclippings --task task_a --split test --limit 200 --seed 42 --threshold-cm 0.22 --output-dir outputs/benchmark_results/newsclippings_test_200_tau22

# 3. Test Subset Run (N=200, Calibrated tau=0.28)
python evaluate_benchmark.py --dataset newsclippings --task task_a --split test --limit 200 --seed 42 --threshold-cm 0.28 --output-dir outputs/benchmark_results/newsclippings_test_200_tau28

# 4. Full Test Run (N=7,264, Baseline Heuristic tau=0.22)
python evaluate_benchmark.py --dataset newsclippings --task task_a --split test --threshold-cm 0.22 --output-dir outputs/benchmark_results/newsclippings_test_full_tau22

# 5. Full Test Run (N=7,264, Calibrated tau=0.28)
python evaluate_benchmark.py --dataset newsclippings --task task_a --split test --threshold-cm 0.28 --output-dir outputs/benchmark_results/newsclippings_test_full_tau28
```

### 4.2 Comprehensive Empirical Results Table
| Evaluation Run | Split | $N$ | Threshold $\tau$ | Accuracy | 95% Wilson CI | Balanced Acc | Macro-Prec | Macro-Rec | Macro-F1 | ROC-AUC | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Exploratory** | `val` | 100 | 0.22 | 56.00% | `[46.23%, 65.33%]` | 56.00% | 64.20% | 56.00% | 48.57% | 0.7314 | 0.99 s |
| **Validation Exploratory** | `val` | 200 | 0.22 | 55.50% | `[48.57%, 62.22%]` | 55.50% | 65.99% | 55.50% | 46.77% | 0.7286 | 0.89 s |
| **Validation Calibrated** | `val` | 200 | 0.28 | **67.50%** | `[60.79%, 73.57%]` | 67.50% | 68.75% | 67.50% | **66.98%** | 0.7286 | 0.89 s |
| **Test Subset Baseline** | `test` | 200 | 0.22 | 52.00% | `[45.10%, 58.82%]` | 52.00% | 57.68% | 52.00% | 41.11% | 0.6585 | 0.90 s |
| **Test Subset Calibrated** | `test` | 200 | 0.28 | **56.50%** | `[49.57%, 63.18%]` | 56.50% | 57.19% | 56.50% | **55.43%** | 0.6585 | 0.91 s |
| **Full Official Test Split** | `test` | 7,264 | 0.22 | **54.68%** | `[53.53%, 55.82%]` | 54.68% | 69.04% | 54.68% | **44.15%** | **0.7197** | 1.42 s |
| **Full Official Test Split** | `test` | 7,264 | 0.28 | **63.53%** | `[62.42%, 64.63%]` | 63.53% | 64.78% | 63.53% | **62.75%** | **0.7197** | 1.33 s |

### 4.3 Confusion Matrices on Full Test Split ($N = 7,264$)
* **Full Test Baseline ($\tau = 0.22$):**
  - Pristine: **3,563** True Pristine, **69** False Falsified (Recall: **98.10%**, Precision: **52.51%**, F1: **68.40%**).
  - Falsified: **409** True Falsified, **3,223** False Pristine (Recall: **11.26%**, Precision: **85.56%**, F1: **19.90%**).
* **Full Test Calibrated ($\tau = 0.28$):**
  - Pristine: **2,835** True Pristine, **797** False Falsified (Recall: **78.06%**, Precision: **60.49%**, F1: **68.16%**).
  - Falsified: **1,780** True Falsified, **1,852** False Pristine (Recall: **49.01%**, Precision: **69.07%**, F1: **57.34%**).

---

## 5. Statistical Significance Testing (Paired McNemar's Test)

To formally test whether the +8.85 percentage point accuracy gain achieved by threshold calibration ($\tau=0.28$ vs. $\tau=0.22$) is statistically significant, **McNemar's test for paired binary classifications** was computed across all 7,264 individual test predictions.

### 5.1 Paired $2 \times 2$ Contingency Table
Let $C_{22}(i) \in \{0, 1\}$ and $C_{28}(i) \in \{0, 1\}$ represent correctness on test sample $i$:

| Metric | Threshold $\tau = 0.28$ Correct | Threshold $\tau = 0.28$ Incorrect | Marginal Total ($\tau = 0.22$) |
| :--- | :---: | :---: | :---: |
| **Threshold $\tau = 0.22$ Correct** | $a = 3,244$ (Both Correct) | $b = 728$ ($0.22$ Wins) | 3,972 (54.68%) |
| **Threshold $\tau = 0.22$ Incorrect** | $c = 1,371$ ($0.28$ Wins) | $d = 1,921$ (Both Incorrect) | 3,292 (45.32%) |
| **Marginal Total ($\tau = 0.28$)** | 4,615 (63.53%) | 2,649 (36.47%) | $N = 7,264$ |

### 5.2 Test Statistics and $p$-Values
* **Discordant Pairs:** $b + c = 728 + 1,371 = \mathbf{2,099}$.
* **McNemar $\chi^2$ Statistic (Edwards Continuity Correction):**
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} = \frac{(|728 - 1,371| - 1)^2}{2,099} = \frac{(642)^2}{2,099} = \mathbf{196.3621} \quad (\text{df} = 1)$$
* **Asymptotic $p$-Value ($\chi^2$, 1 df):** $p = \mathbf{1.30 \times 10^{-44}}$.
* **Exact Two-Sided Binomial $p$-Value:** $p = \mathbf{2.65 \times 10^{-45}}$.
* **Conclusion:** The null hypothesis of marginal homogeneity is rejected ($p \ll 0.001$). The +8.85% accuracy improvement is confirmed to be highly statistically significant.

---

## 6. Runtime Latency Characterization

> [!IMPORTANT]
> **Runtime Measurement Disclosure:** The reported latency of **1.33 seconds for 7,264 test samples** ($< 0.0002$ seconds / sample) represents **precomputed-embedding cosine similarity scoring, tensor indexing, and threshold decision time**. It does **not** measure full end-to-end model inference from raw image pixels.
>
> If raw image rasterization, PIL loading, and live OpenCLIP ViT-B/32 neural encoding were executed on 7,264 images:
> * Expected CPU latency: $\approx 150\text{--}300\text{ ms}$ per sample ($\approx 18\text{--}36$ minutes total).
> * Expected GPU latency: $\approx 10\text{--}20\text{ ms}$ per sample ($\approx 1.2\text{--}2.4$ minutes total).
>
> Using official benchmark embeddings isolates decision-boundary calibration from compute hardware bottlenecks.

---

## 7. Breakdown by Adversarial Generator

Disaggregating performance across the four official NewsCLIPpings generation algorithms ($N=1,816$ each) reveals structural strengths and vulnerabilities of zero-shot visual-semantic matching:

| Generator Name | Mismatch Method | Acc ($\tau=0.22$) | Acc ($\tau=0.28$) | Pristine Rec ($\tau=0.28$) | Falsified Rec ($\tau=0.28$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Gen 1: `semantics_clip_text_text`** | Text semantic nearest neighbor | 55.12% | **81.44%** | 89.10% | **73.79%** |
| **Gen 2: `scene_resnet_place`** | Scene feature substitution | 60.35% | **67.79%** | 72.03% | **63.55%** |
| **Gen 3: `person_sbert_text_text`** | Person entity overlap + SBERT | 54.52% | **61.07%** | 82.27% | **39.87%** |
| **Gen 0: `semantics_clip_text_image`** | Unconstrained CLIP text-to-image NN | 48.73% | **43.83%** | 68.83% | **18.83%** |

### Insights for the Dissertation Discussion:
1. **Strong Detection on Cross-Modal Topic Shifts (Gen 1 & 2):** When an out-of-context image depicts a different scene or news event, the calibrated comparator achieves 81.44% and 67.79% accuracy.
2. **Adversarial Blindspots in Shared Metric Space (Gen 0):** In Gen 0, the adversary deliberately matched the fake image by maximizing CLIP cosine similarity against the claim caption. Because the adversary optimizes the same metric space used for defense, zero-shot CLIP similarity is partially blinded (43.83% accuracy), demonstrating that zero-shot visual similarity alone cannot defend against representation-aware adversaries without external factual knowledge retrieval.

---

## 8. Factify 2 Status and Outstanding Experiments

### 8.1 Status: Access-Blocked Experiment (Not a Completed Benchmark)
Factify 2 is formally documented as an **access-blocked experiment, not a completed benchmark**. As established during Phase 2C, authentic Factify 2 data is hosted on [CodaLab Competitions (ID: 8275)](https://codalab.lisn.upsaclay.fr/competitions/8275) and requires user credential authentication and completion of De-Factify Google Forms. Under university research ethics and automated agent constraints, the agent did not fabricate credentials, bypass authentication, or automate terms consent.

In strict adherence to research integrity standards, **no synthetic, mock, or fabricated data was substituted as benchmark results**.

### 8.2 List of Outstanding Experiments (Awaiting Human Data Download)
Because authentic Factify 2 data could not be downloaded automatically, the following planned experiments remain outstanding:
1. **Factify 2 Task B (Textual Claim–Document Verification):** NLI-based factual stance verification evaluating whether reference document text entails, neutralizes, or refutes claim assertions.
2. **Factify 2 Task C (Multimodal Claim–Document Verification):** 5-class and 3-class multimodal epistemic adjudication across claim–document pairs (`Support_Multimodal`, `Support_Text`, `Refute`, `Neutral`, `Insufficient_Multimodal`).
3. **Pipeline Ablation 1 (`--ablation no_decomp`):** Measuring verification accuracy loss when compound news assertions are not decomposed into atomic propositions.
4. **Pipeline Ablation 2 (`--ablation no_rel_filter`):** Measuring error propagation when entity-anchored topical relevance filtering is removed from candidate evidence.
5. **Pipeline Ablation 3 (`--ablation text_only`):** Quantifying the precise margin of multimodal improvement by evaluating the pipeline without the OpenCLIP visual branch.
6. **Pipeline Ablation 4 (`--ablation forced_binary`):** Evaluating classification degradation when collapsing granular epistemic verdicts into a forced binary ("Real" vs "Fake") taxonomy.

*Note:* All corresponding dataset adapters, ablation switches, and metric calculators are implemented in `src/benchmark_adapters.py` and `evaluate_benchmark.py` and verified via unit tests (`tests/test_benchmark_infrastructure.py`). They will execute immediately if authentic data is provided.

### 8.3 Factify 2 Onboarding Checklist (For the User)
Once the human researcher signs into CodaLab and downloads the validation set, the existing architecture is fully implemented to ingest it immediately:

* [ ] **Target Directory:** `data/raw/factify2/`
* [ ] **Expected Filename:** `data/raw/factify2/val.json` (or `.jsonl`, `.tsv`, `.csv`)
* [ ] **Required JSON Keys per Record:**
  - `sample_id` (string): Unique identifier (e.g., `"f2_val_001"`).
  - `claim_text` (string): The factual assertion under verification.
  - `label` (string): Must belong to `{"Support_Multimodal", "Support_Text", "Refute", "Neutral", "Insufficient_Multimodal"}`.
* [ ] **Optional Context Keys:**
  - `document_text` (string): Paired reference evidence text.
  - `claim_image` (string): Relative path or URI to claim image.
  - `document_image` (string): Relative path or URI to evidence image.
* [ ] **Validation Verification Command:**
  ```bash
  python -c "from src.data_loader import load_factify2_metadata; samples = load_factify2_metadata('data/raw/factify2/val.json'); print(f'Validated {len(samples)} records.')"
  ```
* [ ] **Evaluation Command (Once Placed):**
  ```bash
  python evaluate_benchmark.py --dataset factify2 --task task_b --split val
  python evaluate_benchmark.py --dataset factify2 --task task_c --split val --protocol c2
  ```

---

## 9. Limitations and Verification Status

1. **Application Workflow & Smoke-Test Verification (PASSED):** All 7 critical operational workflow scenarios in `tests/test_final_app_smoke.py` passed with 100% compliance:
   - Image-plus-claim end-to-end multimodal pipeline
   - Text-only input (clean image-free mode with neutral 0.50 score)
   - Uncertain evidence (unsubstantiated claims yield UNCERTAIN)
   - Conflicting evidence (material disagreement yields CONFLICTING)
   - Unavailable sources & network failure fallback (graceful degradation)
   - Malformed input resilience (empty, whitespace, 5000+ chars, invalid image, corrupted image, unicode)
   - Preset demo visual isolation (custom claims cannot accidentally reuse preset demo images)
2. **Factify 2 External Blocker & Research Integrity:** Genuine authorized Factify 2 data is account-gated behind CodaLab Competition 8275 and registration Google Forms requiring human account authentication. Automatic agent pursuit has ceased. In compliance with strict research integrity standards, no synthetic or fabricated benchmark results have been substituted.
3. **Pipeline Ablations Reserved:** The four architectural ablations (`no_decomp`, `no_rel_filter`, `text_only`, `forced_binary`) are methodologically designed for multi-premise evidence retrieval (Factify 2) and are preserved in `evaluate_benchmark.py` ready for ingestion if authentic data is supplied.
4. **Domain Transfer:** While test ROC-AUC (0.7197) mirrors validation ROC-AUC (0.7286), NewsCLIPpings evaluates professional journalism photography (VisualNews). Performance on social media memes or compressed mobile images remains unassessed.

---

## 10. Software Finalization Phase Completion Verdict

**Software Finalization Phase Complete:** All application workflows, edge cases, and automated test suites (62/62 tests passing, 0 failures) have been successfully verified. The protected baseline workspace remains untouched (`7/10/2026 8:20:55 PM`). 

No further development phases are required. The empirical evaluation for Task A is complete, statistically verified, and frozen for MSc dissertation submission. Incorporate these validated findings, McNemar statistical tests, and generator breakdown tables into Chapter 4 (Results) and Chapter 5 (Discussion) of the MSc dissertation.
