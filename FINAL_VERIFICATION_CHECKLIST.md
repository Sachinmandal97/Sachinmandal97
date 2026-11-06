# Final Verification Checklist — MSc Research Project

**Project Title:** Explainable Multimodal Vision-Language Framework for AI-Generated Fake News and Misinformation Detection  
**Reviewer Role:** Senior Machine Learning Research Engineer  
**Audit Date:** October 13, 2026  
**Target Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission` (Submission workspace only)  
**Protected Baseline Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-verifier` (Untouched, `7/10/2026 8:20:55 PM`)  

---

## 1. Verified Facts

### 1.1 Dataset Provenance and Cryptographic Integrity
* **Corpus:** NewsCLIPpings (Luo et al., Findings of EMNLP 2021), derived from VisualNews.
* **Licensing:** Apache-2.0 (Public, open academic/commercial access).
* **Inventory:** All 8 benchmark files are verified, uncorrupted, and preserved in `data/raw/newsclippings/`:
  - `merged_balanced/val.json`: 762,083 bytes | SHA-256: `892AB89F9DD94302CAADFEEA3CDE43C24499E22F34A3220158B0589FF767FE52`
  - `merged_balanced/test.json`: 788,016 bytes | SHA-256: `616E06AF39FD8AFA88D1920E455E4D6CA69E013C9B2FA4DADDA593FB0E813AEA`
  - `metadata/val.json`: 37,288,368 bytes | SHA-256: `D703C8836B0A6BFDDDD2F5967D4793CD003403573890AFD7282C47B8396D7D27`
  - `metadata/test.json`: 37,369,991 bytes | SHA-256: `1B2BC8438135AC8C5E4B78289F6FEC6D6956D92F30850945C057F3733A998BB5`
  - `embeddings/clip_text_embeddings/clip_text_embeddings_val.pkl`: 86,261,352 bytes | SHA-256: `3B345116BFC30D5D7BC3179E4D17B41853E7DE0776F6A1F80B1A9BE0A023124F`
  - `embeddings/clip_image_embeddings/clip_image_embeddings_val.pkl`: 105,641,474 bytes | SHA-256: `70B9C3EFEAC45D7A5A8A73A9025CDB62ABE40FB6D6809B9FC842BC299F4FD669`
  - `embeddings/clip_text_embeddings/clip_text_embeddings_test.pkl`: 86,368,983 bytes | SHA-256: `7E4C96F98952B57D3D2EBD83476DB3F1BB5202AC54DEDCDB94E0A04DEFCFA438`
  - `embeddings/clip_image_embeddings/clip_image_embeddings_test.pkl`: 105,640,860 bytes | SHA-256: `408AA7AD2F7319A18B51412985D65535D1945F4155742AA4889536CB90514570`
* **Embedding Verification:** 100% of validation records (7,024/7,024) and test records (7,264/7,264) contain authentic 512-dimensional unit-normalized OpenCLIP ViT-B/32 text and image vectors (`dtype=float32`, $||\mathbf{e}||_2 = 1.000000$). Zero missing embeddings.
* **Prevalence Notice:** The 50/50 balance in both splits is an artificial experimental design for sensitivity/specificity evaluation, not reflective of open-web news distributions where out-of-context images represent a rare anomaly ($< 5\%$).

### 1.2 Independent Metric Recalculation (From Raw Prediction JSONLs)
All reported metrics were independently re-derived directly from saved per-sample JSONL prediction records using `tests/verify_all_metrics.py`. Every number matches reported values to 4 decimal places:

| Run Key | File Path | $N$ | Accuracy | Balanced Acc | Macro-F1 | 95% Wilson CI | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Test ($\tau=0.22$)** | `outputs/.../newsclippings_test_full_tau22/...` | 7,264 | **54.68%** | 54.68% | **44.15%** | `[53.53%, 55.82%]` | **0.7197** |
| **Full Test ($\tau=0.28$)** | `outputs/.../newsclippings_test_full_tau28/...` | 7,264 | **63.53%** | 63.53% | **62.75%** | `[62.42%, 64.63%]` | **0.7197** |
| **Test 200 ($\tau=0.22$)** | `outputs/.../newsclippings_test_200_tau22/...` | 200 | **52.00%** | 52.00% | **41.11%** | `[45.10%, 58.82%]` | **0.6585** |
| **Test 200 ($\tau=0.28$)** | `outputs/.../newsclippings_test_200_tau28/...` | 200 | **56.50%** | 56.50% | **55.43%** | `[49.57%, 63.18%]` | **0.6585** |
| **Val 200 ($\tau=0.22$)** | `outputs/.../newsclippings_val_200/...` | 200 | **55.50%** | 55.50% | **46.77%** | `[48.57%, 62.22%]` | **0.7286** |
| **Val 100 ($\tau=0.22$)** | `outputs/.../newsclippings_val_100/...` | 100 | **56.00%** | 56.00% | **48.57%** | `[46.23%, 65.33%]` | **0.7314** |

### 1.3 Exact McNemar's Paired Significance Test
Using the paired predictions for all 7,264 test samples across both fixed thresholds:
* **Both Correct ($a$):** 3,244
* **Correct @ 0.22, Incorrect @ 0.28 ($b$):** 728
* **Incorrect @ 0.22, Correct @ 0.28 ($c$):** 1,371
* **Both Incorrect ($d$):** 1,921
* **Discordant Pairs ($b + c$):** **2,099**
* **Edwards Continuity-Corrected McNemar Statistic:**
  $$\chi^2 = \frac{(|728 - 1,371| - 1)^2}{2,099} = \frac{642^2}{2,099} = \mathbf{196.3621} \quad (\text{df} = 1)$$
* **Asymptotic $p$-Value:** $p = \mathbf{1.30 \times 10^{-44}}$
* **Exact Binomial Two-Sided $p$-Value:** $p = \mathbf{2.65 \times 10^{-45}}$
* **Conclusion:** The +8.85% accuracy gain of $\tau=0.28$ over $\tau=0.22$ is confirmed to be statistically significant beyond any doubt ($p \ll 0.001$).

### 1.4 Threshold Selection Provenance & Tie-Breaking Rule
* **Dataset Used for Calibration:** Exclusively the validation split (`val.json`, $N=100$ and $N=200$).
* **Primary Selection Metric:** Maximization of **Balanced Accuracy** (mean of class recalls).
* **Observed Data:**
  - On Val 100: Balanced Accuracy peaked at 67.00% across $\tau \in \{0.26, 0.28\}$. Between them, $\tau=0.28$ strictly dominated in Macro-F1 (66.60% vs. 64.41%).
  - On Val 200: Balanced Accuracy peaked at 67.50% across $\tau \in \{0.28, 0.29\}$.
* **Tie-Breaking Rule:** Between $\tau=0.28$ and $\tau=0.29$ on Val 200, $\tau=0.28$ was selected because it preserves superior specificity on the pristine class (Pristine Recall: 80.00% at $\tau=0.28$ vs. 71.00% at $\tau=0.29$) while tying top balanced accuracy (67.50%).
* **Protocol Timing:** Frozen in code and reports **before** test split evaluation.

### 1.5 Runtime Characterization
* **What was timed:** Precomputed 512-d OpenCLIP embedding indexing, PyTorch tensor cosine similarity calculation, threshold condition evaluation, and dataclass creation in `src/evaluation_harness.py`.
* **Latency:** 1.33 seconds total for 7,264 samples ($< 0.0002$ s/sample).
* **What was NOT timed:** Raw JPEG decoding, PIL image transforms, bicubic resizing, Vision Transformer pixel encoding, and BPE text tokenization. Live pixel inference would take approximately 18–36 minutes on CPU or 1–2 minutes on GPU.

### 1.6 Scientific Scope and Task Boundaries
* **Evaluated Task:** Visual-semantic consistency for out-of-context cheapfake news detection (authentic photographs paired with mismatched authentic captions).
* **Excluded Task:** Pixel-level synthetic AI image detection (Midjourney/DALL-E/GAN artifacts).

---

### 1.7 Application Workflow & Smoke-Test Audit
A dedicated end-to-end smoke test suite (`tests/test_final_app_smoke.py`) was executed on the live application pipeline covering all critical operational edge cases:
* **Image-Plus-Claim Processing (PASSED):** Verified end-to-end multimodal pipeline execution across OCR, OpenCLIP visual alignment, spaCy proposition decomposition, BM25 retrieval, DeBERTa NLI, decision fusion, and structured report synthesis.
* **Text-Only Input Handling (PASSED):** Verified that claim verification without images operates cleanly without errors. Cross-modal scoring returns neutral similarity ($0.50$) with `mismatch_detected=False` and explicit text-only metadata status.
* **Uncertain Evidence Verification (PASSED):** Obscure, unverified claims lacking external corroboration properly yield `DecisionVerdict.UNCERTAIN` with high uncertainty score ($\ge 0.50$) and explicit citation disclosure.
* **Conflicting Evidence Adjudication (PASSED):** When retrieved credible sources present opposing factual relations (entailment vs. contradiction), the pipeline correctly yields `DecisionVerdict.CONFLICTING` at both atomic claim and passage aggregation levels.
* **Unavailable Sources & Network Failure (PASSED):** Simulated API timeouts, network disconnections, and empty retrieval responses gracefully fall back to an unverified epistemic status without raising unhandled exceptions.
* **Malformed Input Resilience (PASSED):** The system successfully ingests and handles empty strings, whitespace-only claims, extreme text lengths ($> 5,000$ characters), missing image paths, corrupted non-image binaries, and Unicode/HTML/null-byte injections without crashing.
* **Preset Demo Visual Isolation (PASSED):** Verified that custom claims cannot accidentally reuse preset demonstration visuals. Editing or typing a custom claim in the interface immediately decouples the preset visual (`temp_img_file_path = None`), forcing text-only mode unless an image is explicitly uploaded.

---

## 2. Corrections Made

1. **Statistical Significance Phrasing:** Replaced the generic informal note "$p < 0.001$" with the exact Edwards continuity-corrected McNemar $\chi^2 = 196.3621$ ($p = 1.30 \times 10^{-44}$) and exact binomial $p = 2.65 \times 10^{-45}$.
2. **Threshold Tie-Breaking Rule:** Explicitly formalized the tie-breaking protocol between candidate thresholds $\tau=0.28$ and $\tau=0.29$ based on pristine specificity preservation.
3. **Latency Description:** Added explicit disclaimers in all reports that 1.33-second throughput measures precomputed-vector comparison, not full pixel-level Vision Transformer inference.
4. **Smoke-Test Assertions:** Aligned text-only smoke test assertions with the designed neutral fallback score ($\text{sim} = 0.50$, $\text{mismatch} = \text{False}$).

---

## 3. Unresolved Issues and Research Blockers

| Research Axis | Status | Blocker Description | Resolution Path |
| :--- | :---: | :--- | :--- |
| **Factify 2 (Task B & Task C)** | **Blocked** | Account-gated behind CodaLab Competition 8275 and registration Google Forms. Cannot be accessed by automated AI agents. | User must download `val.json` from CodaLab and place in `data/raw/factify2/val.json`. Adapter is implemented and ready. Automated pursuit stopped; no benchmark results fabricated. |
| **Pipeline Ablations (`no_decomp`, `no_rel_filter`, `text_only`, `forced_binary`)** | **Reserved** | Methodologically inapplicable to NewsCLIPpings (single image-caption pairs without retrieval). | Will execute immediately once multi-proposition Factify 2 data is placed. |

---

## 4. Commands Executed and Verification Reproducibility

### Evaluation Commands:
```bash
# Full Test Baseline (tau=0.22)
python evaluate_benchmark.py --dataset newsclippings --task task_a --split test --threshold-cm 0.22 --output-dir outputs/benchmark_results/newsclippings_test_full_tau22

# Full Test Calibrated (tau=0.28)
python evaluate_benchmark.py --dataset newsclippings --task task_a --split test --threshold-cm 0.28 --output-dir outputs/benchmark_results/newsclippings_test_full_tau28

# Independent Metric Recalculation
python tests/verify_all_metrics.py

# Generator Breakdown Analysis
python tests/analyze_test_generators.py

# Application Smoke Test Suite
pytest tests/test_final_app_smoke.py -v
```

---

## 5. Automated Test Suite Results

All test suites were executed on Python 3.13.5 (PyTest 9.1.1):

| Test Module | Tests Run | Result | Duration | Scope Tested |
| :--- | :---: | :---: | :---: | :--- |
| `tests/test_final_app_smoke.py` | 7 | **7 PASSED** | 204.81 s | Image-plus-claim, text-only, uncertain/conflicting evidence, unavailable sources, malformed input, preset isolation |
| `tests/test_benchmark_infrastructure.py` | 9 | **9 PASSED** | 23.36 s | Wilson CIs, Mann-Whitney ROC-AUC, Stratified sampling, Error isolation, Genuine data loading |
| `tests/test_phase_p0_fixes.py` | 8 | **8 PASSED** | 4.23 s | Root URL rejection, Unresolved wrapper handling, Long claim tokenization, Text-only decoupling |
| `tests/test_evidence_hotfix.py` | 8 | **8 PASSED** | 5.12 s | Live fetch throttling, Snippet preservation, Content extraction guards |
| `tests/test_submission_pipeline.py` | 1 | **1 PASSED** | 3.40 s | Pipeline initialization and offline fallback |
| `tests/test_nli_diagnostic.py` | 9 | **9 PASSED** | 24.87 s | DeBERTa NLI premise scoring, Contradiction precedence, Batch inference |
| `tests/test_universal_fact_checking.py` | 10 | **10 PASSED** | 4.50 s | Multi-category claim decomposition, Epistemic verdict aggregation |
| `tests/test_universal_verification.py` | 10 | **10 PASSED** | 3.92 s | Search provider parsing, Query TTL caching, Irrelevant evidence filtering |
| **TOTAL** | **62** | **62 PASSED (0 Failures)** | **274.21 s** | Complete submission codebase verification |

---

## 6. Protected Baseline Workspace Verification

* **Path:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-verifier`
* **Last Modified Timestamp:** `7/10/2026 8:20:55 PM`
* **Status:** **Completely untouched**. No files modified, created, or deleted.

---

## 7. Final Recommendation & Completion Verdict

**Software Finalization Phase Complete:** All documented application workflows, edge cases, and automated test suites (62/62 tests) pass without errors. 

1. **Factify 2 Status:** Genuine authorized Factify 2 data is not available in the repository due to external CodaLab/Google Forms authentication barriers. In compliance with research integrity standards, automatic download attempts have ceased, and no synthetic or fabricated benchmark results have been substituted.
2. **Task A Evaluation:** The empirical benchmark evaluation on the full NewsCLIPpings test set ($N=7,264$) is complete, mathematically and statistically verified (McNemar $\chi^2 = 196.3621$, $p \ll 0.001$), and auditable from raw saved predictions.
3. **No Further Development:** With all application smoke tests, integration tests, and benchmark evaluations fully passing and verified, no further development phases are required. The codebase is frozen for MSc dissertation submission.
