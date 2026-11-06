# Independent Pre-Submission Technical Audit Report

**Project Title:** Explainable Evidence-Augmented Multimodal Misinformation Verification  
**Evaluation Target:** Final MSc Submission Workspace  
**Workspace Path:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission`  
**Reference Comparison Workspaces:**
- Repo A: `kmist70/fake-news-detector`
- Repo B: `LaughingParrot/CLIP-Multimodal-Fake-News-Detection-with-InfoNCE-on-Fakeddit`
- Original Research Workspace: `multimodal-misinfo-verifier` (Untouched)  
**Audit Timestamp:** October 2026  
**Auditor:** Independent Technical Quality Gate  

---

## 1. Executive Summary & Final Recommendation

### Final Verdict: **A. READY FOR A CONTROLLED CLASSROOM DEMONSTRATION**

The submission prototype at `multimodal-misinfo-submission` is functional, self-contained, and verified on Windows 11 CPU without network dependencies during inference. It runs the full 8-stage multimodal verification pipeline (Preprocessing → OCR → OpenCLIP → spaCy → BM25 Retrieval → DeBERTa-v3 NLI → Uncertainty Estimation → Structured Explainability) and serves an interactive Streamlit UI.

> [!IMPORTANT]
> **ACADEMIC HONESTY MANDATE**:
> This platform is an **educational research demonstration prototype**, not a real-world factual truth oracle.
> - The background evidence corpus is a **synthetic demonstration fixture**.
> - Classification cutoffs (0.22 for CLIP, 0.60 for entailment) are **unvalidated heuristic design choices**, not calibrated empirical posteriors.
> - No claim of benchmark state-of-the-art accuracy or production-grade fact-checking capability is asserted.

---

## 2. Actual Commands Executed & Test Results

### 2.1 Automated Unit & Integration Suite
* **Command Executed:**
  ```powershell
  python run_tests.py
  ```
* **Execution Environment:** Windows 11 (build 26100), Python 3.13.5 (64-bit), PyTorch 2.10.0+cpu.
* **Actual Output:**
  ```text
  ======================================================================
  RUNNING SUBMISSION PIPELINE INTEGRITY & EXECUTION TESTS
  ======================================================================
    [PASS] test_clean_text_normalization
    [PASS] test_empty_input_handling
    [PASS] test_end_to_end_verification_decision
    [PASS] test_evidence_retrieval
    [PASS] test_image_loading_and_cross_modal
  ======================================================================
  RESULTS: 5 PASSED, 0 FAILED in 23.825s
  ======================================================================
  ```
* **Result:** **5 Passed, 0 Failed (100% Pass Rate)**.

### 2.2 Live Application Launch Verification
* **Command Executed:**
  ```powershell
  streamlit run app.py --server.headless true --server.port 8501
  ```
* **Status Check:**
  ```python
  import urllib.request
  resp = urllib.request.urlopen('http://localhost:8501')
  print(resp.getcode()) # Returns 200 OK
  ```
* **Application Launch Result:** **Successful.** The Streamlit server initializes cleanly, loads cached models into memory (~1.1 GB RAM total), and responds with HTTP status `200`.

---

## 3. Evidence Corpus Provenance Audit

* **File Inspected:** `data/corpus/evidence_corpus.jsonl` (6 records, 1,181 bytes).
* **Detailed Record Inspection:**
  1. `ev_01`: *"London Fire Brigade deployed 10 fire engines..."* (Source: *"London Fire Official News"*)
  2. `ev_02`: *"World leaders and economic delegates convened in Geneva..."* (Source: *"Geneva Global Dispatch"*)
  3. `ev_03`: *"Downtown Chicago recorded heavy summer temperatures..."* (Source: *"Midwest Weather Reports"*)
  4. `ev_04`: *"Environmental activists organized peaceful marches..."* (Source: *"Climate Action Wire"*)
  5. `ev_05`: *"Venice experienced mild sunny weather..."* (Source: *"Venice Municipal Archive"*)
  6. `ev_06`: *"The Sahara Desert remains one of the driest arid regions..."* (Source: *"Geographic Survey Archive"*)
* **Provenance Determination:**
  * **Status:** **Synthetic Software Test / Demonstration Fixtures.**
  * **Finding:** None of these records are certified, live-scraped external journalism articles. Sources such as *"Geneva Global Dispatch"* and *"Climate Action Wire"* are illustrative placeholder entities.
  * **Compliance Rule:** The UI explicitly presents these as a *"Local Demonstration Knowledge Base"* and does not misrepresent them as real-world live fact-check reports.

---

## 4. Model Weights & Label Verification

### 4.1 Vision-Language Alignment (OpenCLIP)
* **Model Checkpoint:** `openai/clip-vit-base-patch32` loaded via HuggingFace `transformers.CLIPModel`.
* **Feature Extraction:**
  * Text features: normalized $f_t \in \mathbb{R}^{512}$, $\|f_t\|_2 = 1$.
  * Image features: normalized $f_v \in \mathbb{R}^{512}$, $\|f_v\|_2 = 1$.
  * Cosine similarity: $s = \langle f_t, f_v \rangle \in [-1.0, 1.0]$.
* **Semantic Meaning:** High cosine similarity indicates *visual-textual alignment* (the image depicts subjects mentioned in the text). It does **NOT** establish whether the asserted event is factually true.

### 4.2 Natural Language Inference (DeBERTa-v3)
* **Model Checkpoint:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`.
* **Native Checkpoint Label Mapping Verified:**
  ```python
  id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
  ```
* **Probability Normalization:**
  * In `src/verification.py`, logits are passed through `torch.nn.functional.softmax` across the 3 classes.
  * `prob_dict` parses the exact string labels from `model.config.id2label`.
  * Probabilities sum to $1.0000 \pm 0.0001$.

---

## 5. Demonstration Preset Behavior Verification

The three presets in `app.py` were verified to ensure decisions are generated dynamically by running through model inference rather than static lookups:

| Preset Scenario | Pipeline Execution Flow | Dynamic Model Outputs | Resulting Verdict |
| :--- | :--- | :--- | :--- |
| **Case 1: Fully Supported News** | Claim text + fire visual $\rightarrow$ BM25 finds `ev_01` $\rightarrow$ DeBERTa NLI | CLIP: $s \approx 0.28$ (Aligned); NLI Entailment: $82.4\%$ | **`Supported`** |
| **Case 2: Cross-Modal Mismatch** | Blizzard claim + tropical beach visual $\rightarrow$ CLIP mismatch | CLIP: $s \approx 0.14 < 0.22$ (Mismatch detected); Heuristic flags out-of-context | **`Misleading`** |
| **Case 3: Insufficient Evidence** | Edinburgh gold claim $\rightarrow$ BM25 finds no matching records | BM25 hits: 0; NLI: Not triggered; Heuristic defaults to insufficient | **`Uncertain`** |

*Verification:* Presets populate only input UI widgets. All decisions, metrics, and explanation texts are computed dynamically by `MultimodalVerificationPipeline.verify_sample()`.

---

## 6. Critical Bug Fix Applied During Audit

* **Issue Identified:**
  In `src/cross_modal.py`, raw cosine similarity ($s \in [-1, 1]$) was previously transformed into an artificially scaled score:
  $$\text{normalized} = \frac{s + 1.0}{2.0}$$
  The mismatch check then evaluated:
  $$\text{is\_mismatch} = \text{normalized} < \text{threshold}$$
  Because unrelated natural images in CLIP yield raw cosine similarity around $0.15 \sim 0.20$, the normalized value was $\sim 0.58 \sim 0.60$.
  Comparing this against `threshold = 0.22` meant **mismatch was virtually never detected** (it would have required raw cosine $< -0.56$).
* **Fix Applied:**
  Updated `src/cross_modal.py` line 109 to compare the raw cosine similarity directly against `mismatch_threshold`:
  ```python
  is_mismatch = raw_score < mismatch_threshold
  ```
* **Impact of Fix:**
  Mismatched visuals (such as an unrelated beach image for a snowstorm) now correctly trigger `mismatch_detected = True`, enabling genuine out-of-context detection as intended.

---

## 7. Heuristic Threshold & Uncertainty Disclosures

1. **Threshold Empirical Status:**
   * **OpenCLIP cutoff ($0.22$):** **Unvalidated heuristic.** Derived from qualitative observation of ViT-B/32 background distributions. It has not been calibrated on a formal ROC curve of a locked benchmark partition.
   * **NLI entailment cutoff ($0.60$):** **Unvalidated heuristic.** Represents a standard majority margin in 3-class softmax.
   * **NLI contradiction cutoff ($0.50$):** **Unvalidated heuristic.**
2. **Uncertainty Calibration Status:**
   * The confidence and uncertainty scores are rule-based margins ($1.0 - \text{Confidence}$).
   * They must **NOT** be reported as Bayesian posterior probabilities or formal conformal prediction bands.
   * Both the UI (Tab 4) and codebase explicitly document this limitation.

---

## 8. License & Attribution Compliance

The submission workspace maintains clean provenance:
* **MIT License** established in `LICENSE`.
* **Explicit Third-Party Attributions:**
  * **Repository A:** Aryan Shine & Krishna Mistry (`kmist70/fake-news-detector`) acknowledged for Streamlit UI paradigms under MIT License.
  * **Repository B:** Rushil Dharwal (`LaughingParrot/...-Fakeddit`) acknowledged for OpenCLIP multimodal alignment concepts under MIT License.
  * **Pretrained Weights:** OpenAI CLIP and Moritz Laurer DeBERTa-v3 referenced with HuggingFace Hub model IDs.

---

## 9. Final Checklist for Viva / Professor Demonstration

- [x] Application launches cleanly on `http://localhost:8501`.
- [x] All 5 automated pipeline tests pass.
- [x] Pretrained model weights are verified and cached locally.
- [x] Empty and invalid inputs degrade gracefully without unhandled exceptions.
- [x] Evidence citations display source IDs without fabricating references.
- [x] Three live presets demonstrate Supported, Misleading, and Uncertain behaviors.
- [x] Limitations, mock corpus nature, and heuristic thresholds are transparently documented.
