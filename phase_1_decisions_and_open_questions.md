# Phase 1: Research Decisions & Open Questions
## MSc Research Project — Academic Alignment & Open Engineering Decisions

**Project Title:** Explainable Multimodal Vision-Language Framework for AI-Generated Fake News and Misinformation Detection  
**Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission`  
**Date:** October 13, 2026  
**Auditor / Candidate:** MSc Data Science Candidate / Autonomous Pair Programmer  
**Document Purpose:** Log of critical academic decisions, open questions, and supervisor sign-off items prior to experimental benchmarking.

---

## 1. Executive Summary

This document enumerates the core architectural, methodological, and experimental decisions that must be resolved prior to beginning Phase 2 (Benchmark Evaluation Scaffolding). In an academic MSc research project, making silent engineering assumptions about dataset labels, evaluation protocols, or scope boundaries can jeopardize academic defense. 

The items below are organized into:
1. **Scope & Title Alignment Decisions (Critical Supervisor Sign-Off)**
2. **Methodological & Modeling Decisions**
3. **Dataset Selection & Label Harmonization Decisions**
4. **Explainability & Evaluation Design Decisions**
5. **Infrastructure & Dependency Decisions**

---

## 2. Scope & Title Alignment Decisions (Supervisor Alignment Required)

### Decision 1: Project Scope Regarding "AI-Generated" Imagery vs. "Out-of-Context" Imagery
* **Current Implementation Reality:**  
  The current codebase contains **zero** synthetic image artifact detectors (no spatial CNNs, frequency spectrum classifiers, or diffusion artifact forensic heads). It implements **semantic image-text alignment** via OpenCLIP, which detects *cheapfakes* (real images repurposed with false or out-of-context headlines).
* **Open Academic Question:**  
  Should the research project:
  - **Option A (Recommend Scope Realignment):** Explicitly clarify in the dissertation that "AI-Generated Fake News" is evaluated at the *multimodal assertion level* (detecting AI-generated text claims paired with misleading imagery via evidence retrieval and semantic consistency), rather than pixel-level deepfake forensics?
  - **Option B (Title Adjustment):** Recommend to the MSc supervisor adjusting the project title slightly to:  
    *Explainable Multimodal Vision-Language Framework for Evidence-Augmented and Out-of-Context Misinformation Detection*?
  - **Option C (Feature Extension in Phase 3):** Implement a dedicated pretrained synthetic image classifier (e.g. Wang et al., CNNDetection or Dire) as an additional visual branch before final thesis submission?
* **Recommendation:** Pursue **Option A** for dissertation framing immediately, and evaluate the feasibility of **Option C** in Phase 3 if time and compute permit.

---

## 3. Methodological & Modeling Decisions

### Decision 2: Decision Fusion Paradigm — Rule-Based Logic vs. Learned Head
* **Current Implementation Reality:**  
  `fuse_multimodal_decision()` uses an explicit deterministic rule table based on contradiction precedence, partial support, and visual mismatch flags.
* **Trade-off Analysis:**
  - *Rule-Based Multi-Branch Fusion (Current):* Highly interpretable, transparent, guarantees strict precedence (contradictions always flag falsehood), zero data leakage, runs zero-shot without training.
  - *Learned Classification Head (Alternative):* Train an MLP, Gradient Boosted Tree, or Logistic Regression layer over $[s_{TI}, p_{\text{ent}}, p_{\text{neu}}, p_{\text{contra}}, r_{\text{rel}}]$ using benchmark training splits. May achieve higher statistical F1 on a specific dataset, but sacrifices deterministic transparent rules and risks overfitting.
* **Open Decision:**  
  Retain the rule-based fusion as the primary explainable framework, and evaluate the learned classifier as an **ablation baseline** in the dissertation experiments.

### Decision 3: Uncertainty Formulation — Heuristic Margins vs. Formal Conformal Prediction
* **Current Implementation Reality:**  
  `compute_decision_confidence_and_uncertainty()` blends OpenCLIP cutoff margins and DeBERTa NLI probability margins linearly:
  $$c = 0.50 + 0.25 \cdot \Delta_{\text{cm}} + 0.25 \cdot \Delta_{\text{nli}}, \quad u = 1.0 - c$$
* **Trade-off Analysis:**
  - The current equation is a clean heuristic proxy, but does not provide statistically valid coverage guarantees.
  - Implementing **Split Conformal Prediction** on a calibration split would allow the system to output prediction sets with formal $1 - \alpha$ statistical coverage (e.g. "90% confidence that the true label is within $\{ \text{Supported}, \text{Partially Supported} \}$").
* **Open Decision:**  
  Document the current margin equation as the baseline confidence index for the prototype, and implement formal Conformal Prediction as an academic enhancement during Phase 2/3.

### Decision 4: Live Web Retrieval vs. Frozen Dataset Document Grounding
* **Current Implementation Reality:**  
  The prototype features live web retrieval (Google News RSS and Wikipedia API).
* **The Benchmark Conflict:**  
  Benchmark datasets like **Factify 2** provide paired ground-truth documents (Claim + Document). Querying the live internet for a claim created in 2022 might retrieve 2026 news articles discussing retrospective events, introducing severe temporal distribution shift and potential evaluation leakage.
* **Open Decision:**  
  For benchmark evaluation on Factify 2, the evaluation harness must evaluate the pipeline using **the dataset's provided reference document** as the candidate evidence (testing NLI, atomic verification, and decision fusion directly). Live web retrieval will be evaluated in a separate **live generalization / real-world case study chapter**.

---

## 4. Dataset Selection & Label Harmonization Decisions

### Table 4.1: Dataset Suitability & Label Mapping Decisions

| Dataset | Research Question Addressed | Native Label Schema | Proposed Mapping to Framework Verdicts | Critical Decision / Open Issue |
| :--- | :--- | :--- | :--- | :--- |
| **NewsCLIPpings** | RQ2 (Cross-Modal Alignment) | Binary:<br>• `pristine`<br>• `falsified` | • `pristine` $\to \text{Consistent } (m_{TI} = 0)$<br>• `falsified` $\to \text{Mismatch } (m_{TI} = 1)$ | Text assertions in NewsCLIPpings are real captions; only the image pairing is falsified. Does not evaluate external retrieval. |
| **Factify 2** | RQ1, RQ3, RQ4 (Full Multimodal Pipeline) | 5 Classes:<br>• `Support_Multimodal`<br>• `Support_Text`<br>• `Insufficient_Multimodal`<br>• `Refute`<br>• `Neutral` | • `Support_Multimodal` $\to \text{SUPPORTED}$<br>• `Support_Text` $\to \text{PARTIALLY_SUPPORTED}$<br>• `Refute` $\to \text{MISLEADING}$<br>• `Insufficient_Multimodal` $\to \text{UNCERTAIN}$<br>• `Neutral` $\to \text{UNCERTAIN}$ | Direct alignment with our 5-state epistemic schema. Schema validator already verified in [`src/data_loader.py`](file:///C:/Users/94in/.gemini/antigravity/scratch/multimodal-misinfo-submission/src/data_loader.py#L156). |
| **MMFakeBench** | Cross-Domain Generalization | Multi-faceted manipulation categories | Requires schema inspection upon download | Confirm dataset public availability and file size before committing. |

* **Open Question for Supervisor:** Does the assessment panel require evaluation on both NewsCLIPpings (cheapfakes) and Factify 2 (fact-checking), or is a focused, deep evaluation on Factify 2 preferred?
* **Recommendation:** Evaluate **NewsCLIPpings** for visual-textual alignment ablation (RQ2) and **Factify 2** for the complete evidence-augmented verification pipeline (RQ1, RQ3, RQ4).

---

## 5. Explainability & Evaluation Design Decisions

### Decision 5: Explainability Evaluation Methodology
* **Academic Problem:** Proving that an explanation is "good" cannot rely solely on qualitative Streamlit screenshots.
* **Proposed Evaluation Framework:**
  1. **Faithfulness (Fidelity Testing):** Run ablation experiments where the cited evidence passage is masked or replaced with random text. If the model's verdict does not change, the explanation is unfaithful.
  2. **Citation Accuracy:** On Factify 2, evaluate whether the evidence passage cited in the explanation matches the ground-truth annotated document (Precision/Recall).
  3. **Human Trust User Study (Optional):** Conduct a small survey (10–15 participants) scoring 20 verification reports on a 5-point Likert scale (Clarity, Trust, Evidence Grounding, Decision Utility).

---

## 6. Infrastructure & Dependency Decisions

### Decision 6: OCR Engine Availability on Host Environment
* **Current Status:** `easyocr` is uninstalled in the host Windows environment; OCR gracefully falls back to empty string without failing.
* **Decision:**  
  Because installing heavy PyTorch-dependent OCR binaries on Windows can cause DLL conflicts, keep OCR as an optional graceful fallback during local evaluation. Document this explicitly as an environmental configuration rather than a code defect.

### Decision 7: Google News RSS JavaScript Redirect Resolution
* **Current Status:**  
  Google News RSS uses encrypted client-side redirect tokens (`CBMi...`). Simple HTTP header resolution without a browser runtime frequently fails to resolve the final publisher URL, keeping live articles in `snippet_only` mode.
* **Decision:**  
  Accept `snippet_only` mode for Google News RSS live search as an authentic limitation of headless web scraping. Highlight that Wikipedia provides full-text articles via API without wrapper barriers.

---

## 7. Prioritized Supervisor Discussion Agenda

Prior to commencing Phase 2 benchmarking, review the following five items with the academic supervisor:

1. **Title & Task Alignment:** Confirm acceptance of evaluating "AI-Generated Fake News" via multimodal assertion verification and out-of-context image alignment rather than pixel-level deepfake forensics.
2. **Benchmark Selection:** Confirm the dual-benchmark plan: **NewsCLIPpings** for visual alignment and **Factify 2** for evidence-based verification.
3. **Evaluation Protocol on Factify 2:** Agree on using the paired reference documents for controlled fact-checking evaluation to eliminate live-web temporal distribution shift.
4. **Sample Size for CPU Benchmark Execution:** Propose evaluating a statistically representative subset (e.g. 100–200 validated test samples per benchmark) given local CPU runtime constraints (~5–15 seconds per multimodal sample).
5. **Ablation Study Matrix:** Confirm that the 4 planned ablation studies (No Decomposition, No Relevance Filter, Text-Only, No Abstention) satisfy the dissertation's empirical rigor requirements.
