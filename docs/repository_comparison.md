# Empirical Repository Audit & Comparative Feasibility Assessment

**Project Title:** Explainable Evidence-Augmented Multimodal Misinformation Verification  
**Degree Programme:** MSc Data Science Final-Year Research Project  
**Author:** Candidate (MSc Data Science)  
**Date of Audit:** October 2026  
**Repositories Inspected:**
1. **Repository A:** [`kmist70/fake-news-detector`](https://github.com/kmist70/fake-news-detector)
2. **Repository B:** [`LaughingParrot/CLIP-Multimodal-Fake-News-Detection-with-InfoNCE-on-Fakeddit`](https://github.com/LaughingParrot/CLIP-Multimodal-Fake-News-Detection-with-InfoNCE-on-Fakeddit)

---

## 1. Executive Summary & Core Verdict

| Dimension | Repository A (`kmist70`) | Repository B (`LaughingParrot`) | Our Submission Platform |
| :--- | :--- | :--- | :--- |
| **License** | MIT (Permissive reuse) | MIT (Permissive reuse) | MIT (Full third-party attribution) |
| **Checkpoints Included?** | ❌ **NONE** (Crashes on startup) | ❌ **NONE** (Untrained weight fallback) | ✅ **HuggingFace hub pretrained** |
| **Working UI?** | Streamlit (Text & Video only) | CustomTkinter (Desktop GUI only) | Streamlit (Interactive Web UI) |
| **Image Upload Analysis?** | ❌ No (Video frames only) | ✅ Yes (Single & batch image+text) | ✅ Yes (Image upload + preview + OCR) |
| **Evidence Retrieval / RAG?**| ❌ No (Generic classifier) | ❌ No (Generic classifier) | ✅ Yes (BM25 knowledge base retrieval) |
| **NLI / Fact-Checking?** | ❌ No | ❌ No | ✅ Yes (DeBERTa-v3 3-way NLI) |
| **Explainability** | Word cloud only | Cosine scalar score only | Full structured attribution report |
| **Uncertainty Calibration** | Rule-based < 70% flag | Rule-based < 0.65 flag | Tri-state with uncertainty warnings |
| **Windows Execution** | Fails out-of-the-box (missing `.pt`) | Difficult (requires Tkinter desktop) | Fully verified on Windows 11 CPU |

---

## 2. Deep Dive: Repository A (`kmist70/fake-news-detector`)

### 2.1 License and Permitted Reuse
* **License File:** MIT License (`Copyright (c) 2026 AryoShine`).
* **Permitted Reuse:** Full commercial and academic reuse permitted, provided the copyright notice and permission notice are preserved.

### 2.2 Local Execution on Windows
* **Status:** **Fails on initial startup.**
* **Technical Blocker:**
  * In `loaders.py` (lines 12, 21), the code unconditionally invokes:
    ```python
    ckpt = torch.load("model/text_classifier.pt", map_location=DEVICE)
    ckpt = torch.load("model/multimodal_classifier.pt", map_location=DEVICE)
    ```
  * Neither `model/text_classifier.pt` nor `model/multimodal_classifier.pt` is committed to the Git repository.
  * Executing `streamlit run app.py` terminates immediately with `FileNotFoundError: [Errno 2] No such file or directory: 'model/text_classifier.pt'`.
* **Hardware & Tooling Dependencies:**
  * Requires external `ffmpeg` binary on system PATH for Whisper audio transcription and video slicing.
  * Heavy GPU dependencies if Whisper and DistilBERT are loaded concurrently.

### 2.3 Modality Handling
* **Text Analysis:** Scrapes URLs via `newspaper3k` or accepts raw article text; embeds via DistilBERT (`distilbert-base-uncased`) with an untrained/missing MLP classification head.
* **Image Analysis:** **Not supported.** The repository has no image upload tab or single-image processing pipeline. It only slices 8 video frames from `.mp4` video files.

### 2.4 Evidence, Explainability & Uncertainty
* **Evidence Retrieval:** **Zero.** Does not perform external verification or knowledge retrieval.
* **Explainability:** Generates a Matplotlib word cloud of frequent unigrams. This is an exploratory text visualization, **not an explanation** of model reasoning or factual verification.
* **Uncertainty:** Hardcoded confidence rule: if probability < 70%, outputs `⚠️ UNCERTAIN`.

---

## 3. Deep Dive: Repository B (`LaughingParrot/...-Fakeddit`)

### 3.1 License and Permitted Reuse
* **License File:** MIT License (`Copyright (c) 2026 Rushil Dharwal`).
* **Permitted Reuse:** Full academic reuse permitted with attribution.

### 3.2 Local Execution on Windows & UI Framework
* **UI Technology:** Built using `customtkinter` (2,143 lines in `interface/app.py`) with `matplotlib` TkAgg backend.
* **Streamlit Compatibility:** **Zero.** Does not use Streamlit.
* **Desktop & OS Constraints:**
  * Requires a desktop GUI window manager (fails in headless environments, Docker containers, or web servers).
  * Relies on platform-dependent `tkinterdnd2` for drag-and-drop file events.

### 3.3 Checkpoint Availability & Architecture
* **Status:** Checkpoint `multimodal_model.pt` is **NOT committed** to the repository.
* **Architectural Innovation:**
  * Uses OpenCLIP (`ViT-B-32`) to encode news headlines and images into normalized 512-dim vectors.
  * Implements an NLI-style fusion layer:
    $$\text{interaction} = f_t \odot f_v, \quad \text{difference} = |f_t - f_v|$$
  * Combines $[f_t; f_v; z; s]$ where $s = \cos(f_t, f_v)$ is the explicit cosine alignment scalar.
* **Execution Behavior without Checkpoint:**
  * The loader in `interface/app.py` falls back to initializing untrained random weights:
    `status = "Models Ready (Warning: untrained weights)"`.
  * **Critical Academic Risk:** Running with untrained weights produces random, ungrounded classifications that violate scientific validity.

### 3.4 Evidence, Explainability & Uncertainty
* **Evidence Retrieval:** None. No retrieval from external corpora.
* **Explainability:** Displays scalar cosine similarity between image and headline embeddings.
* **Uncertainty:** Routes predictions with confidence < 0.65 to an "Expert Review" queue.

---

## 4. Synthesis & Architectural Foundation Decision

### Why Neither Repository Can Be Used Directly "As-Is"
1. **Repository A** cannot run without training a classifier on the 72k-row WELFake dataset first (missing `.pt` files), does not accept static news images, and relies on video processing with external `ffmpeg`.
2. **Repository B** does not use Streamlit (built in Tkinter), lacks trained classifier weights, and lacks evidence retrieval and RAG.

### The Optimal MSc Submission Architecture
To meet all urgent submission requirements for the MSc Data Science viva:
1. **UI Foundation (Inspired by Repository A):**
   * Adopt Streamlit as the web framework.
   * Implement a professional, dark-themed dashboard with tabs: **"Multimodal Verification"**, **"Evidence Knowledge Base"**, **"Methodology & How It Works"**, and **"Model Limitations & Viva Guide"**.
2. **Multimodal Alignment Foundation (Inspired by Repository B):**
   * Adopt the OpenCLIP vision-language joint embedding alignment and explicit cosine similarity computation ($s \in [-1, 1]$).
   * Evaluate visual-semantic consistency and detect out-of-context image reuse.
3. **Core Research Engine (From Our Tested Research Pipeline):**
   * Preprocessing & PIL image validation.
   * Optical Character Recognition (OCR) overlay extraction to detect text embedded inside images.
   * spaCy deterministic claim and named entity decomposition.
   * BM25Okapi lexical evidence retrieval over an explicit local knowledge base corpus (`corpus.jsonl`).
   * DeBERTa-v3 Natural Language Inference (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) for claim-evidence verification.
   * Tri-state decision adjudication (`Supported`, `Misleading`, `Uncertain`).
   * Structured attribution report and transparent uncertainty communication.
4. **Workspace Isolation:**
   * Built in dedicated submission workspace: `multimodal-misinfo-submission/`.
   * The original research repository (`multimodal-misinfo-verifier/`) remains **completely untouched**.
