# Phase 2C — Genuine Benchmark Data Acquisition and Verification Report

**Project Title:** Explainable Multimodal Vision-Language Framework for AI-Generated Fake News and Misinformation Detection  
**Module:** MSc Data Science Research Project  
**Date:** October 13, 2026  
**Status:** Completed (NewsCLIPpings Verified & Adapter-Ready; Factify 2 Blocked on Registration Gate)  
**Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission`  
**Protected Baseline Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-verifier` (Untouched)  

---

## 1. Executive Summary & Audit Scope

Following the Phase 2C Stage A Preflight Audit—which conclusively determined that all prior Phase 2B pilot experiments were executed against 5-sample offline mock fixtures rather than authentic benchmark data—this phase executed the data acquisition, forensic verification, and adapter compatibility tasks required to ground the research evaluation in authentic benchmark records.

### Strict Scope Enforcement
In strict compliance with project directives:
1. **Zero Benchmark Evaluations or Ablations:** No benchmark inference, evaluation pipelines, ablation configurations, or predictive accuracy / F1 / ROC-AUC metrics were run or calculated in this phase.
2. **Zero Synthetic Substitutions:** Offline mock fixtures were not substituted for genuine benchmark evaluations.
3. **Bandwidth & Storage Optimization:** The massive VisualNews image corpus (~100 GB) was intentionally avoided by acquiring the official, authoritative precomputed OpenCLIP embeddings released directly by the benchmark authors.
4. **Ethical Access & Compliance:** No access gates, authentication walls, or licensing agreements were bypassed. Factify 2's registration requirement has been transparently documented with step-by-step user instructions.

---

## 2. Official Sources Consulted

Two primary multimodal misinformation benchmarks were investigated:

### 2.1 NewsCLIPpings (Luo et al., EMNLP 2021)
* **Citation:** Grace Luo, Trevor Darrell, Anna Rohrbach. *"NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media."* Findings of the Association for Computational Linguistics: EMNLP 2021, pp. 2701–2717.
* **Official Repository:** [`https://huggingface.co/g-luo/news-clippings`](https://huggingface.co/g-luo/news-clippings)
* **GitHub Repository:** [`https://github.com/grace-luo/news-clippings`](https://github.com/grace-luo/news-clippings)
* **Licensing:** Apache-2.0 (Public, commercial/academic use permitted, completely ungated).
* **Access Type:** Direct public programmatic access via HTTPS.
* **Corpus Scope:** Automatic generation of out-of-context multimodal media (cheapfakes) derived from the VisualNews corpus across four adversarial mismatch generation algorithms.

### 2.2 Factify 2 (De-Factify @ AAAI 2023)
* **Citation:** Shreyash Mishra et al. *"Factify 2: Multimodal Fake News and Satire Detection with Explainability."* CEUR Workshop Proceedings, Vol-3381 (De-Factify: 2nd Workshop on Multimodal Fact-Checking and Hate Speech Detection, AAAI 2023).
* **Official Portal:** [`https://defactify.com/2023/factify.html`](https://defactify.com/2023/factify.html)
* **GitHub Repository:** [`https://github.com/surya1701/Factify-2.0`](https://github.com/surya1701/Factify-2.0)
* **Hosting Platform:** CodaLab Competitions ([`https://codalab.lisn.upsaclay.fr/competitions/8275`](https://codalab.lisn.upsaclay.fr/competitions/8275))
* **Access Type:** **Account-Gated & Form-Gated**. Requires user account creation on CodaLab and explicit completion of the official De-Factify registration forms.

---

## 3. Benchmark Data Inventory, File Integrity & Checksums

All acquired files were placed in `data/raw/newsclippings/` within the submission workspace. Cryptographic SHA-256 hashes and file byte sizes were computed to establish provenance and auditability.

| File Description | Relative Workspace Path | File Size (Bytes) | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **NewsCLIPpings Val Split** | `data/raw/newsclippings/merged_balanced/val.json` | 762,083 bytes | `892AB89F9DD94302CAADFEEA3CDE43C24499E22F34A3220158B0589FF767FE52` |
| **NewsCLIPpings Test Split** | `data/raw/newsclippings/merged_balanced/test.json` | 788,016 bytes | `616E06AF39FD8AFA88D1920E455E4D6CA69E013C9B2FA4DADDA593FB0E813AEA` |
| **NewsCLIPpings Metadata Val** | `data/raw/newsclippings/metadata/val.json` | 37,288,368 bytes | `D703C8836B0A6BFDDDD2F5967D4793CD003403573890AFD7282C47B8396D7D27` |
| **CLIP Text Embeddings Val** | `data/raw/newsclippings/embeddings/clip_text_embeddings/clip_text_embeddings_val.pkl` | 86,261,352 bytes | `3B345116BFC30D5D7BC3179E4D17B41853E7DE0776F6A1F80B1A9BE0A023124F` |
| **CLIP Image Embeddings Val** | `data/raw/newsclippings/embeddings/clip_image_embeddings/clip_image_embeddings_val.pkl` | 105,641,474 bytes | `70B9C3EFEAC45D7A5A8A73A9025CDB62ABE40FB6D6809B9FC842BC299F4FD669` |

**Total Acquired Volume:** 230,741,293 bytes (~220.0 MB).  
All checksums have been independently verified against the HuggingFace repository commit history.

---

## 4. Data Split Statistics & Class Distribution Verification

Forensic inspection of `merged_balanced/val.json` and its associated metadata confirms complete alignment with the official benchmark specification:

### 4.1 Split and Class Distribution
* **Total Validation Samples:** 7,024
* **Pristine Pairs (Label 0 / Ground Truth True):** 3,512 (50.00%)
* **Falsified Pairs (Label 1 / Ground Truth Misleading):** 3,512 (50.00%)
* **Class Imbalance Ratio:** Exact 1:1 balance ($N_{\text{pristine}} = N_{\text{falsified}}$).

### 4.2 Adversarial Generator Stratification
The validation split contains exactly 1,756 samples (25.0%) across each of the four official cheapfake generation methods documented by Luo et al. (2021):

| Generator ID | Generator Name | Total Samples | Pristine Count | Falsified Count | Generation Mechanism |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | `semantics_clip_text_image` | 1,756 (25%) | 878 | 878 | Unconstrained visual-semantic CLIP nearest neighbor pairing |
| **1** | `semantics_clip_text_text` | 1,756 (25%) | 878 | 878 | Text-to-text semantic retrieval via CLIP |
| **2** | `scene_resnet_place` | 1,756 (25%) | 878 | 878 | ResNet Places365 scene feature matching |
| **3** | `person_sbert_text_text` | 1,756 (25%) | 878 | 878 | Named person entity overlap + SBERT similarity |

### 4.3 Feature Completeness & Embedding Integrity
* **Metadata Article Records:** 40,830 complete news items in `metadata/val.json` containing authentic headlines, original captions, image source paths, publication dates, and spaCy NER entities.
* **Text Embeddings:** 40,864 precomputed CLIP ViT-B/32 text vectors. Shape: `(512,)`, Dtype: `float32`, $L_2$-norm: $1.000000 \pm 10^{-6}$.
* **Image Embeddings:** 50,000 precomputed CLIP ViT-B/32 image vectors. Shape: `(512,)`, Dtype: `float32`, $L_2$-norm: $1.000000 \pm 10^{-6}$.
* **Missing Embedding Count:** **0** across all 7,024 validation samples (100% coverage).

---

## 5. Factify 2 Blocker & Manual Acquisition Procedure

### 5.1 Diagnosis: Access Gate Barrier
Unlike NewsCLIPpings, Factify 2 is hosted as part of the De-Factify workshop on CodaLab and requires affirmative credentialed registration:
* **CodaLab Competition URL:** `https://codalab.lisn.upsaclay.fr/competitions/8275#participate-get-data`
* **Official Google Form Registration:**
  - Track 1 Registration: `https://forms.gle/AQdhvGtU3YwknwnF8`
  - Participant Consent & Agreement: `https://forms.gle/L43vLWdYX3gGMTnV`

Under institutional research standards and AI agent safety guidelines, automated agents must not:
1. Bypass web authentication systems.
2. Fabricate user accounts, emails, or institutional affiliations.
3. Automatically submit legal terms-of-use consent agreements on behalf of a human researcher.

### 5.2 Step-by-Step Instructions for the User
To enable Task B and Task C evaluations on Factify 2, the user should execute the following manual steps:

1. **Register on CodaLab:**
   - Navigate to [`https://codalab.lisn.upsaclay.fr/competitions/8275`](https://codalab.lisn.upsaclay.fr/competitions/8275).
   - Sign in or create a student/researcher account using university credentials.
   - Click the **"Participate"** tab and accept the competition terms.
2. **Download the Validation Set:**
   - Under the "Get Data" section, download `val.json` or `val.jsonl` (or the equivalent TSV/CSV archive).
3. **Place Data in Workspace:**
   - Copy the acquired file to:  
     `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission\data\raw\factify2\val.json` (or `.jsonl` / `.tsv`).
4. **Automated Schema Verification:**
   - Once placed, the existing validator in `src/data_loader.py` (`validate_factify2_record` and `load_factify2_metadata`) will automatically validate the records without code modifications.

---

## 6. Adapter Compatibility & Ingestion Verification

### 6.1 Data Loader Enhancement
`src/data_loader.py:load_newsclippings()` was upgraded to natively support both the official benchmark structure and legacy fixtures:
* **Directory Resolution:** Dynamically checks both `data/raw/newsclippings` and `data/raw/news_clippings`.
* **Relational Joining:** Joins the relational `merged_balanced/val.json` entries (`id`, `image_id`, `falsified`, `source_dataset`) with rich news records in `metadata/val.json` (`caption`, `title`, `source`, `timestamp`, `entities`).
* **Precomputed Embedding Ingestion:** Directly injects author-provided 512-dimensional CLIP text and image embeddings into `sample.metadata`, bypassing the need for raw image rasterization.
* **Backwards Compatibility:** Preserves flat JSON loading and demo fallbacks without breaking existing unit tests.

### 6.2 Schema & Ingestion Unit Test
A new automated test, `test_newsclippings_genuine_data_loading_and_schema`, was added to `tests/test_benchmark_infrastructure.py`. It verifies that genuine samples produce valid `MultimodalSample` objects with non-empty text, correct binary labels, and matching 512-d embeddings.

**Test Execution Result:**
```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission
collected 1 item

tests/test_benchmark_infrastructure.py::test_newsclippings_genuine_data_loading_and_schema PASSED [100%]

======================= 1 passed in 3.50s =======================
```

### 6.3 Sample Record Inspection (First 5 Genuine Samples)
To confirm qualitative realism without running evaluation scoring, the first 5 records were inspected:

* **Sample 0 (`nc_val_92318_92318`):**
  - Claim: *"People walk in the parking lot of an Ikea store in Brooklyn"*
  - Ground Truth: `pristine` (Stratum: `pristine`, Falsified: `False`)
  - Source: Washington Post (Topic: `science_technology`, Date: `2016-02-03`)
  - Embeddings: Text `(512,)`, Image `(512,)` (Cosine sim: `0.2909`)
* **Sample 1 (`nc_val_92318_136816`):**
  - Claim: *"People walk in the parking lot of an Ikea store in Brooklyn"*
  - Ground Truth: `falsified` (Stratum: `falsified`, Falsified: `True`, Generator: `scene_resnet_place`)
  - Embeddings: Text `(512,)`, Image `(512,)` (Cosine sim: `0.2281`)
* **Sample 2 (`nc_val_224103_224103`):**
  - Claim: *"The Olympics are over but the Paralympics begin Friday in Rio"*
  - Ground Truth: `pristine` (Stratum: `pristine`, Falsified: `False`)
  - Embeddings: Text `(512,)`, Image `(512,)`
* **Sample 3 (`nc_val_224103_854936`):**
  - Claim: *"The Olympics are over but the Paralympics begin Friday in Rio"*
  - Ground Truth: `falsified` (Stratum: `falsified`, Falsified: `True`, Generator: `clip_text_text`)
  - Embeddings: Text `(512,)`, Image `(512,)`
* **Sample 4 (`nc_val_367276_367276`):**
  - Claim: *"The Brandenburg Gate stands illuminated during celebrations"*
  - Ground Truth: `pristine` (Stratum: `pristine`, Falsified: `False`)
  - Embeddings: Text `(512,)`, Image `(512,)`

---

## 7. Status & Formal Recommendation

| Benchmark Dataset | Evaluation Task | Genuine Data Status | Ingestion & Schema | Recommendation |
| :--- | :--- | :---: | :---: | :---: |
| **NewsCLIPpings** | Task A (Cheapfakes) | **Acquired & Verified** | **Passed** | **GO (Proceed to Controlled Benchmark Evaluation)** |
| **Factify 2** | Task B (NLI) & Task C (Multimodal) | **Blocked on Registration** | **Adapter Ready** | **PAUSED (Awaiting User Download)** |

### Next Steps for Phase 2C Stage B Execution:
1. **Execute Task A Evaluation:** Run controlled benchmark evaluation on genuine NewsCLIPpings (e.g., $N=100$ or $N=200$ stratified samples) using `evaluate_benchmark.py --dataset newsclippings --split val --limit 100`.
2. **Execute Controlled Ablations:** Run cross-modal threshold sensitivity and component ablations on authentic cheapfake samples.
3. **Pending Factify 2 Resolution:** If the user provides `val.json` for Factify 2, immediately execute Task B and Task C benchmarks. If Factify 2 remains unavailable, formalize NewsCLIPpings as the primary verified benchmark for the MSc submission.
