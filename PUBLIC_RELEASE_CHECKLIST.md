# Public GitHub Repository Pre-Flight & Release Checklist

## VERITAS-MM 2.0: Academic & Open-Source Publication Protocol

---

Before publishing the `multimodal-misinfo-submission` workspace to a public GitHub repository, complete the following pre-flight verification checklist.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PUBLIC RELEASE PRE-FLIGHT AUDIT                                 │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Credentials, Secrets & Security Scan
- [x] **No Hardcoded API Keys:** Inspected `configs/config.yaml`, `src/news_ingestion.py`, and `app.py`. System uses unauthenticated public RSS/Atom endpoints and the open GDELT DOC 2.0 API. Zero private tokens or secret keys are stored in source code.
- [x] **Environment Exclusions:** `.env`, `.streamlit/secrets.toml`, and local credential stores are explicitly blocked in `.gitignore`.
- [x] **Safe Error Logging:** Exceptions caught in `src/retrieval.py` and `src/pipeline.py` do not dump raw environmental variables or host paths.

---

### 2. Data Privacy & Local Database Protection
- [x] **Exclusion of Active News Database:** `data/news_intelligence.db`, `data/news_intelligence.db-wal`, and `data/news_intelligence.db-shm` are excluded via `.gitignore`. Users initialize an empty database via `init_db()`.
- [x] **Reference Evidence Corpus Sanitization:** `data/evidence_corpus.jsonl` contains public, encyclopedic verification snippets for offline unit testing; verified free of PII (Personally Identifiable Information).
- [x] **Cache Isolation:** `datasets/raw/`, `datasets/external/`, `*.parquet`, and `*.arrow` are excluded from version control to prevent repository bloat.

---

### 3. File Size & Large Binary Weight Management
- [x] **GitHub 100 MB Hard Limit Compliance:** Inspected all tracked files.
  - Custom trained adapter: `models/rhetoric_adapter_gated.pt` is **6.3 MB** (well below GitHub's 50 MB warning threshold and 100 MB hard reject limit).
  - High-res figures in `docs/portfolio/figures/`: All PNGs are $< 1.5\text{ MB}$.
  - Foundation models: DeBERTa-v3 ($435\text{M}$) and OpenCLIP ($151\text{M}$) are downloaded dynamically into `~/.cache/huggingface/` on initial run, keeping the cloned repository lightweight ($< 25\text{ MB}$).

---

### 4. Code Integrity & Test Verification
- [x] **Automated Regression Suite:** All 122 tests pass cleanly across unit, integration, and UI modules:
  ```powershell
  pytest -v tests/
  ```
- [x] **Independent Metric Assertion:** `python tests/verify_all_metrics.py` confirms exact concordance with source JSON telemetry.
- [x] **Path Portability:** All file path resolutions in `src/` use relative `pathlib.Path` structures rooted at project directories rather than hardcoded Windows drive letters (`C:\Users\...`).

---

### 5. Scientific Accuracy & Epistemic Non-Claims
- [x] **Selective Classification Clarification:** Everywhere the $87.44\%$ accuracy metric is cited, it is explicitly accompanied by:
  - The coverage figure: **$10.85\%$ ($N=788$)**.
  - The $95\%$ Wilson Confidence Interval: **$[84.94\%, 89.57\%]$**.
  - The full-test baseline accuracy: **$63.53\%$ ($N=7,264$)**.
- [x] **Explicit Non-Claim Disclosure:** Documentation explicitly warns that the system evaluates semantic cheapfakes and rhetorical manipulation; it does **not** evaluate pixel-level generative synthesis (Midjourney / Stable Diffusion).
- [x] **Modality Salience Distinction:** Gating outputs are labeled strictly as "Modality Salience Weights" rather than causal explanations.

---

### 6. Recommended Git Initialization & Push Commands

To initialize and publish the repository to your public GitHub account:

```powershell
# 1. Navigate to the project directory
cd C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission

# 2. Initialize git repository (if not already initialized)
git init

# 3. Check git status to ensure .gitignore is properly masking DBs and caches
git status

# 4. Stage tracked files
git add .

# 5. Commit with professional academic release message
git commit -m "feat: initial public release of VERITAS-MM 2.0 multimodal verification framework"

# 6. Set main branch
git branch -M main

# 7. Add remote GitHub origin (replace with your repository URL)
git remote add origin https://github.com/<your-username>/veritas-mm-2.0.git

# 8. Push to GitHub
git push -u origin main
```
