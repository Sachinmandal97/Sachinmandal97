# Installation, Configuration & Execution Guide

## VERITAS-MM 2.0: Step-by-Step Deployment on Windows & Linux

---

## 1. System Requirements & Prerequisites

### Hardware Requirements
* **Processor:** Modern multi-core x86_64 CPU (Intel Core i5/i7/i9 8th Gen+ or AMD Ryzen 5/7/9). Multi-threading is automatically managed.
* **Memory (RAM):**
  - Minimum: $8\text{ GB}$ RAM.
  - Recommended: $16\text{ GB}$ RAM (optimal for running Streamlit, DeBERTa-v3, and OpenCLIP simultaneously in-memory).
* **Disk Space:** Minimum $10\text{ GB}$ available SSD storage (accommodates Python environment, PyTorch wheels, transformer model caches, and SQLite stores).
* **GPU Acceleration:** **Optional.** The complete system is profiled and optimized for CPU inference. If an NVIDIA GPU with CUDA 11.8/12.1 is present, PyTorch will automatically leverage GPU tensor cores.

### Software Prerequisites
* **Operating System:** Windows 10 / Windows 11 (tested on Windows 11 PowerShell 7 / Windows PowerShell 5.1) or Linux (Ubuntu 20.04/22.04 LTS).
* **Python Runtime:** Python `3.10.x` or `3.11.x` (64-bit).
* **Git:** Version control client for repository management.

---

## 2. Windows Quickstart Installation (PowerShell)

### Step 1: Open PowerShell and Clone Repository
Open Windows PowerShell (Run as Administrator if necessary to configure script execution policy) and navigate to your project directory:

```powershell
cd C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission
```

If PowerShell restricts virtual environment script activation, enable local script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### Step 2: Create and Activate Python Virtual Environment
Isolate project dependencies by creating a dedicated virtual environment:

```powershell
# Create virtual environment named 'venv'
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

*(Your terminal prompt should now display `(venv)` preceding the current path).*

---

### Step 3: Upgrade Packaging Utilities
Ensure pip and setuptools are up to date:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

---

### Step 4: Install PyTorch Runtime
Install PyTorch optimized for your hardware setup:

* **For CPU-Only Systems (Default / Standard Laptop):**
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

* **For Systems with NVIDIA CUDA GPU (e.g., CUDA 12.1):**
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

---

### Step 5: Install Project Dependencies
Install all required libraries specified in `requirements.txt`:

```powershell
pip install -r requirements.txt
```

---

### Step 6: Download spaCy Linguistic Models
Pillar 1 relies on spaCy for dependency-based claim decomposition and named entity extraction. Download the required lightweight English model:

```powershell
python -m spacy download en_core_web_sm
```

---

### Step 7: Initialize SQLite Intelligence Database
Initialize the database tables (`articles`, `verifications`, `feed_registry`, `system_locks`) and seed verified publisher feeds:

```powershell
python -c "from src.database import init_db; init_db(); print('SQLite Database Successfully Initialized!')"
```

---

## 3. Running the Application

### Launch Streamlit Multi-Tab Web Application
Execute the following command inside your activated virtual environment:

```powershell
streamlit run app.py
```

Upon successful startup, the console will output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open `http://localhost:8501` in your browser (Google Chrome, Microsoft Edge, or Mozilla Firefox recommended).

---

## 4. Running the Automated Test Suite

VERITAS-MM 2.0 maintains a comprehensive suite of unit, integration, and regression tests ensuring zero regressions across all pipeline components.

### Run All 122 Tests
```powershell
pytest -v tests/
```

### Run Benchmark Verification Telemetry Script
Verify all empirical numbers, confidence intervals, and confusion matrices against source JSON outputs:

```powershell
python tests/verify_all_metrics.py
```

### Run Individual Test Suites
* **Pillar 1 RAG & NLI Stance Tests:**
  ```powershell
  pytest tests/test_universal_verification.py -v
  ```
* **Pillar 2 OpenCLIP Alignment & Selective Classification Tests:**
  ```powershell
  pytest tests/test_benchmark_infrastructure.py -v
  ```
* **Pillar 3 Gated Fusion Adapter Integration Tests:**
  ```powershell
  pytest tests/test_rhetoric_integration.py -v
  ```
* **News Intelligence & SQLite Ingestion Tests:**
  ```powershell
  pytest tests/test_news_intelligence.py -v
  ```

---

## 5. Troubleshooting & Frequently Asked Questions (FAQ)

### Q1: `Activate.ps1 cannot be loaded because running scripts is disabled`
**Cause:** Windows PowerShell execution policy default restricts unsigned scripts.  
**Fix:** Run the following command in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q2: `sqlite3.OperationalError: database is locked`
**Cause:** An unclosed background process or terminated instance held an unreleased lock.  
**Fix:** 
1. Check if another instance of `streamlit run app.py` or a background Python scraper is running in another terminal window.
2. Ensure WAL mode is active. If orphaned lock files exist in `data/`, remove temporary shm files:
   ```powershell
   Remove-Item data\news_intelligence.db-shm -ErrorAction SilentlyContinue
   ```

### Q3: `Port 8501 is already in use`
**Cause:** Another local service or previous Streamlit process is occupying port 8501.  
**Fix:** Specify an alternative port:
```powershell
streamlit run app.py --server.port 8502
```

### Q4: Long startup latency on first claim verification
**Cause:** On initial execution, HuggingFace downloads and caches `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` and `openai/clip-vit-base-patch32` into `~/.cache/huggingface/hub`.  
**Note:** This download occurs **only once**. Subsequent runs load directly from the local disk cache into memory in $< 2.5\text{s}$.
