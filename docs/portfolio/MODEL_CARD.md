# Machine Learning Model Card: VERITAS-MM 2.0

## Production Foundation Encoders & Custom Multimodal Fusion Adapter

---

## 1. Model Details

### Overview
VERITAS-MM 2.0 incorporates a hybrid architecture consisting of two frozen foundation transformer backbones and a specialized, lightweight parameter-efficient fusion adapter:

```
┌─────────────────────────────────┬──────────────────────────────────┬─────────────────┬──────────────────────────┐
│ MODEL COMPONENT                 │ BASE ARCHITECTURE                │ PARAMETER COUNT │ INFERENCE STATUS         │
├─────────────────────────────────┼──────────────────────────────────┼─────────────────┼──────────────────────────┤
│ Pillar 1: Directional NLI Model │ DeBERTa-v3-large (Moritz Laurer) │ 435 Million     │ Frozen In-Memory Base    │
├─────────────────────────────────┼──────────────────────────────────┼─────────────────┼──────────────────────────┤
│ Pillar 2: Vision-Language Model │ OpenCLIP ViT-B/32 (OpenAI)       │ 151 Million     │ Frozen In-Memory Base    │
├─────────────────────────────────┼──────────────────────────────────┼─────────────────┼──────────────────────────┤
│ Pillar 3: Gated Fusion Adapter  │ Dual-Projection + Sigmoid Gate   │ 1.58 Million    │ Custom Trained (Option 3)│
└─────────────────────────────────┴──────────────────────────────────┴─────────────────┴──────────────────────────┘
```

* **Organization/Developer:** Candidate, M.Sc. Data Science, Somaiya Vidyavihar University.
* **Model Release Date:** November 2026.
* **Model Format:** PyTorch state dict (`.pt`), TorchScript / NumPy cached embeddings.
* **License:** MIT License for custom adapter code; respective open-source licenses for pre-trained backbones (Apache 2.0 / MIT).

---

## 2. Intended Use & Applications

### Primary Intended Uses
1. **Investigative Journalism Assistance:** Assisting human fact-checkers and newsrooms in evaluating breaking news claims against authoritative external evidence.
2. **Contextual Misattribution Screening:** Identifying unmanipulated stock imagery or historical news photos repurposed with deceptive captions ("cheapfakes").
3. **Rhetorical Manipulation Detection:** Scoring headlines and lead visuals for sensationalist framing, clickbait phrasing, or emotional manipulation.
4. **Real-Time News Monitoring:** Ingesting and indexing national and global news feeds to maintain an auditable database of verified events.

### Prohibited & Out-of-Scope Uses
* **Fully Autonomous Censorship:** The system must **never** be deployed as an unmonitored censorship tool. Automated decisions must remain subject to human editorial review.
* **Pixel-Level Generative Image Detection:** The model does **not** evaluate pixel compression artifacts, GAN artifacts, or diffusion generation fingerprints. A synthetic image from Midjourney will not be flagged as a deepfake unless its semantic content contradicts the caption.
* **Definitive Legal Fact Determination:** System outputs represent probabilistic epistemic adjudications based on retrieved public evidence; they do not constitute legal proof of truth or falsehood.

---

## 3. Training Data & Preprocessing

### Custom Gated Adapter Training
* **Dataset:** VLDBench Multimodal Manipulation Benchmark (31,339 records).
* **Partition Protocol:** Protocol A (Group-Aware Random Split, zero syndication wire overlap across train/val/test).
* **Sanitization Pass:** Mandatory anchor-bounded regular expression purging of trailing publisher attributions (e.g., `"- Reuters"`, `"| BBC"`) to prevent metadata memorization.
* **Optimization Setup:**
  - Optimizer: AdamW ($\text{lr} = 1 \times 10^{-4}$, weight decay $= 1 \times 10^{-2}$).
  - Loss Function: Weighted Cross-Entropy Loss to account for minor class imbalance.
  - Regularization: Dropout ($p = 0.3$), LayerNorm, early stopping with patience $= 5$ based strictly on validation Macro-F1.

---

## 4. Evaluation Metrics & Performance Summary

### Pillar 2: OpenCLIP ViT-B/32 on NewsCLIPpings Task A Test ($N = 7,264$)
* **Full-Test Baseline ($\tau^* = 0.28, \Phi = 100\%$):**
  - Accuracy: **$63.53\%$** (95% Wilson CI: $[62.42\%, 64.63\%]$)
  - Balanced Accuracy: $63.53\%$
  - Macro-F1: $62.75\%$
  - ROC-AUC: $0.7197$
* **Accepted-Subset under Chow's Reject Option ($\Delta^* = 0.07, \Phi = 10.85\%, N=788$):**
  - Accepted Accuracy: **$87.44\%$** (95% Wilson CI: $[84.94\%, 89.57\%]$)
  - Accepted Macro-F1: $87.24\%$
  - Selective Risk: $12.56\%$
  - Deferred Cohort: $89.15\%$ ($6,476$ samples deferred for human review).

### Pillar 3: Gated Multimodal Fusion Adapter on VLDBench Protocol A Test ($N = 1,094$)
* **Test Accuracy:** **$64.72\%$**
* **Balanced Accuracy:** **$64.80\%$**
* **Test Macro-F1:** **$64.54\%$** (95% Group Bootstrap CI: $[61.73\%, 67.20\%]$)
* **Test ROC-AUC:** $0.7086$
* **Modality Salience Distribution:** $w_{\text{text}} = 38.2\%$, $w_{\text{vision}} = 61.8\%$

---

## 5. Ethical Considerations & Governance Safeguards

1. **Anti-Leakage Headline Sanitization:** Eliminating publisher brand tokens prevents the model from developing structural prejudices against independent or regional news organizations.
2. **Selective Classification for Fairness:** By declining to force classifications on borderline cases ($|s_{TI} - 0.28| < 0.07$), the system avoids generating false accusations against legitimate journalism in high-ambiguity contexts.
3. **Transparent Evidence Provenance:** Pillar 1 does not merely output a verdict; it renders full source titles, publishers, URLs, snippet quotes, and directional NLI probabilities for every evaluated proposition.
4. **Epistemic Modesty:** When external corroboration is missing, the system outputs `UNCERTAIN` rather than forcing a speculative truth verdict.
