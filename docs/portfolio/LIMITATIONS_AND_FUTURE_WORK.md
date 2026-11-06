# Known Limitations, Scientific Boundaries & Future Research Roadmap

## VERITAS-MM 2.0: Methodological Rigor & Failure Mode Analysis

---

## 1. Explicit Scientific Boundaries & Non-Claims

Academic and operational credibility requires rigorous disclosure of what an automated verification system can and **cannot** perform. VERITAS-MM 2.0 operates under four strict, non-negotiable scientific boundaries:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  EXPLICIT RESEARCH BOUNDARIES                                   │
├───────────────────────────────────┬─────────────────────────────────────────────────────────────┤
│ BOUNDARY / NON-CLAIM              │ OPERATIONAL IMPLICATION                                     │
├───────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1. No Pixel-Level Forensics       │ Does NOT detect generative diffusion artifacts, GAN faces,  │
│    (Cheapfakes != Deepfakes)      │ splicing boundaries, or Photoshop error-level anomalies.    │
├───────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2. Semantic Alignment != Truth    │ High cross-modal cosine similarity denotes visual relevance;│
│    (Concordance != Provenance)    │ it does NOT prove the photograph was taken at that event.   │
├───────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 3. Selective Accuracy != Full Acc │ 87.44% accuracy applies strictly to high-margin predictions │
│    (Chow's Reject Coverage Trade) │ at 10.85% coverage (N=788). Full-test baseline is 63.53%.   │
├───────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 4. Salience != Causal Proof       │ Modality gate weights (w_T=38.2%, w_I=61.8%) reflect channel│
│    (Activation != Attribution)    │ activation magnitudes; they are NOT counterfactual proofs.  │
└───────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 2. Deep Dive: Known Failure Modes & Vulnerabilities

### 2.1 The OpenCLIP Adversarial Nearest-Neighbor Blindspot (Generator 0)
* **The Vulnerability:** On the NewsCLIPpings benchmark, Generator 0 pairs false captions with images retrieved by finding nearest neighbors in CLIP's visual-text embedding space.
* **Empirical Impact:** On Generator 0, zero-shot OpenCLIP accuracy collapses to **$43.83\%$** (with falsified recall dropping to $9.58\%$, missing $821$ out of $908$ cheapfakes).
* **Root Cause:** Dual-encoder models (such as CLIP) project high-level semantic abstractions onto a unit hypersphere. When an adversarial attacker selects an authentic photo matching the topical entities of a false story (e.g., using a 2018 photo of a protest in Athens to depict a 2026 strike in Rome), CLIP detects high semantic overlap (`"protest"`, `"crowd"`, `"banners"`, `"police"`) and assigns an elevated cosine score ($s_{TI} \ge 0.28$).
* **System Safeguard:** This failure mode is precisely why VERITAS-MM 2.0 **decouples** verification into Tri-Pillars. Pillar 1 independently queries external knowledge bases for the historical entity coordinates of the claim, preventing the system from falsely verifying a claim merely because its photo "looks relevant."

---

### 2.2 The Selective Classification Coverage–Accuracy Tradeoff
* **The Tradeoff:** Under Chow's Reject Option with margin half-width $\Delta^* = 0.07$, the system achieves **$87.44\%$ accuracy** (95% Wilson CI: $[84.94\%, 89.57\%]$). However, this high precision requires deferring **$89.15\%$ ($6,476 / 7,264$) of test assertions** to subsequent human review.
* **Operational Implication:** Selective classification is not a magical accuracy amplifier; it is an **uncertainty management filter**. In high-stakes journalism and national security intelligence, automated systems must never make guesses on borderline assertions ($|s_{TI} - 0.28| < 0.07$). Automated triage is provided for the clear-cut $10.85\%$, while complex ambiguities are safely escalated to professional human fact-checkers.

---

### 2.3 Temporal Latency in Breaking News Verification
* **The Challenge:** Pillar 1 relies on multi-source web retrieval (Google News RSS and Wikipedia REST API).
* **Zero-Day Latency:** In developing crises or breaking news events occurring within the last $15$ to $30$ minutes, authoritative fact-checking organizations and mainstream newsrooms have not yet published corroborating or debunking articles.
* **Epistemic Resolution:** Rather than hallucinating a guess, VERITAS-MM 2.0's deterministic aggregation rule labels uncorroborated assertions strictly as **`UNCERTAIN`**, presenting the user with an explicit notice that no corroborating external evidence could be verified.

---

### 2.4 Modality Salience vs. Causal Attribution
* **The Distinction:** Pillar 3 outputs modality salience weights ($w_{\text{text}} = 38.2\%$, $w_{\text{vision}} = 61.8\%$) derived from the scalar mean of the Gated Multimodal Fusion Adapter's sigmoid gate vector $\mathbf{g} \in (0, 1)^{512}$.
* **Governance Safeguard:** These weights quantify the relative numerical contribution of text and image feature channels inside the adapter's projection manifold. They do **not** imply that the visual was the primary "cause" of the misinformation. The user interface explicitly labels these values as "Representation Salience Weights" to avoid misleading end users.

---

## 3. Future Research & Development Roadmap

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        VERITAS-MM FUTURE RESEARCH ROADMAP                              │
└────────────────────────────────────────────────────────────────────────────────────────┘

  Phase 1: Spatio-Temporal Entity Grounding
  ├── Geolocation estimation from visual landmarks (PlaNet / OSINT models)
  ├── Metadata & EXIF reverse-image chronological verification
  └── Satellite imagery cross-referencing for conflict event claims

  Phase 2: Multimodal Large Language Model (MLLM) Reasoning Engine
  ├── Integration of structured reasoning chains (Gemini 1.5 Pro / Claude 3.5 Sonnet)
  ├── Dual-system architecture: Fast feature-based triage + Deep MLLM verification
  └── Self-critique prompting for adversarial consistency checking

  Phase 3: Active Learning & Analyst Feedback Loop
  ├── Human-in-the-loop workflow for the 89.15% deferred selective queue
  ├── Analyst adjudication feedback updating lightweight adapter weights
  └── Domain-specific continuous learning for election and health monitoring

  Phase 4: Extended Video & Audio Forensics
  ├── Frame-level keyframe extraction for video misinformation analysis
  ├── Audio deepfake detection (synthetic voice cloning / speech synthesis)
  └── Temporal synchronization verification between audio tracks and video frames
```
