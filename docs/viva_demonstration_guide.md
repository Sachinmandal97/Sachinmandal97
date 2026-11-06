# MSc Viva Demonstration & Defense Guide

**Project Title:** Explainable Evidence-Augmented Multimodal Misinformation Verification  
**Target Audience:** MSc Data Science Examination Committee & Supervisor  
**Platform:** Streamlit Web Application (`app.py`)  
**Workspace:** `C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission`

---

## 1. 60-Second Opening Pitch for Your Professor

> *"Good morning, Professor. My MSc project addresses the critical limitation of traditional automated fact-checking: end-to-end classifiers act as black-box oracles that fail to ground their decisions in verifiable evidence or explain their reasoning.*
> 
> *Our framework introduces an **Explainable Evidence-Augmented Multimodal Pipeline**. When given a news claim and an image, the system:*
> 1. *Analyzes visual-semantic alignment using OpenCLIP to detect out-of-context image reuse (cheapfakes).*
> 2. *Decomposes complex claims into atomic propositions using spaCy.*
> 3. *Retrieves corroborating or refuting evidence from a verified knowledge base via BM25 retrieval.*
> 4. *Evaluates claim-evidence entailment using a DeBERTa-v3 cross-encoder NLI model.*
> 5. *Outputs an AI-assisted adjudication—**Supported**, **Misleading**, or **Uncertain**—accompanied by predictive uncertainty bounds and an auditable attribution report.*
> 
> *Let me demonstrate the three core scenarios live on the dashboard."*

---

## 2. Live Demonstration Script (Three Test Scenarios)

### Scenario 1: Fully Supported Multimodal News
* **Action:** Select **Case 1** from the sidebar dropdown (*"Case 1: Fully Supported News"*).
* **Inputs Loaded:**
  * **Text:** *"Firefighters battle a major industrial warehouse blaze in East London on Sunday morning."*
  * **Visual:** Warehouse fire scene.
* **Click:** **"🚀 Verify Multimodal Content"**.
* **What to Show Your Professor:**
  1. **Verdict Badge:** `✅ SUPPORTED / REAL` (Confidence: ~85%).
  2. **OpenCLIP Metric:** High cosine similarity (~0.28 to 0.32 > 0.22 threshold), confirming visual alignment.
  3. **Evidence Box:** Surfaces Document `ev_01` (*"London Fire Brigade deployed 10 fire engines..."*) with DeBERTa NLI Entailment probability > 80%.
  4. **Key Takeaway:** Both modalities and retrieved evidence confirm the claim.

### Scenario 2: Cross-Modal Mismatch / Cheapfake (Out-of-Context Image)
* **Action:** Select **Case 2** from the sidebar dropdown (*"Case 2: Cross-Modal Mismatch"*).
* **Inputs Loaded:**
  * **Text:** *"Massive blizzard paralyzes public transportation across downtown Chicago with over 2 feet of snow."*
  * **Visual:** Sunny tropical beach scene.
* **Click:** **"🚀 Verify Multimodal Content"**.
* **What to Show Your Professor:**
  1. **Verdict Badge:** `🚨 MISLEADING / CONTRADICTED`.
  2. **OpenCLIP Metric:** Low cosine similarity (~0.12 to 0.16 < 0.22 threshold).
  3. **Warning Alert:** Shows *"Visual Mismatch Detected: Image appears out-of-context or repurposed"*.
  4. **Key Takeaway:** A text-only model would miss this manipulation completely, but our multimodal alignment stage flags the cross-modal discrepancy immediately.

### Scenario 3: Insufficient / Uncertain Claim
* **Action:** Select **Case 3** from the sidebar dropdown (*"Case 3: Insufficient / Uncertain Claim"*).
* **Inputs Loaded:**
  * **Text:** *"Archaeologists unearth secret underground gold vaults beneath the city of Edinburgh."*
* **Click:** **"🚀 Verify Multimodal Content"**.
* **What to Show Your Professor:**
  1. **Verdict Badge:** `⚠️ UNCERTAIN / INSUFFICIENT`.
  2. **Uncertainty Metric:** High predictive uncertainty (~75%+).
  3. **Retrieved Evidence Box:** Shows *"No Relevant Evidence Found in Knowledge Base"*.
  4. **Key Takeaway:** Unlike black-box classifiers that hallucinate a forced Real/Fake label, our system safely abstains when evidence is missing.

---

## 3. Anticipated Viva Defense Questions & Answers

### Q1: Why not simply fine-tune a binary classifier (Real vs. Fake) end-to-end?
* **Answer:** *"End-to-end binary classifiers suffer from severe shortcut learning and dataset artifacts. They memorize stylistic cues or domain biases rather than checking facts. When tested in the wild, their performance collapses. By decoupling evidence retrieval from cross-encoder NLI verification, our system grounds every claim in retrievable documents and provides human-auditable citations."*

### Q2: Does high CLIP similarity prove that the news is true?
* **Answer:** *"No, Professor. CLIP measures visual-textual consistency, not real-world factual truth. An entirely fabricated story can be paired with an AI-generated or matching stock photo; CLIP would report high alignment despite the claim being false. That is why our architecture combines CLIP for out-of-context detection with external RAG retrieval for factual verification."*

### Q3: How do you handle uncertainty, and why is it important?
* **Answer:** *"In high-stakes misinformation detection, false positives and false negatives carry serious societal costs. We implement selective classification: when external evidence is insufficient, contradictory, or of low retrieval confidence, the system outputs 'Uncertain'. This flags the item for human expert fact-checking rather than generating an ungrounded guess."*

### Q4: Are the confidence scores calibrated probabilities?
* **Answer:** *"In this baseline demonstration, confidence and uncertainty are heuristic fusion scores computed from NLI entropy and cross-modal margin distances. In our formal research evaluation protocol (`docs/evaluation_protocol.md`), we explicitly document that these are heuristic indices, and we outline post-hoc calibration techniques (such as Temperature Scaling or Conformal Prediction) for formal uncertainty guarantees."*

### Q5: How did you prevent data leakage during system design?
* **Answer:** *"We strictly segregated prediction from evaluation. During pipeline inference, the model never accesses dataset ground-truth labels. Furthermore, in our evaluation protocol, we distinguish Oracle-Evidence Evaluation (where gold paired documents are supplied to test reasoning in isolation) from Retrieval-Based Evaluation (where evidence must be discovered from an unannotated corpus without oracle shortcuts)."*
