# Datasets, Experimental Protocols & Anti-Leakage Methodology

## VERITAS-MM 2.0: Empirical Evaluation Framework

---

## 1. Benchmarking Landscape & Problem Scopes

Automated multimodal misinformation evaluation requires distinct benchmark datasets tailored to specific threat surfaces. VERITAS-MM 2.0 benchmark evaluations are grounded in two primary, rigorously verified academic datasets, supported by extensive comparative investigation into state-of-the-art fact-checking corpora.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 BENCHMARKING TAXONOMY & SCOPES                                  │
├───────────────────────────────┬─────────────────────────────────┬───────────────────────────────┤
│ BENCHMARK DATASET             │ TARGET PHENOMENON               │ PRIMARY EVALUATION METRIC     │
├───────────────────────────────┼─────────────────────────────────┼───────────────────────────────┤
│ NewsCLIPpings (Aneja et al.)  │ Visual-Semantic Cheapfake       │ Accuracy, Macro-F1, ROC-AUC,  │
│ (N = 7,264 Test; 7,024 Val)   │ Misattribution (Out-of-Context) │ Generator Breakdown, CI       │
├───────────────────────────────┼─────────────────────────────────┼───────────────────────────────┤
│ VLDBench (31,339 Records)     │ Multimodal Rhetorical Framing   │ Macro-F1, Balanced Accuracy,  │
│ (Protocol A: N = 1,094 Test)  │ & Manipulation Analysis         │ Group-Aware Bootstrap CI      │
├───────────────────────────────┼─────────────────────────────────┼───────────────────────────────┤
│ VERITE (1,000 Pairs)          │ Modality Interaction & Out-of-  │ Cross-Referenced Design Probe │
│ (Boididou et al. & Springstein)│ Context Misinformation         │ for Decoupling Boundaries     │
├───────────────────────────────┼─────────────────────────────────┼───────────────────────────────┤
│ M4FC Benchmark                │ Multi-Domain, Multimodal Fact-  │ Structural Investigation for  │
│ (Multilingual Fact-Checking)  │ Checking Pipeline Validation    │ Claim Decomposition Protocols │
└───────────────────────────────┴─────────────────────────────────┴───────────────────────────────┘
```

---

## 2. NewsCLIPpings Benchmark: Cheapfake Detection (Task A)

### Overview & Data Topology
The **NewsCLIPpings** benchmark (Aneja et al., EMNLP 2021) evaluates automated detection of **unmanipulated images repurposed in misleading textual contexts** ("cheapfakes"). Rather than using synthetic pixel deepfakes, NewsCLIPpings constructs adversarial image-caption pairs using authentic photojournalism from the Visual News corpus.

* **Total Test Set Size:** $N = 7,264$ balanced examples ($3,632$ pristine concordant pairs, $3,632$ adversarial falsified pairs).
* **Validation Set Size:** $N = 7,024$ balanced examples used exclusively for threshold calibration and margin tuning.
* **Task Formulation:** Binary classification determining whether image $I$ was originally published with caption $T$ ($y=1$, Pristine) or paired out-of-context ($y=0$, Falsified).

### Adversarial Mismatch Generation Mechanics
NewsCLIPpings generates non-trivial falsified pairs using four distinct adversarial retrieval generators ($N = 1,816$ test samples per generator split):

```
┌───────────────────────────┬─────────────────────────────────────────────────┬───────────────────────────┐
│ GENERATOR SPLIT           │ RETRIEVAL & PAIRING STRATEGY                    │ ADVERSARIAL DIFFICULTY    │
├───────────────────────────┼─────────────────────────────────────────────────┼───────────────────────────┤
│ Generator 0: Semantics    │ CLIP ViT text-to-image nearest neighbors.       │ Extreme (Near-Identical   │
│ (Text-Image NN)           │ High cross-modal semantic similarity.           │ Semantic Visual Features) │
├───────────────────────────┼─────────────────────────────────────────────────┼───────────────────────────┤
│ Generator 1: Semantics    │ SBERT text-to-text nearest neighbor search on   │ Moderate (Thematic Overlap│
│ (Text-Text NN)            │ caption text; swaps underlying visuals.         │ Across Different Events)  │
├───────────────────────────┼─────────────────────────────────────────────────┼───────────────────────────┤
│ Generator 2: Scene        │ Places365 ResNet-50 visual scene feature        │ Moderate-High (Visually   │
│ (Scene ResNet)            │ nearest neighbor search across newsrooms.       │ Concordant Environments)  │
├───────────────────────────┼─────────────────────────────────────────────────┼───────────────────────────┤
│ Generator 3: Person       │ Facial recognition & SBERT person entity match; │ High (Same Entities in    │
│ (Person Match)            │ swaps imagery depicting identical public figures│ Divergent News Contexts)  │
└───────────────────────────┴─────────────────────────────────────────────────┴───────────────────────────┘
```

---

## 3. VLDBench Benchmark: Rhetorical Framing & Manipulation

### Overview & Composition
**VLDBench** is a large-scale multimodal manipulation benchmark consisting of **31,339 news records** across mainstream digital journalism. It annotates both textual and visual manipulation, capturing sensationalist headlines, emotive clickbait, hyper-partisan framing, and rhetorical slant.

### Syndication Network Leakage: The Hidden Flaw in News Datasets
A critical finding during Phase 3 of the VERITAS-MM 2.0 research lifecycle was the discovery of **syndication graph leakage**:
* Major wire services (Associated Press, Reuters, Agence France-Presse) distribute identical or slightly edited reports to hundreds of affiliate publishers.
* Naive random splitting causes the same underlying news story to appear simultaneously in both training and test sets under different publisher URLs.
* Models trained on naive random splits achieve artificially inflated test scores by memorizing recurring syndication wire tokens rather than learning generalized rhetorical patterns.

```
       [ Reuters Syndication Wire: "Global Climate Summit Concludes in Geneva" ]
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
   [ Publisher A: The Guardian ]                   [ Publisher B: NDTV ]
   "Climate Summit Concludes"                      "Geneva Summit Reaches Deal"
   (Assigned to Training Set)                      (Assigned to Test Set)
             ▲                                               ▲
             └─────────────── SYNDICATION LEAKAGE ───────────┘
               Model memorizes wire phrases -> Test score artificially high
```

### Rigorous Split Protocols Implemented

To eradicate syndication leakage, VERITAS-MM 2.0 implemented a formal bipartite graph clustering pipeline over publisher-article pairs:

1. **Protocol A (Group-Aware Random Split):**
   - Articles belonging to the same syndication cluster are assigned atomically to either train ($70\%$), validation ($15\%$), or test ($15\%$, $N=1,094$).
   - Guarantees zero story overlap across partitions.

2. **Protocol B Inclusive (Temporal Split):**
   - Partitions articles strictly by publication date (training on historical articles, evaluating on future articles).

3. **Protocol B Strict (Temporal Purge Split):**
   - Implements temporal partitioning while completely **purging 129 syndication groups** that span the temporal split boundary.
   - Eliminates all residual long-running wire syndication contamination.

4. **Publisher-Held-Out Evaluation:**
   - Evaluates models on newsrooms completely absent from the training set, evaluating true domain generalization to unseen editorial houses.

---

## 4. Headline Sanitization & Anti-Shortcut Methodology

In Phase 5, an empirical feature audit revealed that models trained on raw headlines learned **publisher metadata shortcuts**:
* Headlines ending with `"- Reuters"`, `"| BBC News"`, or `"[The Hindu]"` allowed the text encoder to identify the source organization.
* The classifier learned to associate specific publishers with lower manipulation rates rather than analyzing the rhetorical substance of the text.

### Anchor-Bounded Regex Purge
VERITAS-MM 2.0 enforces a mandatory, deterministic sanitization step applied prior to all feature extraction and inference:

```python
import re

def sanitize_headline(text: str) -> str:
    """
    Purges trailing publisher attributions, wire tags, and bracketed newsroom metadata.
    Prevents neural models from exploiting brand shortcuts.
    """
    # 1. Strip trailing brand anchors: " - Reuters", " | CNN", " // AP News"
    cleaned = re.sub(
        r"(?i)\s*[-|–—:/\\]+\s*(reuters|associated press|ap|bbc|cnn|the guardian|the hindu|ndtv|ani|bloomberg|fox news|al jazeera)\s*$", 
        "", 
        text
    )
    # 2. Strip bracketed / parenthetical newsroom prefixes: "[BBC]", "(Reuters)"
    cleaned = re.sub(
        r"(?i)^\s*[\[\(](reuters|ap|bbc|cnn|the hindu|ndtv)[\]\)]\s*[-:–—]*\s*", 
        "", 
        cleaned
    )
    return cleaned.strip()
```

### Empirical Validation of Sanitization
* **Text-Only Model on Raw Unsanitized Headlines:** Macro-F1 $= 58.42\%$ (exploiting publisher shortcuts).
* **Text-Only Model on Sanitized Headlines:** Macro-F1 $= 55.00\%$ (reflecting true rhetorical signal).
* **Publisher-Only Baseline:** Achieved Macro-F1 $= 44.36\%$ (pure metadata guessing), confirming that sanitization successfully removed predictive publisher leakage.

---

## 5. Frozen Representation Caching Architecture

To guarantee exact reproducibility, avoid GPU memory exhaustion, and prevent train-time encoder drift, foundation model representations were pre-extracted and persisted to high-speed binary caches:

```
┌─────────────────────────────────┬─────────────────┬──────────────────┬──────────────────────────┐
│ FEATURE CACHE                   │ BACKBONE MODEL  │ OUTPUT DIMENSION │ CACHED DATASET RECORDS   │
├─────────────────────────────────┼─────────────────┼──────────────────┼──────────────────────────┤
│ Sanitized DeBERTa-v3 Text Cache │ DeBERTa-v3-base │ 768-d (Pooled)   │ 31,339 Records (VLDBench)│
├─────────────────────────────────┼─────────────────┼──────────────────┼──────────────────────────┤
│ OpenCLIP Vision Feature Cache   │ ViT-B/32        │ 768-d (Pooled)   │ 31,339 Records (VLDBench)│
├─────────────────────────────────┼─────────────────┼──────────────────┼──────────────────────────┤
│ OpenCLIP Hypersphere Cache      │ ViT-B/32        │ 512-d (Unit L2)  │ 14,288 Records (NewsCLIP)│
└─────────────────────────────────┴─────────────────┴──────────────────┴──────────────────────────┘
```

All downstream adapter architectures (Option 1 Late Concatenation and Option 3 Gated Multimodal Fusion) were trained exclusively on these frozen, verified feature caches using fixed random seed `seed=42`.
