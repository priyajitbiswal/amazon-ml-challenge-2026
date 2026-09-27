# Amazon ML Challenge 2026: Business Entity Resolution

<br>

**Team Name:** BrawlDevs  
**Team Members:** Sai Pranav S R, Prannav R, Priyajit Biswal, Pretham Kumar K  
**Evaluation Metric:** S1-level Macro-$F_{0.5}$ (Precision-weighted, $\beta = 0.5$)  
**Offline Holdout Validation:** **Macro-$F_{0.5} = 0.8122$** (Macro Precision: **0.8848**, Macro Recall: **0.6927**)

<br>

---

## 1. Executive Summary

In large-scale commercial platforms, business identity data originates from multiple heterogeneous, unlinked sources—each presenting noisy, partial, and unstandardized fragments of real-world entities. 

The **Amazon ML Challenge 2026** requires linking records across three distinct sources:
- **Source 1 (`S1-`):** The deduplicated reference source (1.73M test entities).
- **Source 2 (`S2-`) & Source 3 (`S3-`):** Candidate data sources (~10M total records).

Our solution implements an end-to-end, high-precision architecture combining:
1. **8-Channel DuckDB Candidate Blocker** (Priority Cap = 150 candidates/S1).
2. **65 Canonical Pairwise Features** capturing string similarity, token overlap, numeric address alignment, script detection, and blocking provenance.
3. **Calibrated XGBoost Model C1** (`hist` tree method, `scale_pos_weight=5.84`, decision threshold $\tau^* = 0.88$).
4. **Candidate Exclusivity & Address Consistency Engine** (`src/resolver.py`) that enforces the critical ground-truth invariant: *each candidate record belongs to at most one Source 1 entity*.

This pipeline eliminated over **606,000 false positive merges**, rescued **15,000+ singletons**, and operates with **100% offline license compliance** (zero deep neural nets, zero web APIs, zero external databases).

---

## 2. Project Directory Structure

The project has been organized into a clean, flat, modular structure:

```text
amazon/
├── README.md                 # Complete documentation, methodology & reproduction guide
├── requirements.txt          # Pinned production dependencies
├── run_pipeline.py           # End-to-end inference entry point
├── package_submission.py     # Automated submission packager (generates <team>_submission.zip)
├── .gitignore                # Ignores large TSV outputs, datasets, bytecode, and zips
├── src/                      # Modular pipeline components
│   ├── __init__.py
│   ├── preprocessing.py      # Open-set text, legal suffix & address normalization
│   ├── blocking.py           # 8-channel candidate blocker (Cap 150)
│   ├── features.py           # 65 pairwise feature extractor
│   ├── feature_schema.py     # Canonical feature definitions & order
│   ├── model.py              # Supervised XGBoost wrapper & inference
│   ├── resolver.py           # Candidate exclusivity & address consistency engine
│   └── evaluate.py           # S1-level Macro-F0.5 evaluation engine
├── models/
│   └── frozen_model_c1.pkl   # Serialized production Model C1 checkpoint (260 KB)
├── output/                   # Generated submission outputs (gitignored due to file size)
│   ├── matching_results.tsv  # Scored matches for leaderboard upload (79.7 MB)
│   └── candidate_pairs.tsv   # Blocking candidate set (2.01 GB)
├── dataset/                  # Competition dataset (gitignored due to file size)
│   ├── train/                # train_source1/2/3.tsv, train_ground_truth.tsv
│   └── test/                 # test_source1/2/3.tsv
├── utils/
│   └── validate_submission.py# Official stdlib submission validator
└── docs/                     # Official challenge problem statement & instructions
    ├── 6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf
    └── 6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf
```

---

## 3. How to Generate Large Output Files

Due to GitHub's file size limits (50–100 MB per file), the large TSV outputs and raw dataset files are excluded from git via `.gitignore`:
- `output/candidate_pairs.tsv` (~2.01 GB, 154.5M candidate pairs)
- `output/matching_results.tsv` (~79.7 MB, 1.73M entity rows)
- `dataset/` (~5 GB raw data)

### 3.1 Quick Smoke Test (50 Entities, ~3 seconds)
To verify the complete pipeline without running the full dataset:

```bash
python run_pipeline.py --smoke-test --limit 50 --output-dir output
```

### 3.2 Full Test Set Inference (Generates Both Large TSV Files)
To regenerate `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the test data:

```bash
python run_pipeline.py \
    --data-dir dataset/test \
    --output-dir output \
    --model-path models/frozen_model_c1.pkl \
    --threshold 0.88 \
    --cap 150
```

### 3.3 End-to-End Execution Flow in Python
The pipeline executes sequentially in `run_pipeline.py`:

```python
import duckdb
from src.blocking import CandidateBlocker
from src.features import PairwiseFeatureExtractor
from src.model import EntityMatcherModel
from src.resolver import resolve_candidate_exclusivity

# 1. Ingest test source files into DuckDB in-memory tables
con = duckdb.connect()
# (ingest test_source1.tsv, test_source2.tsv, test_source3.tsv)

# 2. Multi-Channel Candidate Blocking (Channels A through G, Cap 150)
blocker = CandidateBlocker(con)
block_res = blocker.generate_candidates(
    s1_table_or_path="cur_s1",
    cand_table_or_path="cur_cand_pool",
    output_table="cur_candidates",
    max_candidates_per_s1=150,
)

# 3. Canonical 65-Feature Extraction
extractor = PairwiseFeatureExtractor()
X_cands = extractor.extract_feature_matrix(df_cands, s1_lookup, cand_lookup)

# 4. Model Scoring & Multi-Match Thresholding (tau* = 0.88)
model = EntityMatcherModel.load("models/frozen_model_c1.pkl")
probs = model.predict_proba(X_cands)
matched_pairs = df_cands[probs >= 0.88]

# 5. Candidate Exclusivity & Address Consistency Resolution
refined_matches = resolve_candidate_exclusivity(
    matched_pairs, s1_lookup, cand_lookup, max_matches_per_s1=8
)

# 6. Export output/matching_results.tsv and output/candidate_pairs.tsv
```

---

## 4. Problem Analysis & Key Findings

Exploratory Data Analysis across the training (2.2M S1 records) and test (1.73M S1 records) corpuses revealed critical structural properties:

### 1. The Candidate Exclusivity Invariant
- A Source 1 entity may match zero records (singletons), one record, or multiple records from Source 2 and Source 3.
- However, ground truth reveals a strict converse rule: **each candidate record (from S2 or S3) belongs to AT MOST ONE Source 1 entity** (0 out of 7,638,365 true matches violate this).
- Unconstrained pairwise classification allowed 103,789 candidate IDs to be multi-assigned across 700,270 pairs (some assigned to up to 457 different S1 entities), causing **596,481 guaranteed false positive matches**.
- Resolving multi-assigned candidates to the single S1 with highest address affinity prunes these false merges completely.

### 2. Hard Country Invariant
- 100.0% of ground-truth matches share identical country labels.
- Enforcing `candidate.country == s1.country` eliminates cross-country candidate pairs with zero recall loss.

### 3. Open-Set Country Distribution (The French Locality Trap)
- Training data contains only `India` and `US`. The test set introduces `France` (259,452 S1 records, ~15% of the test set).
- French business names frequently compound municipal names with generic nouns (e.g., `Bordeaux Club`, `Nantes Ecole`). When city names were matched as address tokens, disparate businesses across different streets within the same city falsely triggered high similarity.
- Enforcing street-type and building-number consistency eliminated these false merges, reducing French average matches from 4.92 down to 2.77 per entity.

### 4. Macro-$F_{0.5}$ Metric Dynamics
$$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$
- $\beta = 0.5$ penalizes precision errors $4\times$ heavier than recall errors.
- True singletons (Source 1 entities with no matches) score **1.0** when correctly predicted as empty, but drop to **0.0** on *any* false positive prediction.
- Pruning false merges on singletons converts scores from 0.0 directly to 1.0, creating massive score gains.

---

## 5. Candidate Generation (Blocking Strategy)

To reduce the $1.73\text{M} \times 9.97\text{M} \approx 17.2\text{ trillion}$ pair space into a feasible candidate pool, we engineered an 8-channel blocker in `src/blocking.py`:

| Channel | Priority | Description | Target Noise Pattern |
| :--- | :---: | :--- | :--- |
| **Channel A** | 100 | Exact normalized core name | Clean entity matches |
| **Channel A2**| 95 | Prefix-stripped core name | Strips `The `, `Dr `, `M/s `, `DBA: ` |
| **Channel E2**| 85 | Address number + locality anchor | Indic trade aliases & non-Latin transliterations |
| **Channel B** | 80 | Rare name tokens (Country DF $\le 200$) | Token transpositions & word reordering |
| **Channel E** | 75 | Numeric anchors (PIN / Street # + 3-char prefix) | High-noise abbreviations |
| **Channel G** | 60 | 1-edit initial-character typo key | OCR errors (`6nni` vs `Gnni`, `0` vs `O`) |
| **Channel D** | 50 | Distinctive address tokens (DF $\le 150$) | Branch locations & campus addresses |
| **Channel C** | 40 | Character 4-gram prefix + suffix index | Minor misspellings & suffix variations |

### Blocking Performance
- **Candidate Pairs Generated:** 154,577,946 pairs across 1,732,544 test S1 entities.
- **Candidate Distribution:** Mean = 89.2 pairs/S1, Median = 96.0 pairs/S1, Max = 150 pairs/S1 (strictly enforced).
- **Zero-Candidate S1s:** 5,929 (0.34%).
- **Candidate Recall Ceiling:** **88.1%** on held-out development validation.

---

## 6. Feature Engineering & Model Architecture

### 6.1 Canonical 65-Feature Taxonomy (`src/feature_schema.py`)

- **Group A: Country Integrity (2):** `country_exact_match`, `country_missing_either`.
- **Group B: Business Name Similarities (22):** Levenshtein ratio, token sort ratio, token set ratio, partial ratio, Jaro-Winkler similarity, length difference, token overlap ratio, Jaccard token similarity, prefix character match, legal suffix match, script match indicator, char 3-gram containment.
- **Group C: Address & Postal Similarities (17):** Exact address match, address Levenshtein ratio, address token Jaccard similarity, numeric token overlap ratio, numeric token exact match, postal code exact match, standardized street name match, address token length delta.
- **Group D: Cross-Field Interactions (7):** High name + high address agreement, exact name + exact postal agreement, name mismatch + address mismatch penalty, combined weighted score.
- **Group E: Source-Specific Indicators (4):** Source 2 indicator, Source 3 indicator, missing address in candidate, missing name in candidate.
- **Group F: Blocking Provenance & Priority (11):** Individual channel flags (A, A2, B, C, D, E, E2, G), cumulative priority score, total channels fired, candidate rank order.

### 6.2 Production Model C1 Configuration

| Parameter | Setting | Rationale |
| :--- | :--- | :--- |
| **Estimator** | `XGBClassifier` (XGBoost 3.2.0) | High-speed tabular gradient boosting with native NaN handling |
| **Tree Method** | `hist` | Memory-efficient histogram binning |
| **Hyperparameters** | `n_estimators=300`, `max_depth=5`, `lr=0.08` | Conservative regularization against noisy tabular text features |
| **Subsample / Colsample** | `0.80` / `0.80` | Stochastic tree & feature regularization |
| **Imbalance Weighting** | `scale_pos_weight=5.84` | Calibrated to candidate pair class imbalance (~1:13) |
| **Decision Threshold** | $\tau^* = 0.88$ | Calibrated on development split for optimal Macro-$F_{0.5}$ |
| **Model Footprint** | 259 KB serialized | Permissive Apache 2.0 license, << 8B parameters |

---

## 7. Results & Benchmarks

### 7.1 Offline Validation on Held-Out S1 Split (5,000 Entities)

Validation was conducted on a strictly held-out partition excluded from training and threshold tuning:

| Metric | Score |
| :--- | :---: |
| **Macro-$F_{0.5}$** | **0.8122** |
| **Macro Precision** | **0.8848** |
| **Macro Recall** | **0.6927** |
| **Zero-Match Entity $F_{0.5}$ (Singletons)** | **0.9412** |
| **Non-Empty Entity $F_{0.5}$** | **0.7891** |

#### Decision Threshold Sensitivity ($\tau$)

| Threshold ($\tau$) | Holdout Macro-$F_{0.5}$ | Macro Precision | Macro Recall | Notes |
| :---: | :---: | :---: | :---: | :--- |
| 0.50 | 0.6421 | 0.6120 | 0.7890 | Excessive false positives |
| 0.70 | 0.7712 | 0.8014 | 0.7289 | Precision deficit |
| 0.80 | 0.7934 | 0.8451 | 0.7145 | Balanced baseline |
| 0.85 | 0.8080 | 0.8710 | 0.7012 | High precision |
| **0.88 (Frozen)** | **0.8122** | **0.8848** | **0.6927** | **Optimal Macro-$F_{0.5}$** |
| 0.90 | 0.8095 | 0.8962 | 0.6720 | Overly conservative recall drop |

### 7.2 Full Test Output Statistics (Incorporating Resolver)

- **Total Source 1 Test Entities:** 1,732,544
- **Total Predicted Matches:** 4,302,370 (pruned 606,760 false positive duplicate assignments)
- **Entities with $\ge 1$ Matches:** 1,503,006 (86.75%)
- **Entities Predicted Empty (Singletons):** 229,538 (13.25% — matching ground truth ~12.5%)
- **Source Breakdown:** Source 2 = 2,210,812 (51.39%), Source 3 = 2,091,558 (48.61%)

#### Test Breakdown by Country

| Country | S1 Entities | % of Total | Predicted Matches | Avg Matches / S1 | Empty S1s | % S1 Empty |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **India** | 809,986 | 46.75% | 1,867,389 | 2.31 | 132,774 | 16.39% |
| **US** | 663,106 | 38.27% | 1,714,137 | 2.59 | 73,217 | 11.04% |
| **France** | 259,452 | 14.98% | 718,988 | 2.77 | 23,547 | 9.08% |
| **Total** | **1,732,544** | **100.0%** | **4,302,370** | **2.48** | **229,538** | **13.25%** |

---

## 8. Validation & Packaging

### 8.1 Official Submission Validation
Run the provided validator script from the project root:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv
```

*Output:*
```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (229538 empty, 1503006 non-empty).
PASS — no blocking issues found. Safe to submit.
```

### 8.2 Generate Official Submission ZIP Archive
To package the flat project workspace into the official submission format (`BrawlDevs_submission.zip`):

```bash
python package_submission.py
```

The script automatically packages:
- `output/matching_results.tsv` and `output/candidate_pairs.tsv`
- `Documentation_template.md` (mapped directly from this README)
- `code/business_entity_resolution/` containing `src/`, `models/`, `run_pipeline.py`, and `requirements.txt`
