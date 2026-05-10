# Hierarchical Insider Threat Detection

**Lightweight Screening + LLM-Based Reasoning + Graph-Informed Narratives**

A two-stage pipeline for detecting insider threats in enterprise audit logs. Stage 1 uses a tree-based screener over user-day windows to cut analyst workload by 30×, then Stage 2 generates graph-informed narratives and scores them with rule, TF-IDF, LLM (Chain-of-Thought), and fusion models.

> University of Ottawa · CSI 5388 — Topics in Applied Artificial Intelligence (Group 8)

## Authors

| Name | Email |
| --- | --- |
| Ziyuan Liu | zliu053@uottawa.ca |
| Jingxuan Xu | jxu022@uottawa.ca |
| Kair Wang | bwang105@uottawa.ca |
| Sabrina Cai | hcai062@uottawa.ca |

## Highlights

- **Benchmark:** CERT r4.2 — 330,285 user-day windows, 1,364 malicious (~0.41%, extreme imbalance).
- **Stage 1 screener (XGBoost):** retains **89.74% of malicious windows in the top 3%** of globally ranked windows. Analyst workload shrinks from 330k → ~10k windows.
- **Stage 2 best single model:** TF-IDF + Logistic Regression — **F1 0.450, PR-AUC 0.365**.
- **Best fusion run:** **Recall 0.810** at F1 0.432 (LLM + TF-IDF + Stage 1 scores).
- **17 structured features** across logon / file / device / email + **graph-informed narratives** that fold in LDAP role context, behavioral history, and cross-source timelines.
- **Scientific takeaway:** strong classical baselines (TF-IDF) remain competitive against LLMs when narratives are partially templated. Fusion is the more honest win.

## Architecture

```
   Raw CERT logs (logon · file · device · email · http · LDAP)
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Stage 1 — Lightweight Screening                        │
   │  XGBoost / Random Forest · 17 features                  │
   │  user-level split · SMOTE · cost-sensitive weighting    │
   └─────────────────────────────────────────────────────────┘
                              │
                              ▼
        Top-K%  ≈ 10k user-day windows  ·  ~90% recall @ pool
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Stage 2 — Graph-Informed Narratives                    │
   │  cross-source timeline · LDAP context · 30-day history  │
   │                                                         │
   │  Scorers: Rule │ TF-IDF + LR │ LLM (CoT) │ Fusion       │
   └─────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Ranked malicious user-days
                    + human-readable rationale
```

## Results

### Stage 1 — Held-out user-level test (CERT r4.2)

| Model | Precision | Recall | F1 | PR-AUC | Top-3% Recall@Pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| **XGBoost** | 0.1037 | 0.4863 | **0.1710** | **0.2837** | **0.8974** |
| Random Forest | 0.0773 | 0.4590 | 0.1323 | 0.1963 | 0.8622 |

XGBoost forwards only **9,908 / 330,285 windows** (≈ 3%) to Stage 2 while retaining **89.74%** of all malicious windows.

### Stage 2 — Aligned evaluation split

| Method | Precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Stage 1 pool only | 0.129 | 1.000 | 0.229 | — |
| Rule baseline | 0.210 | 0.700 | 0.323 | 0.185 |
| **TF-IDF + Logistic Regression** | 0.315 | 0.787 | **0.450** | **0.365** |
| LLM (Chain-of-Thought) | 0.210 | 0.700 | 0.323 | 0.191 |
| Fusion (LLM + TF-IDF + Stage 1) | 0.294 | **0.810** | 0.432 | 0.317 |

Full ablations and robustness analysis are in [`paper.pdf`](paper.pdf).

## Tech Stack

`Python 3.10+` · `pandas` · `numpy` · `scikit-learn` · `xgboost` · `imbalanced-learn` (SMOTE / ADASYN) · `matplotlib` · OpenAI API (Stage 2 LLM evaluator)

## Repository Layout

```
.
├── data_preprocessing/       Log cleaning, sequence building, label extraction
│   ├── build_sequences.py    User-day aggregation across 4 log sources → features.csv
│   ├── integrate_logs.py     Record-level log integration
│   ├── label_extraction.py   Sequence- and record-level malicious labelling
│   ├── feature_engineering.py
│   ├── process_ldap.py       LDAP role/department summaries
│   ├── eda_analysis.py
│   └── data_cleaning.py      Shared cleaning utilities
├── model_training/           Stage 1 + Stage 2 modeling
│   ├── stage1_screening.py        XGBoost / RF screener with SMOTE & user-level CV
│   ├── stage2_narrative.py        Graph-informed narrative generation
│   ├── stage2_baselines.py        Rule + TF-IDF baselines
│   ├── llm_evaluator.py           OpenAI-backed scorer w/ CoT prompt
│   ├── llm_eval_metrics.py        LLM evaluation metrics
│   ├── eval_window_scores.py      Window-level scoring
│   ├── ablation_compare.py        Multi-method comparison table
│   ├── plot_stage2_comparison.py  PR curves & bar charts
│   ├── improve_stage2_scores.py   Logistic-regression fusion
│   ├── eval_robustness_time_quartiles.py   Temporal-slice robustness diagnostic
│   └── stage1_pool_metrics.py
├── utils/
│   └── ldap_helper.py
├── paper.pdf                 Final paper (results, ablations, discussion)
├── D1-Group8.pdf             Initial proposal
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate         # macOS / Linux
# .\.venv\Scripts\Activate.ps1    # Windows PowerShell
pip install -r requirements.txt

# 2. Place CERT r4.2 data under ./r4.2/  (NOT included — see Dataset section)

# 3. Build user-day features
cd data_preprocessing
python build_sequences.py --data-dir ../r4.2 --window day \
    --output ../integrated_sequences_labeled.csv \
    --output-features ../features.csv

# 4. Stage 1 — screening (recommended config)
cd ../model_training
python stage1_screening.py --input ../features.csv --top-k 3 \
    --smote --eval-5fold --top-k-sweep 5,3,2

# 5. Stage 2 — narratives + scoring
python stage2_narrative.py \
    --suspicious suspicious_sequences_xgb.csv \
    --logs ../integrated_logs_labeled.csv \
    --ldap-dir ../r4.2/LDAP --raw-data-dir ../r4.2 \
    --output stage2_narratives_xgb.txt
python stage2_baselines.py        # Rule + TF-IDF baselines
export OPENAI_API_KEY=<your-key>  # only required for LLM evaluation
python llm_evaluator.py
python ablation_compare.py        # produces the comparison table
```

Detailed CLI options live in [`data_preprocessing/README.md`](data_preprocessing/README.md) and [`model_training/README.md`](model_training/README.md).

## Dataset

Experiments use the **CERT Insider Threat Test Dataset (r4.2)** from the Carnegie Mellon University Software Engineering Institute. The data is **not redistributed in this repository** per its End-User Agreement; obtain it directly from CMU SEI:

- DOI: <https://doi.org/10.1184/R1/12841247.v1>

Place the unpacked `r4.2/` folder at the repository root before running the pipeline.

## Paper

[`paper.pdf`](paper.pdf) — *Hierarchical Insider Threat Detection: Integrating Lightweight Screening with LLM-Based Reasoning and Graph-Informed Narratives.*

Covers methodology, feature engineering, hyperparameter selection, ablations (CoT vs. minimal prompt, LDAP / history / cross-source narrative components), temporal-slice robustness, and a critical discussion of when LLMs do — and do not — outperform classical baselines.

## Security Note

API keys must be supplied via environment variables only. **Never** commit `OPENAI_API_KEY` (or any credential) to the repository. If a key is ever leaked, rotate it in your provider's console immediately.
