# Hierarchical Insider Threat Detection

**Lightweight Screening + LLM-Based Reasoning + Graph-Informed Narratives**

A two-stage pipeline for detecting insider threats in enterprise audit logs. Stage 1 uses a tree-based screener over user-day windows to cut analyst workload by ~30×, then Stage 2 generates graph-informed narratives and scores them with rule, TF-IDF, LLM (Chain-of-Thought), and fusion models.

> University of Ottawa · CSI 5388 — Topics in Applied Artificial Intelligence (Group 8)

📄 **Paper:** [`paper.pdf`](paper.pdf) · 📑 **Proposal:** [`D1-Group8.pdf`](D1-Group8.pdf)

🌐 **Languages:** [English](#english) · [中文](#中文)

---

<a id="english"></a>

## English

### Authors

| Name | Email |
| --- | --- |
| Ziyuan Liu | zliu053@uottawa.ca |
| Jingxuan Xu | jxu022@uottawa.ca |
| Kair Wang | bwang105@uottawa.ca |
| Sabrina Cai | hcai062@uottawa.ca |

### Highlights

- **Benchmark:** CERT r4.2 — 330,285 user-day windows, 1,364 malicious (~0.41%, extreme imbalance).
- **Stage 1 screener (XGBoost):** retains **89.74% of malicious windows in the top 3%** of globally ranked windows. Analyst workload shrinks from 330k → ~10k windows.
- **Stage 2 best single model:** TF-IDF + Logistic Regression — **F1 0.450, PR-AUC 0.365**.
- **Best fusion run:** **Recall 0.810** at F1 0.432 (LLM + TF-IDF + Stage 1 scores).
- **17 structured features** across logon / file / device / email + **graph-informed narratives** that fold in LDAP role context, behavioral history, and cross-source timelines.
- **Scientific takeaway:** strong classical baselines (TF-IDF) remain competitive against LLMs when narratives are partially templated. Fusion is the more honest win.

### Architecture

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

### Results

#### Stage 1 — Held-out user-level test (CERT r4.2)

| Model | Precision | Recall | F1 | PR-AUC | Top-3% Recall@Pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| **XGBoost** | 0.1037 | 0.4863 | **0.1710** | **0.2837** | **0.8974** |
| Random Forest | 0.0773 | 0.4590 | 0.1323 | 0.1963 | 0.8622 |

XGBoost forwards only **9,908 / 330,285 windows** (≈ 3%) to Stage 2 while retaining **89.74%** of all malicious windows.

#### Stage 2 — Aligned evaluation split

| Method | Precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Stage 1 pool only | 0.129 | 1.000 | 0.229 | — |
| Rule baseline | 0.210 | 0.700 | 0.323 | 0.185 |
| **TF-IDF + Logistic Regression** | 0.315 | 0.787 | **0.450** | **0.365** |
| LLM (Chain-of-Thought) | 0.210 | 0.700 | 0.323 | 0.191 |
| Fusion (LLM + TF-IDF + Stage 1) | 0.294 | **0.810** | 0.432 | 0.317 |

Full ablations and robustness analysis are in [`paper.pdf`](paper.pdf).

### Tech Stack

`Python 3.10+` · `pandas` · `numpy` · `scikit-learn` · `xgboost` · `imbalanced-learn` (SMOTE / ADASYN) · `matplotlib` · OpenAI API (Stage 2 LLM evaluator)

### Repository Layout

```
.
├── data_preprocessing/       Log cleaning, sequence building, label extraction
├── model_training/           Stage 1 + Stage 2 modeling, baselines, ablations
├── utils/                    Shared helpers (LDAP)
├── paper.pdf                 Final paper (results, ablations, discussion)
├── D1-Group8.pdf             Initial proposal
├── requirements.txt
└── README.md
```

Per-folder docs:
[`data_preprocessing/README.md`](data_preprocessing/README.md) ·
[`model_training/README.md`](model_training/README.md) ·
[`utils/README.md`](utils/README.md)

### Quick Start

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

### Dataset

Experiments use the **CERT Insider Threat Test Dataset (r4.2)** from the Carnegie Mellon University Software Engineering Institute. The data is **not redistributed in this repository** per its End-User Agreement; obtain it directly from CMU SEI:

- DOI: <https://doi.org/10.1184/R1/12841247.v1>

Place the unpacked `r4.2/` folder at the repository root before running the pipeline.

### Security Note

API keys must be supplied via environment variables only. **Never** commit `OPENAI_API_KEY` (or any credential) to the repository. If a key is ever leaked, rotate it in your provider's console immediately.

---

<a id="中文"></a>

## 中文

### 作者

| 姓名 | 邮箱 |
| --- | --- |
| 刘子源 (Ziyuan Liu) | zliu053@uottawa.ca |
| 徐景轩 (Jingxuan Xu) | jxu022@uottawa.ca |
| 王凯 (Kair Wang) | bwang105@uottawa.ca |
| Sabrina Cai | hcai062@uottawa.ca |

### 项目亮点

- **基准数据集**：CERT r4.2 — 33 万条 user-day 窗口，恶意窗口约 1,364 条 (≈ 0.41%)，类别极度不平衡。
- **Stage 1 筛选器（XGBoost）**：在全局排序的前 **3% 窗口里保留了 89.74% 的恶意样本**，分析师审计工作量从 33 万 → 约 1 万窗口（缩减约 30 倍）。
- **Stage 2 最强单模型**：TF-IDF + Logistic Regression — **F1 0.450、PR-AUC 0.365**。
- **最佳融合方案**：LLM + TF-IDF + Stage 1 三路打分融合，**Recall 0.810**，F1 0.432。
- **17 个结构化特征**（覆盖 logon / file / device / email）+ **图增强叙事**（融合 LDAP 角色背景、30 天行为历史、跨源时间线）。
- **学术结论**：当叙事文本部分模板化时，强基线（TF-IDF）在该场景下依旧能压过 LLM；融合反而是更可靠、更诚实的赢法。

### 系统架构

```
   原始 CERT 日志 (logon · file · device · email · http · LDAP)
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Stage 1 — 轻量级筛选                                   │
   │  XGBoost / Random Forest · 17 个特征                    │
   │  用户级划分 · SMOTE · 代价敏感学习                      │
   └─────────────────────────────────────────────────────────┘
                              │
                              ▼
       Top-K%  ≈ 1 万条 user-day 窗口  ·  Recall@Pool ≈ 90%
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Stage 2 — 图增强叙事                                   │
   │  跨源时间线 · LDAP 上下文 · 30 天历史对比               │
   │                                                         │
   │  打分器: 规则 │ TF-IDF + LR │ LLM (CoT) │ 融合          │
   └─────────────────────────────────────────────────────────┘
                              │
                              ▼
                    可疑 user-day 排序结果
                    + 人类可读的判定理由
```

### 实验结果

#### Stage 1 — 用户级留出测试 (CERT r4.2)

| 模型 | Precision | Recall | F1 | PR-AUC | Top-3% Recall@Pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| **XGBoost** | 0.1037 | 0.4863 | **0.1710** | **0.2837** | **0.8974** |
| Random Forest | 0.0773 | 0.4590 | 0.1323 | 0.1963 | 0.8622 |

XGBoost 仅向 Stage 2 转发 **9,908 / 330,285 (≈ 3%)** 窗口，却保留了 **89.74%** 的全部恶意样本。

#### Stage 2 — 对齐评估子集

| 方法 | Precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Stage 1 池内全上报 | 0.129 | 1.000 | 0.229 | — |
| 规则基线 | 0.210 | 0.700 | 0.323 | 0.185 |
| **TF-IDF + Logistic Regression** | 0.315 | 0.787 | **0.450** | **0.365** |
| LLM (Chain-of-Thought) | 0.210 | 0.700 | 0.323 | 0.191 |
| 融合 (LLM + TF-IDF + Stage 1) | 0.294 | **0.810** | 0.432 | 0.317 |

完整消融实验和鲁棒性分析见 [`paper.pdf`](paper.pdf)。

### 技术栈

`Python 3.10+` · `pandas` · `numpy` · `scikit-learn` · `xgboost` · `imbalanced-learn` (SMOTE / ADASYN) · `matplotlib` · OpenAI API（Stage 2 LLM 评估）

### 仓库结构

```
.
├── data_preprocessing/       日志清洗、序列构建、标签提取
├── model_training/           Stage 1 + Stage 2 模型、基线、消融
├── utils/                    通用辅助工具（LDAP）
├── paper.pdf                 最终论文（结果、消融、讨论）
├── D1-Group8.pdf             立项 Proposal
├── requirements.txt
└── README.md
```

子目录文档：
[`data_preprocessing/README.md`](data_preprocessing/README.md) ·
[`model_training/README.md`](model_training/README.md) ·
[`utils/README.md`](utils/README.md)

### 快速开始

```bash
# 1. 环境
python -m venv .venv
source .venv/bin/activate         # macOS / Linux
# .\.venv\Scripts\Activate.ps1    # Windows PowerShell
pip install -r requirements.txt

# 2. 把 CERT r4.2 数据放到 ./r4.2/（不随仓库分发，见「数据集」一节）

# 3. 构建 user-day 特征
cd data_preprocessing
python build_sequences.py --data-dir ../r4.2 --window day \
    --output ../integrated_sequences_labeled.csv \
    --output-features ../features.csv

# 4. Stage 1 — 轻量筛选（推荐配置）
cd ../model_training
python stage1_screening.py --input ../features.csv --top-k 3 \
    --smote --eval-5fold --top-k-sweep 5,3,2

# 5. Stage 2 — 叙事 + 打分
python stage2_narrative.py \
    --suspicious suspicious_sequences_xgb.csv \
    --logs ../integrated_logs_labeled.csv \
    --ldap-dir ../r4.2/LDAP --raw-data-dir ../r4.2 \
    --output stage2_narratives_xgb.txt
python stage2_baselines.py        # 规则 + TF-IDF 基线
export OPENAI_API_KEY=<你的密钥>   # 仅 LLM 打分时需要
python llm_evaluator.py
python ablation_compare.py        # 生成对比表
```

### 数据集

实验使用 **CERT Insider Threat Test Dataset (r4.2)**，由卡内基梅隆大学软件工程研究所 (CMU SEI) 发布。数据集 **不随本仓库分发**（受 End-User Agreement 限制），请直接从 CMU SEI 下载：

- DOI: <https://doi.org/10.1184/R1/12841247.v1>

把解压后的 `r4.2/` 目录放在仓库根下即可运行流水线。

### 安全提醒

API 密钥**只能通过环境变量注入**，**绝对不要**把 `OPENAI_API_KEY` 或其他凭据写入仓库。一旦泄漏，请第一时间到对应平台**轮换密钥**。
