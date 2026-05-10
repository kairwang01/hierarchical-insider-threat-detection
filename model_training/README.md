# Model Training & Evaluation

Stage 1 lightweight screener (XGBoost / Random Forest) + Stage 2 graph-informed narrative generation, multiple scoring baselines (Rule, TF-IDF, LLM with CoT), score fusion, and ablation tooling.

🌐 **Languages:** [English](#english) · [中文](#中文)

---

<a id="english"></a>

## English

### Overview

| Stage | Goal | Entry script | Key outputs |
| --- | --- | --- | --- |
| 1 | Rank user-day windows; forward Top-K% to Stage 2 | `stage1_screening.py` | `suspicious_sequences_*.csv`, model `*.pkl` |
| 2 | Build narratives + score with Rule / TF-IDF / LLM / Fusion | `stage2_narrative.py` → scorers | `stage2_narratives_*.txt`, score CSVs / JSONL |

Run all commands from inside `model_training/`.

### Stage 1 — Screening

#### Recommended

```bash
python stage1_screening.py --input ../features.csv --top-k 3 \
    --smote --eval-5fold --top-k-sweep 5,3,2
```

#### Minimal (fast smoke test)

```bash
python stage1_screening.py --input ../features.csv --top-k 5
```

Both XGBoost and Random Forest are trained in one run; the screener uses a continuous risk score so it naturally fits the *ranking-and-cutoff* role of Stage 1.

#### Key arguments

| Flag | Description |
| --- | --- |
| `--top-k` | Forward Top-K% windows to Stage 2 (default 5) |
| `--top-k-sweep 5,3,2` | Report recall / precision at multiple K% — helps the cost-vs-recall trade-off |
| `--smote` / `--adasyn` | Over-sample the minority class on the training partition (mutually exclusive) |
| `--fn-weight` | Cost-sensitive weighting; raises `scale_pos_weight` / `class_weight` to penalize misses |
| `--eval-5fold` | Add a 5-fold user-level cross-validation report (writes `cv5_evaluation_report.csv`) |
| `--no-hp-tune` | Skip the randomized hyperparameter search (faster) |
| `--iso-forest` | Append Isolation Forest anomaly score as an extra feature |
| `--threshold-mode` | `fixed_top_k` / `percentile` / `adaptive_k` |

#### Outputs

| File | What |
| --- | --- |
| `stage1_xgb_model.pkl`, `stage1_rf_model.pkl` | Fitted models |
| `suspicious_sequences_xgb.csv`, `suspicious_sequences_rf.csv` | Top-K% candidate pool with `risk_score` column |
| `model_comparison.csv` | XGB vs. RF metrics side-by-side |
| `feature_importance_xgb.csv`, `feature_importance_rf.csv` | Per-feature importances |
| `cv5_evaluation_report.csv` | 5-fold CV mean ± std (only when `--eval-5fold`) |

User-level splitting (`file_user`) is enforced everywhere to prevent train/test leakage.

### Stage 2 — Graph-Informed Narratives

```bash
python stage2_narrative.py \
    --suspicious suspicious_sequences_xgb.csv \
    --logs ../integrated_logs_labeled.csv \
    --ldap-dir ../r4.2/LDAP --raw-data-dir ../r4.2 \
    --output stage2_narratives_xgb.txt
```

Each narrative entry covers one `(user, day)` window:

- LDAP context (department, role) and any cross-department conflicts.
- Key file / logon events with sensitivity flags.
- 30-day behavioral history (e.g. *first time accessing this folder*).
- Cross-source timeline (USB, email, HTTP) merged in chronological order.

#### Narrative ablation flags

| Flag | Effect |
| --- | --- |
| `--no-ldap` | Skip LDAP enrichment (no department/role lines, no department-conflict notes) |
| `--no-history` | Skip the 30-day historical comparison block |
| `--no-cross-source` | Skip device / email / http timeline; also avoids loading the three large CSVs (much faster) |
| `--max-users N` | Generate only the first N user-day windows — handy for debugging |

### Stage 2 — Scorers

#### 1. Stage 1 pool baseline (no Stage 2)

```bash
python stage1_pool_metrics.py --suspicious suspicious_sequences_xgb.csv
```

#### 2. Rule baseline (zero training, no API)

```bash
python stage2_baselines.py --mode rule \
    --narratives stage2_narratives_xgb.txt \
    --suspicious suspicious_sequences_xgb.csv \
    --output baseline_rule_scores.csv
```

#### 3. TF-IDF + Logistic Regression

User-level held-out split (default; rigorous):

```bash
python stage2_baselines.py --mode tfidf_lr \
    --narratives stage2_narratives_xgb.txt \
    --suspicious suspicious_sequences_xgb.csv \
    --output baseline_tfidf_test_scores.csv \
    --write-split tfidf_eval_split.json --random-state 42 --test-size 0.2
```

Add `--full-pool-fit` for a full-pool fit (in-sample, optimistic — only for fair comparison against full-pool LLM).

#### 4. LLM (Chain-of-Thought)

```bash
export OPENAI_API_KEY=<your-key>
python llm_evaluator.py \
    --input stage2_narratives_xgb.txt \
    --output llm_predictions_xgb.jsonl \
    --model gpt-4o-mini
```

Hallucination guardrails (built into the prompt):

- Reasoning may **only** reference facts present in the narrative.
- Output must be valid JSON matching a fixed schema.
- `explanation` and `primary_indicators` cannot mention hosts, files, emails, URLs, USB events, or timestamps that are not in the narrative.

To restrict to a held-out test split shared with TF-IDF:

```bash
python llm_evaluator.py --input stage2_narratives_xgb.txt \
    --output llm_predictions_test.jsonl \
    --keys-json tfidf_eval_split.json --keys-subset test --model gpt-4o-mini
```

#### 5. Fusion (Logistic Regression over standardized scores)

`improve_stage2_scores.py` fits `P(malicious) = σ(w · [LLM, Stage1, Rule, TF-IDF])` on the **train** keys of `tfidf_eval_split.json` and writes calibrated probabilities for every pool window.

```bash
# All four signals
python improve_stage2_scores.py --suspicious suspicious_sequences_xgb.csv \
    --llm-jsonl llm_predictions_xgb_all.jsonl \
    --rule-csv baseline_rule_scores.csv \
    --tfidf-csv baseline_tfidf_full_scores.csv \
    --split-json tfidf_eval_split.json \
    --output stage2_fused_with_tfidf.csv

# LLM + TF-IDF only (cleanest fair comparison)
python improve_stage2_scores.py --suspicious suspicious_sequences_xgb.csv \
    --llm-jsonl llm_predictions_xgb_all.jsonl \
    --tfidf-csv baseline_tfidf_honest_fusion.csv --no-stage1 \
    --split-json tfidf_eval_split.json \
    --output stage2_fused_llm_tfidf.csv
```

### Comparison Tables & Plots

```bash
# One-shot ablation table (no API required if score CSVs already exist)
python ablation_compare.py --suspicious suspicious_sequences_xgb.csv \
    --llm-jsonl llm_predictions_xgb.jsonl \
    --rule-csv baseline_rule_scores.csv \
    --tfidf-csv baseline_tfidf_full_scores.csv

# PR curves + bar charts (auto threshold sweep, best-F1)
python plot_stage2_comparison.py --suspicious suspicious_sequences_xgb.csv \
    --rule-csv baseline_rule_scores.csv \
    --tfidf-csv baseline_tfidf_full_scores.csv \
    --llm-jsonl llm_predictions_xgb.jsonl \
    --out-dir figures
```

`plot_stage2_comparison.py` supports several `--eval-mode` choices: `best_f1`, `fixed_recall`, `fixed_precision`, `fixed_threshold`. Optional fusion overlays via `--fused-csv` / `--fused-extra-csv`.

To assemble narrative-ablation panels from multiple summary CSVs:

```bash
python plot_narrative_ablation_panel.py \
    --run "Full=figures_full/stage2_metrics_summary.csv" \
    --run "No LDAP=figures_no_ldap/stage2_metrics_summary.csv" \
    --run "No cross-source=figures_no_xsrc/stage2_metrics_summary.csv" \
    --output figures/ablation_narrative_panel.png
```

### Additional Experiments

| Question | Tooling |
| --- | --- |
| **CoT vs minimal prompt** — does step-by-step reasoning help? | `llm_evaluator.py --prompt-style {cot,minimal}` |
| **Single-stage LLM control** — is hierarchy itself worth it? | `build_random_pool_csv.py` builds a same-budget random pool; rerun narratives + LLM on it |
| **Temporal robustness** — do scores hold across calendar quartiles? | `eval_robustness_time_quartiles.py` partitions evaluation windows by date and recomputes metrics |

### Honest Comparison Protocol

To claim *"the LLM helps"*, all narrative scorers must run on **identical** windows:

1. Run TF-IDF without `--full-pool-fit` and add `--write-split tfidf_eval_split.json`.
2. Run LLM with `--keys-json tfidf_eval_split.json --keys-subset test` so it scores the same test users.
3. Use the same JSON in `plot_stage2_comparison.py` (`--keys-json ... --keys-subset test`).

Optimism caveats to disclose in any paper: Stage 2 thresholds are tuned on the evaluation set itself; SMOTE inside the Stage 1 hyperparameter search is applied once on the full training partition before internal CV (held-out test users remain user-disjoint).

### File Reference

| File | Role |
| --- | --- |
| `stage1_screening.py` | Stage 1 XGB / RF screener with SMOTE, user-level CV, Top-K sweep |
| `stage1_pool_metrics.py` | "Forward all pool windows" baseline |
| `stage2_narrative.py` | Graph-informed narrative builder (LDAP + history + cross-source) |
| `stage2_baselines.py` | Rule + TF-IDF + LR baselines |
| `llm_evaluator.py` | OpenAI-backed scorer with CoT / minimal prompt |
| `llm_eval_metrics.py` | Metrics for LLM JSONL output |
| `eval_window_scores.py` | Window-level metrics from a generic score CSV |
| `improve_stage2_scores.py` | Logistic-regression fusion of standardized signals |
| `ablation_compare.py` | One-shot multi-method comparison table |
| `plot_stage2_comparison.py` | PR curves + bar charts with threshold sweeps |
| `plot_narrative_ablation_panel.py` | Multi-narrative ablation panel |
| `build_random_pool_csv.py` | Same-budget random pool for single-stage LLM control |
| `eval_robustness_time_quartiles.py` | Temporal-slice robustness diagnostic |
| `build_eval_sets.py` | Balanced eval sample extraction from full narratives |
| `label_utils.py` | Shared labelling helpers |

### Security

Never commit `OPENAI_API_KEY`. Source it from the environment:

```bash
export OPENAI_API_KEY=<your-key>      # bash / zsh
$env:OPENAI_API_KEY = "<your-key>"    # PowerShell
```

If a key has ever been committed, rotate it in the provider console immediately.

---

<a id="中文"></a>

## 中文

### 总览

| 阶段 | 目标 | 入口脚本 | 主要产物 |
| --- | --- | --- | --- |
| 1 | 给所有 user-day 窗口排序，把 Top-K% 转交 Stage 2 | `stage1_screening.py` | `suspicious_sequences_*.csv`、模型 `*.pkl` |
| 2 | 生成叙事 + 用 Rule / TF-IDF / LLM / 融合多种方式打分 | `stage2_narrative.py` → 各打分器 | `stage2_narratives_*.txt`、各类分数 CSV / JSONL |

所有命令都在 `model_training/` 目录下执行。

### Stage 1 — 轻量筛选

#### 推荐配置

```bash
python stage1_screening.py --input ../features.csv --top-k 3 \
    --smote --eval-5fold --top-k-sweep 5,3,2
```

#### 最简（快速跑通）

```bash
python stage1_screening.py --input ../features.csv --top-k 5
```

XGBoost 和 Random Forest 一次跑两个；筛选器输出连续风险分，天然契合 Stage 1「**排序 + 截断**」的角色。

#### 关键参数

| 参数 | 说明 |
| --- | --- |
| `--top-k` | 转交 Stage 2 的窗口比例（默认 5%） |
| `--top-k-sweep 5,3,2` | 同时输出多档 K% 的召回 / 精度，便于做成本–召回权衡 |
| `--smote` / `--adasyn` | 训练集少数类过采样（二选一） |
| `--fn-weight` | 代价敏感学习；提高 `scale_pos_weight` / `class_weight` 加重漏报惩罚 |
| `--eval-5fold` | 加跑 5 折用户级交叉验证（写出 `cv5_evaluation_report.csv`） |
| `--no-hp-tune` | 跳过随机超参搜索（更快） |
| `--iso-forest` | 加上 Isolation Forest 异常分作为额外特征 |
| `--threshold-mode` | `fixed_top_k` / `percentile` / `adaptive_k` |

#### 输出文件

| 文件 | 内容 |
| --- | --- |
| `stage1_xgb_model.pkl`、`stage1_rf_model.pkl` | 训练好的模型 |
| `suspicious_sequences_xgb.csv`、`suspicious_sequences_rf.csv` | Top-K% 候选池，含 `risk_score` |
| `model_comparison.csv` | XGB vs RF 指标对比 |
| `feature_importance_xgb.csv`、`feature_importance_rf.csv` | 各特征重要性 |
| `cv5_evaluation_report.csv` | 5 折 CV 的均值 ± 标准差（仅 `--eval-5fold`） |

全流程都按 `file_user` **用户级划分**，杜绝训练/测试泄漏。

### Stage 2 — 图增强叙事

```bash
python stage2_narrative.py \
    --suspicious suspicious_sequences_xgb.csv \
    --logs ../integrated_logs_labeled.csv \
    --ldap-dir ../r4.2/LDAP --raw-data-dir ../r4.2 \
    --output stage2_narratives_xgb.txt
```

每段叙事覆盖一个 `(用户, 日期)` 窗口：

- LDAP 上下文（部门、角色）和「跨部门访问」一类的冲突信息。
- 当日关键文件 / 登录事件，以及敏感性标记。
- 过去 30 天历史对比（例如「**首次**访问该目录」）。
- 跨源时间线（USB、邮件、HTTP），按时间合并排序。

#### 叙事消融开关

| 参数 | 作用 |
| --- | --- |
| `--no-ldap` | 不查 LDAP，无部门/角色行，无 cross-department 注释 |
| `--no-history` | 去掉「过去 30 天历史对比」段落 |
| `--no-cross-source` | 去掉 device / email / http 时间线；同时**不加载**三张大 CSV，明显加速 |
| `--max-users N` | 仅生成前 N 个 user-day 窗口（调试用） |

### Stage 2 — 各打分器

#### 1. Stage 1 池内全上报基线（无 Stage 2）

```bash
python stage1_pool_metrics.py --suspicious suspicious_sequences_xgb.csv
```

#### 2. 规则基线（零训练、无 API）

```bash
python stage2_baselines.py --mode rule \
    --narratives stage2_narratives_xgb.txt \
    --suspicious suspicious_sequences_xgb.csv \
    --output baseline_rule_scores.csv
```

#### 3. TF-IDF + Logistic Regression

默认按用户留出 (test 严格独立)：

```bash
python stage2_baselines.py --mode tfidf_lr \
    --narratives stage2_narratives_xgb.txt \
    --suspicious suspicious_sequences_xgb.csv \
    --output baseline_tfidf_test_scores.csv \
    --write-split tfidf_eval_split.json --random-state 42 --test-size 0.2
```

加 `--full-pool-fit` 切换为全池拟合（in-sample，偏乐观；仅用于和全池 LLM 对齐）。

#### 4. LLM（Chain-of-Thought）

```bash
export OPENAI_API_KEY=<你的密钥>
python llm_evaluator.py \
    --input stage2_narratives_xgb.txt \
    --output llm_predictions_xgb.jsonl \
    --model gpt-4o-mini
```

Prompt 内置幻觉防护：

- 推理只能引用叙事中已有的事实。
- 输出必须是符合固定 schema 的 JSON。
- `explanation` / `primary_indicators` 不得编造叙事中没有出现过的主机、文件、邮件、URL、USB 事件或时间戳。

如果只想跑 TF-IDF 共享的 test 子集：

```bash
python llm_evaluator.py --input stage2_narratives_xgb.txt \
    --output llm_predictions_test.jsonl \
    --keys-json tfidf_eval_split.json --keys-subset test --model gpt-4o-mini
```

#### 5. 融合（标准化分数 + 逻辑回归）

`improve_stage2_scores.py` 在 `tfidf_eval_split.json` 的 **train** 键上拟合 `P(恶意) = σ(w · [LLM, Stage1, Rule, TF-IDF])`，对全池窗口写出校准概率。

```bash
# 四路全融合
python improve_stage2_scores.py --suspicious suspicious_sequences_xgb.csv \
    --llm-jsonl llm_predictions_xgb_all.jsonl \
    --rule-csv baseline_rule_scores.csv \
    --tfidf-csv baseline_tfidf_full_scores.csv \
    --split-json tfidf_eval_split.json \
    --output stage2_fused_with_tfidf.csv

# 仅 LLM + TF-IDF 协同（最公平的对照）
python improve_stage2_scores.py --suspicious suspicious_sequences_xgb.csv \
    --llm-jsonl llm_predictions_xgb_all.jsonl \
    --tfidf-csv baseline_tfidf_honest_fusion.csv --no-stage1 \
    --split-json tfidf_eval_split.json \
    --output stage2_fused_llm_tfidf.csv
```

### 对比表与图表

```bash
# 一键消融对比表（已有分数 CSV 时无需 API）
python ablation_compare.py --suspicious suspicious_sequences_xgb.csv \
    --llm-jsonl llm_predictions_xgb.jsonl \
    --rule-csv baseline_rule_scores.csv \
    --tfidf-csv baseline_tfidf_full_scores.csv

# PR 曲线 + 柱状图（自动扫阈值取 best-F1）
python plot_stage2_comparison.py --suspicious suspicious_sequences_xgb.csv \
    --rule-csv baseline_rule_scores.csv \
    --tfidf-csv baseline_tfidf_full_scores.csv \
    --llm-jsonl llm_predictions_xgb.jsonl \
    --out-dir figures
```

`plot_stage2_comparison.py` 支持多种 `--eval-mode`：`best_f1`、`fixed_recall`、`fixed_precision`、`fixed_threshold`。可以用 `--fused-csv` / `--fused-extra-csv` 叠加融合曲线。

把多份消融 summary 拼成一张对比图：

```bash
python plot_narrative_ablation_panel.py \
    --run "Full=figures_full/stage2_metrics_summary.csv" \
    --run "No LDAP=figures_no_ldap/stage2_metrics_summary.csv" \
    --run "No cross-source=figures_no_xsrc/stage2_metrics_summary.csv" \
    --output figures/ablation_narrative_panel.png
```

### 补充实验

| 想回答的问题 | 用什么 |
| --- | --- |
| **CoT vs 简短 prompt** — 逐步推理是否真的有用？ | `llm_evaluator.py --prompt-style {cot,minimal}` |
| **单阶段 LLM 对照** — 分层架构本身值不值？ | `build_random_pool_csv.py` 在等预算下随机抽池，再跑叙事 + LLM |
| **时间鲁棒性** — 跨日期分位的指标是否稳定？ | `eval_robustness_time_quartiles.py` 按日期四分位重算指标 |

### 公平对比协议

如果想说「**LLM 真的有提升**」，所有打分方法必须在**完全相同**的窗口上比：

1. 跑 TF-IDF 时不要 `--full-pool-fit`，加 `--write-split tfidf_eval_split.json`。
2. 跑 LLM 加 `--keys-json tfidf_eval_split.json --keys-subset test`，对同一批 test 用户打分。
3. `plot_stage2_comparison.py` 也用同一份 JSON（`--keys-json ... --keys-subset test`）。

写论文时需要诚实声明的乐观偏差：Stage 2 阈值是在评估集上扫出来的；Stage 1 超参搜索内的 SMOTE 是在整个训练分区上一次性应用的（held-out test 仍然是用户独立的）。

### 文件索引

| 文件 | 作用 |
| --- | --- |
| `stage1_screening.py` | Stage 1 XGB / RF 筛选器，含 SMOTE、用户级 CV、Top-K 扫描 |
| `stage1_pool_metrics.py` | 「池内全上报」基线 |
| `stage2_narrative.py` | 图增强叙事生成（LDAP + 历史 + 跨源） |
| `stage2_baselines.py` | 规则 + TF-IDF + LR 基线 |
| `llm_evaluator.py` | OpenAI 打分（CoT / minimal prompt） |
| `llm_eval_metrics.py` | LLM JSONL 输出的指标计算 |
| `eval_window_scores.py` | 通用分数 CSV 的窗口级指标 |
| `improve_stage2_scores.py` | 标准化分数 + 逻辑回归融合 |
| `ablation_compare.py` | 一键多方法对比表 |
| `plot_stage2_comparison.py` | PR 曲线 + 柱状图，含阈值扫描 |
| `plot_narrative_ablation_panel.py` | 多份叙事消融拼成一张图 |
| `build_random_pool_csv.py` | 同预算随机池（单阶段 LLM 对照） |
| `eval_robustness_time_quartiles.py` | 时间分位鲁棒性诊断 |
| `build_eval_sets.py` | 从完整叙事抽出平衡评估子集 |
| `label_utils.py` | 通用打标辅助 |

### 安全提醒

绝对不要把 `OPENAI_API_KEY` 写进仓库，统一从环境变量读：

```bash
export OPENAI_API_KEY=<你的密钥>      # bash / zsh
$env:OPENAI_API_KEY = "<你的密钥>"    # PowerShell
```

如果密钥曾被提交，请立刻在平台上**轮换密钥**。
