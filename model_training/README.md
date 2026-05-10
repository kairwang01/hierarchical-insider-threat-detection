# 模型训练 (Model Training)

本文件夹包含第一阶段轻量级筛选模型的训练脚本。

## 使用方法

### 第一阶段：运行模型训练和对比（Stage 1）

在 **`model_training/`** 目录下执行。r4.2 上 `features.csv` 恶意比例约 **0.4%**，正类极少。

**最简：**

```powershell
python stage1_screening.py --input ../features.csv --top-k 5
```

**推荐（SMOTE + 用户级 5 折评估 + 多档 Top-K 汇报）：**

```powershell
python stage1_screening.py --input ../features.csv --top-k 5 --smote --eval-5fold --top-k-sweep 5,3,2
```

（Linux/macOS 下命令相同，将路径换成你的项目目录即可。）

### 第二阶段：生成图增强叙事文本（Stage 2 Narrative）

在完成 Stage 1 并生成可疑序列和整合日志后，可以使用 `stage2_narrative.py` 构建给 LLM 使用的“故事化”输入。

#### 前置条件

- 已在项目根目录跑完数据预处理，得到：
  - `integrated_logs_labeled.csv`
- 已在本目录跑完 Stage 1，得到：
  - `suspicious_sequences_xgb.csv`（或 `suspicious_sequences_rf.csv`）
- 原始日志位于 `../r4.2`：
  - `logon.csv`, `file.csv`, `device.csv`, `email.csv`, `http.csv`
  - LDAP 位于 `../r4.2/LDAP/`

#### 运行示例

在 `model_training/` 目录下（PowerShell 建议写成一行，不要用 `^` 续行）：

```bash
# 基于 XGBoost 的可疑序列生成英文叙事
python stage2_narrative.py --suspicious suspicious_sequences_xgb.csv --logs ../integrated_logs_labeled.csv --ldap-dir ../r4.2/LDAP --raw-data-dir ../r4.2 --output stage2_narratives_xgb.txt
```

整合日志加载完成后，脚本会依次读取 `device.csv` / `email.csv` / `http.csv`（尤其 `email.csv` 行数多），磁盘与解析可能占用数分钟；终端会打印当前正在读哪个文件。随后会对整合日志与三份原始表做**按用户 / (用户, 日期) 的索引**（一次性 `groupby`，避免每个可疑窗口全表扫描），再出现按窗口的进度行（`--progress-every`）。

关键参数说明：

- `--suspicious`: Stage 1 输出的可疑序列 CSV，默认 `suspicious_sequences_xgb.csv`，也可以改成 `suspicious_sequences_rf.csv`。
- `--logs`: 含标签的整合日志，默认 `../integrated_logs_labeled.csv`。
- `--ldap-dir`: LDAP CSV 目录，默认 `../r4.2/LDAP`。
- `--raw-data-dir`: 原始日志目录（包含 `device.csv`, `email.csv`, `http.csv`），默认 `../r4.2`。
- `--output`: 叙事文本输出文件名，默认 `stage2_narratives_xgb.txt`。
- `--max-users`: （可选）只对前 N 个 user-date 窗口生成叙事，便于快速调试。

输出的 `stage2_narratives_xgb.txt` 将按用户+日期组织，每段文本包括：

- 用户在该日期的部门、角色（来自 LDAP）。
- 关键文件/登录事件及与部门的冲突信息。
- 过去 30 天的行为对比（是否首次发生此类访问）。
- 按时间顺序串联的跨源时间线（logon/file + device + email + http），便于直接送入 LLM。

## 参数说明

- `--input, -i`: 输入特征文件（默认: ../features.csv）
- `--test-size`: 测试集比例（默认: 0.2）
- `--top-k`: Top K% 筛选比例（默认: 5.0）
- `--target-recall`: 目标召回率（默认: 0.99，最大化召回率）
- `--eval-5fold`: 先做 5 折用户级交叉验证评估，输出每折指标及均值±标准差到 `cv5_evaluation_report.csv`
- `--n-folds`: GroupKFold 折数（默认: 5，用于超参搜索）
- `--hp-n-iter`: 超参随机搜索次数（默认: 10；需要更细搜可改为 20+）。XGB/RF 默认结构与搜索空间已精简以加快 Stage 1。
- `--no-hp-tune`: 关闭超参搜索，仅用固定参数
- `--smote`: 对训练集做 SMOTE 过采样，缓解恶意样本极少的不平衡（与 HP 搜索同用时改为 KFold）
- `--adasyn`: 对训练集做 ADASYN 过采样（与 `--smote` 二选一）
- `--fn-weight`: 漏报代价权重（默认 2.0），用于 scale_pos_weight/class_weight，提高召回
- `--threshold-mode`: 筛选策略 — `fixed_top_k`（固定 top k%）、`percentile`（得分≥训练集 95 分位）、`adaptive_k`（按整体风险动态 k%）
- `--iso-forest`: 将 Isolation Forest 异常分作为额外特征输入 XGB/RF（无监督信号，利于未见过异常）
- `--top-k-sweep`: 逗号分隔的 k% 列表（如 `5,3,2`），在主要筛选外再汇报各 k% 下的召回与精度，便于阈值下移（如压到 3%/2% 以降低 Stage2 API 成本）

**特征**：脚本会自动添加交叉特征 `ops_per_unique_file`（总操作数/唯一文件数）、`off_hour_activity_ratio`（非工作时间活动占比，有则用）。

## 输出文件

### 5 折评估（仅当使用 `--eval-5fold` 时）
- `cv5_evaluation_report.csv` - 5 折用户级 CV 的 Precision/Recall/F1/PR-AUC 均值与标准差

### 模型文件
- `stage1_xgb_model.pkl` - XGBoost模型
- `stage1_rf_model.pkl` - Random Forest模型

### 可疑序列（用于第二阶段）
- `suspicious_sequences_xgb.csv` - XGBoost筛选的Top K%
- `suspicious_sequences_rf.csv` - Random Forest筛选的Top K%

### 分析文件
- `model_comparison.csv` - 两个模型的详细对比
- `feature_importance_xgb.csv` - XGBoost特征重要性
- `feature_importance_rf.csv` - Random Forest特征重要性

## 模型对比

脚本会同时训练XGBoost和Random Forest两个模型，并生成对比报告。

重点关注指标：
- **Test_Recall**: 测试集召回率（越高越好，减少漏报）
- **TopK_Recall**: Top K%中捕获的恶意记录比例
- **Test_PR-AUC**: Precision-Recall曲线下面积

## 注意事项

- 模型已优化为最大化召回率（减少False Negatives）
- **用户级划分**：按 `file_user` 划分，同一用户只出现在 train 或 test 一方，防止数据泄露
- **5 折用户级 CV**：加 `--eval-5fold` 时，将用户分成 5 组，每轮 4 组训练、1 组测试，轮换 5 次并汇报均值±标准差
- 需要先完成数据预处理步骤

## 已完成的工作概览

从数据预处理到两阶段检测与叙事，目前整体流水线已经具备：

- **统一数据清洗**（`data_cleaning.py`）  
  - 对 logon/file 等日志统一做时间解析、缺失值/无效值清理、去重与时间排序，并在主流程（integrate_logs、label_extraction、feature_engineering）中统一调用。

- **行为序列构建 + 多源融合**（`data_preprocessing/build_sequences.py`）  
  - 按「用户 + 时间窗口（天/周）」聚合行为序列，而不是单条记录；当前推荐配置会在 r4.2 上生成约 33 万条 `(user, day)` 窗口，其中恶意窗口约 1,364 条。  
  - 融合 logon/file/device/email 四类日志，统计登录、文件、USB、邮件的窗口级计数与统计特征，并派生 `*_vs_user_avg` 类突发性特征，用“当前窗口 / 用户长期均值”的比值刻画行为突然放大程度。  
  - 使用 `answers/insiders.csv` 在**序列级**打标：若某个用户在某时间窗口与恶意时间段有重叠，则该序列 `is_malicious=1`；`label_extraction.load_insiders` 已修复早期 `dataset` 类型不一致（float vs 字符串）导致过滤为 0 条的问题，现在在 `dataset == '4.2'` 时会正确加载约 70 名恶意用户并产生上述 1,364 条恶意窗口。

- **序列级特征工程**  
  - 登录特征：`n_logon`、`n_logon_after_hours`、`logon_after_hours_ratio` 等。  
  - 文件特征：敏感扩展名、启发式敏感目录（如 backup / C: 等）。  
  - USB 特征：`n_device_connect` / `n_device_disconnect` / `n_device_total`。  
  - 邮件特征：外发到外部域、附件/大小 95 分位以上的异常邮件计数。  
  - 突发性特征：`*_vs_user_avg` 类特征，用当前窗口与该用户长期均值的比值表示“突然升高”的行为，减弱对单一强特征（如 `is_terminated`）的依赖。

- **Stage 1 轻量级筛选模型（XGBoost + RandomForest）**（`stage1_screening.py`）  
  - 用户级数据划分：`user_level_split` 确保同一用户不会同时出现在 train/test。  
  - 5 折 GroupKFold 超参搜索（按用户分组），加上可选的 5 折用户级交叉验证评估（`--eval-5fold`）。  
  - 不平衡处理与代价敏感学习：可选 `--smote` / `--adasyn` 对训练集过采样，`--fn-weight` 控制对漏报的惩罚（scale_pos_weight / class_weight）。  
  - 阈值策略与 Top-K：支持固定 top k%、基于训练集 95 分位的概率阈值、以及按整体风险动态调整 k%；支持 `--top-k-sweep` 在多档 k% 下同时汇报召回和精度，用于系统分析「从 5% 下调到 3% / 2% 时，Recall 与 Precision 的变化」，为后续 Stage 2/人工审核的成本–收益权衡提供依据。  
  - 无监督异常信号：`--iso-forest` 将 Isolation Forest 异常分作为额外特征输入，帮助识别未见过模式的异常行为。

- **Stage 2：图增强叙事（Graph-Augmented Narrative）**（`stage2_narrative.py`）  
  - 使用 LDAP（`utils/ldap_helper.py`）获取用户的部门、角色等背景信息。  
  - 基于 `integrated_logs_labeled.csv` 和 Stage 1 的 `suspicious_sequences_*.csv`，按「用户 + 日期」构建英文叙事：  
    - 描述当天的关键文件/登录事件，并标注“部门与资源类型不匹配”等冲突信息。  
    - 对比过去 30 天的历史行为：识别“首次出现”的行为模式。  
    - 读取原始 `device.csv` / `email.csv` / `http.csv`，在叙事最后附上一段按时间排序的多源时间线（USB 插拔、外发邮件、网页访问），形成可直接喂给 LLM 的上下文“故事”。

整体上，你已经从原始日志出发，完成了：

1. 统一清洗和多源日志整合。  
2. 用户+时间窗口粒度的行为序列与特征构建。  
3. 用户级、带 5 折评估与代价敏感的 Stage 1 筛选模型。  
4. 针对 Top-K 的召回/成本权衡工具（`--top-k-sweep`）。  
5. 能结合 LDAP 背景、历史模式和多源日志的 Stage 2 图增强叙事文本生成。  

这些内容在本 README 以及根目录 `README.md`（端到端「从头跑」）中都有相应说明。

---

## 答辩与设计说明（TA / 报告：XGB、基线、幻觉、消融）

本节可直接用于回答：**Stage 1 模型选错怎么办**、**为何用 XGBoost**、**Stage 2 baseline 是什么**、**幻觉如何处理**、**如何证明 LLM 有提升**、**如何做分阶段消融**。下方命令均在 `model_training/` 下执行。

### Stage 1：为何选 XGBoost？它为何适合 Top-K 筛选？

- **任务形态**：Stage 1 的目标是 **对海量 `(用户, 窗口)` 按风险排序，再取 Top-K%** 送入 Stage 2，而不是在 Stage 1 上做最终「有罪/无罪」判决。树模型（梯度提升）对 **表格特征、非线性、特征交互、类别极不平衡** 适应好，且输出 **连续风险分**，天然适合 **排序 + 截断**。
- **并非排他**：本仓库 **同时训练 XGBoost 与 Random Forest**（`stage1_screening.py`），并输出 `model_comparison.csv`、特征重要性等；报告里应写清 **「以树模型族做粗筛，XGB 为默认，RF 为对照」**，而不是「只能用 XGB」。
- **若「模型选错」**：Stage 1 是 **可替换模块**。可 (1) 对比 **XGB vs RF** 的 Top-K 召回与池内噪声；(2) 调 **`--top-k` / `--top-k-sweep`** 做成本–召回敏感性分析；(3) 必要时 **提高 K% 或换特征/阈值策略**，再观察 Stage 2 指标变化。恶意样本大多落在 Top-K 内是分层的 **前提**，应用 **Stage 1 单独指标**（如 TopK_Recall、`--eval-5fold`）论证。

### Stage 2：Baseline 是什么？与 LLM 的关系

| 方法 | 说明 | 脚本 / 产物 |
|------|------|-------------|
| **池内全上报** | 进入 `suspicious_sequences_*.csv` 的窗口一律视为告警（无 Stage 2） | `stage1_pool_metrics.py`；`ablation_compare` 中 `Stage1_pool_all_1` |
| **Rule** | 叙事文本上 **正则/启发式加权**，**无训练、无 API** | `stage2_baselines.py --mode rule` → `baseline_rule_scores.csv` |
| **TF-IDF + LR** | 叙事上的 **有监督** 文本基线；**默认按用户留出测试** 更严谨；`--full-pool-fit` 为同批 in-sample 上界 | `stage2_baselines.py --mode tfidf_lr` |

**LLM**（`llm_evaluator.py` → `llm_predictions_*.jsonl`）在 **同一叙事、同一标签** 上与上表对比，才能论证「语义推理是否带来增益」。

### LLM 幻觉（Hallucination）如何应对？

- **输入锚定**：叙事由 **日志 + LDAP + 规则模板** 生成（`stage2_narrative.py`）。  
- **Prompt 硬性约束（已实现）**：`llm_evaluator.py` 的 system prompt 要求 **不得编造**叙事中未出现的主机、文件、邮件、URL、USB、时间戳；`explanation` / `primary_indicators` **只能引用叙事内事实**。  
- **输出约束**：仅输出 **结构化 JSON**，便于解析与审计。  
- **评估**：以 **窗口级标签** 的 P/R/F1 为主；高告警建议 **人工复核**。

### 如何证明「LLM 确实提升了效果」：公平对比协议

1. **固定同一批窗口与标签**：同一 `suspicious_sequences_xgb.csv`、同一叙事文件、`is_malicious` 作 ground truth。  
2. **与 TF-IDF 用户留出对齐（已实现）**：  
   - 跑 TF-IDF（**不要** `--full-pool-fit`）时加 `--write-split tfidf_eval_split.json`，得到与 sklearn `GroupShuffleSplit` 一致的 **train/test 窗口键**。  
   - 跑 LLM 时对 **同一叙事全文** 使用：`llm_evaluator.py --keys-json tfidf_eval_split.json --keys-subset test ...`，仅对 **测试子集** 调用 API。  
   - 图表脚本对同一 JSON 使用 `--keys-json ... --keys-subset test`，Rule/TF-IDF/LLM 在 **完全相同窗口** 上比 P/R/F1 与 PR 曲线。  
3. **全池对比（补充）**：`--full-pool-fit` 的 TF-IDF 与 **全量** `llm_predictions_*.jsonl` 可用于「同池 in-sample 上界」对照，须在报告中 **单独标注** 易乐观。  
4. **阈值**：`plot_stage2_comparison.py` 对 Rule/TF-IDF/LLM **扫描 0–1 阈值** 取 **验证集上 best F1** 汇报；`ablation_compare.py` 仍可用固定阈值快速看表。  
5. **LLM 聚合**：同一 `(user, date)` 多条 JSONL 取 **`risk_score` 最大**。  
6. **成本**：进入池子窗口数 × token；可调小 `--top-k`。

### 分阶段消融清单（证明 pipeline 每一步有用）

| 阶段 | 消融 / 对照 | 目的 | 命令或产物 |
|------|-------------|------|------------|
| **Stage 1** | XGB vs RF | 粗筛可替换 | `model_comparison.csv`、`suspicious_sequences_*.csv` |
| **Stage 1** | 不同 Top-K% | 成本 vs 召回 | `stage1_screening.py --top-k-sweep 5,3,2` |
| **Stage 1** | 池内全上报 vs Stage 2 | 证明精读有用 | `ablation_compare.py` |
| **Stage 2** | Rule / TF-IDF vs LLM | 语义 vs 基线 | `ablation_compare.py`、`plot_stage2_comparison.py` |
| **Stage 2** | **叙事消融（已实现）** | LDAP / 历史 / 跨源时间线贡献 | 见下节 **`stage2_narrative.py` 参数** |

### 原型范围（Prototype 声明）

当前原型在 **CERT 类基准（如 r4.2）** 上已 **端到端可跑通**：预处理 → 序列特征 → Stage 1 用户级划分与 Top-K → 叙事生成 → Stage 2 LLM 与非 LLM 基线 → 统一评估脚本。诚实边界：**full-pool TF-IDF 指标偏乐观**、**LLM 阈值需标定**、**全量 LLM 有 API 成本与延迟**——写进报告反而体现严谨性。

---

## Stage 2 基线、消融与统一评估（Baselines & ablation）

用于证明 **Stage 2（LLM）相对 Stage 1 池内全上报** 以及 **相对无 LLM 基线** 的提升；所有窗口级指标均在 `suspicious_sequences_xgb.csv` 的 `(file_user, 日期)` 标签上计算。答辩层面的设计说明见上一节 **《答辩与设计说明》**。

### 1. Stage 1 池内基线（无 Stage 2）

「凡进入 Top-K 可疑 CSV 的窗口一律视为上报」：

```bash
python stage1_pool_metrics.py --suspicious suspicious_sequences_xgb.csv
```

### 2. Stage 2 规则基线（无 API、无训练）

对叙事文本做英文关键词加权得分，输出 `baseline_rule_scores.csv`：

```bash
python stage2_baselines.py --mode rule --narratives stage2_narratives_xgb.txt --suspicious suspicious_sequences_xgb.csv --output baseline_rule_scores.csv
python eval_window_scores.py --scores baseline_rule_scores.csv --suspicious suspicious_sequences_xgb.csv --threshold 0.15 --method-name rule
```

规则分数上界通常约 **0.22**，请先 `ablation_compare` / 直方图确认再设阈值；可尝试 **0.12–0.20**。

### 3. Stage 2 TF-IDF + Logistic（轻量文本基线）

- **与全量 LLM 同协议（同一批窗口、易乐观）**：在全池上训练并打分（可能过拟合，仅用于与 `llm_predictions_*` 横向对比）：

```bash
python stage2_baselines.py --mode tfidf_lr --narratives stage2_narratives_xgb.txt --suspicious suspicious_sequences_xgb.csv --output baseline_tfidf_full_scores.csv --full-pool-fit
python eval_window_scores.py --scores baseline_tfidf_full_scores.csv --suspicious suspicious_sequences_xgb.csv --threshold 0.5 --method-name tfidf_full
```

- **按用户留出（更严谨）**：默认不写 `--full-pool-fit`，只在 **20% 用户** 的测试窗口上输出分数并打印 test 指标；输出 CSV 仅含测试子集，应用 `eval_window_scores.py` 时注意标签子集（可与 README 中「用户级 test」描述一起写进论文方法）。

### 4. LLM 结果评估（已有）

```bash
python llm_eval_metrics.py --predictions llm_predictions_xgb.jsonl --suspicious suspicious_sequences_xgb.csv --threshold 0.5
```

### 5. 一键汇总对比表（不调用 API）

先生成规则 / TF-IDF 分数 CSV（见上），再：

```bash
python ablation_compare.py --suspicious suspicious_sequences_xgb.csv --llm-jsonl llm_predictions_xgb.jsonl --rule-csv baseline_rule_scores.csv --tfidf-csv baseline_tfidf_full_scores.csv --threshold-llm 0.5 --threshold-rule 0.15 --threshold-tfidf 0.5
```

缺少的文件会自动跳过并在表中提示。若规则 CSV 的 **max(score) < --threshold-rule**，脚本会打印 **WARNING**（避免全零指标）。**默认 `--threshold-rule` 为 0.15**（与规则分数刻度匹配）。

### 6. 叙事消融（`stage2_narrative.py`，已落实）

在完整叙事上 **逐项关闭** 模块，生成不同 `.txt` 再跑同一套 Stage2 基线/LLM 即可做消融：

| 参数 | 作用 |
|------|------|
| `--no-ldap` | 不查 LDAP，标题行不含部门/角色，且无基于部门的 cross-department [NOTE] |
| `--no-history` | 去掉「过去 30 天历史对比」整段 |
| `--no-cross-source` | 去掉 device/email/http 时间线；**且不加载**三张大 CSV，生成叙事快很多 |

示例：

```bash
python stage2_narrative.py --suspicious suspicious_sequences_xgb.csv --logs ../integrated_logs_labeled.csv --ldap-dir ../r4.2/LDAP --raw-data-dir ../r4.2 --output stage2_narratives_ablate_no_ldap.txt --no-ldap
```

### 7. TF-IDF 与 LLM 同一测试窗口（已落实）

```bash
# 1) 用户留出的 TF-IDF，并写出划分 JSON（与训练时同一 random_state/test_size）
python stage2_baselines.py --mode tfidf_lr --narratives stage2_narratives_xgb.txt --suspicious suspicious_sequences_xgb.csv --output baseline_tfidf_test_scores.csv --write-split tfidf_eval_split.json --random-state 42 --test-size 0.2

# 2) 仅对 test 窗口跑 LLM（建议单独输出文件，勿与全量 jsonl 混用导致 resume 错位）
python llm_evaluator.py --input stage2_narratives_xgb.txt --output llm_predictions_test.jsonl --keys-json tfidf_eval_split.json --keys-subset test --model gpt-4o-mini
```

# 3) （推荐）按**同一** `tfidf_eval_split.json` 重算 TF-IDF：写出 **test-only** + **train+test 诚实融合用** CSV（test 侧为用户留出，避免融合里误用 full-pool TF-IDF）
python stage2_baselines.py --mode tfidf_lr --narratives stage2_narratives_xgb.txt --suspicious suspicious_sequences_xgb.csv --use-split-json tfidf_eval_split.json --output baseline_tfidf_test_scores.csv --honest-fusion-csv baseline_tfidf_honest_fusion.csv --random-state 42 --test-size 0.2

### 8. 图表对比（`plot_stage2_comparison.py`，已落实）

依赖：`pip install matplotlib`（已写入根目录 `requirements.txt`）。

在 **同一批 eval 窗口** 上对比 Stage1 全上报、Rule、TF-IDF、LLM；对打分方法 **自动扫阈值** 取 **best F1**，并输出 **PR 曲线**（sklearn `precision_recall_curve` + AUC）。

```bash
# 全池窗口（16k+）
python plot_stage2_comparison.py --suspicious suspicious_sequences_xgb.csv --rule-csv baseline_rule_scores.csv --tfidf-csv baseline_tfidf_full_scores.csv --llm-jsonl llm_predictions_xgb.jsonl --out-dir figures

# 与 TF-IDF test 子集严格对齐（需先有 test 分数 CSV + 测试集 LLM jsonl）
python plot_stage2_comparison.py --suspicious suspicious_sequences_xgb.csv --keys-json tfidf_eval_split.json --keys-subset test --rule-csv baseline_rule_scores.csv --tfidf-csv baseline_tfidf_test_scores.csv --llm-jsonl llm_predictions_test.jsonl --out-dir figures
```

生成：`figures/stage2_metrics_summary.csv`（及同系列 `*.png`）。若使用 **`--eval-mode`** 非 `best_f1`，文件名会自动加后缀（或用 **`--out-suffix`**），避免覆盖默认 best-F1 图。

**固定约束（与「各方法各自 best F1」对比）**：例如全体在 **召回不低于 0.75** 的前提下比精度：

```bash
python plot_stage2_comparison.py --suspicious suspicious_sequences_xgb.csv --keys-json tfidf_eval_split.json --keys-subset test --rule-csv baseline_rule_scores.csv --tfidf-csv baseline_tfidf_test_scores.csv --llm-jsonl llm_predictions_test.jsonl --eval-mode fixed_recall --constraint-value 0.75 --out-dir figures
```

- `fixed_precision`：精度 ≥ 给定值后再尽量提高召回。  
- `fixed_threshold`：`--constraint-value` 作为**统一阈值**（注意 Rule 分数尺度约 0–0.22，阈值不宜过大）。

可加 **`--fused-csv`** 叠一条 **逻辑回归融合**曲线；图例名用 **`--fused-label`**。若需 **两条** 融合曲线，再加 **`--fused-extra-csv`** / **`--fused-extra-label`**（见下节）。

#### 8.2 叙事消融：多份 summary 合成一张对比图（`plot_narrative_ablation_panel.py`）

对不同叙事产物分别跑完 **`plot_stage2_comparison.py`**（可指向不同 `--out-dir`），再用 **标签=CSV** 拼一张横排子图（便于报告）：

```bash
python plot_narrative_ablation_panel.py \
  --run "Full=figures_full/stage2_metrics_summary.csv" \
  --run "No LDAP=figures_no_ldap/stage2_metrics_summary.csv" \
  --run "No cross-source=figures_no_xsrc/stage2_metrics_summary.csv" \
  --output figures/ablation_narrative_panel.png \
  --csv-out figures/ablation_narrative_long.csv
```

`--methods` 默认为 `Rule` `TF-IDF` `LLM` `Fused`（按 CSV 里 `method` 列精确或前缀匹配）。消融需先对每种叙事跑 **规则/TF-IDF/LLM/（可选）融合** 并生成对应 summary。

#### 8.1 提升 LLM 排序：Stage1 + Rule + LLM（+ TF-IDF）逻辑回归融合（`improve_stage2_scores.py`）

在 **训练子集**（`tfidf_eval_split.json` 的 `train_keys`，或脚本内部 **按用户 GroupShuffleSplit**）上拟合：

`P(恶意) = σ( w · [ LLM_score, Stage1_risk_score?, Rule_score?, TF-IDF_score? ] )`（特征先 `StandardScaler`；默认列顺序 **LLM → Stage1 → Rule → TF-IDF**。加 **`--no-stage1`** 时去掉 Stage1，用于 **LLM + TF-IDF 协同**（再加 `--rule-csv` 可同时保留规则分）。）

- **Stage1** 分数来自 `suspicious_sequences_xgb.csv` 的 **`risk_score`**（同一窗口多行取 **max**）。  
- **TF-IDF**：与柱状图里「用户留出 TF-IDF」公平对比时，融合应使用 **`baseline_tfidf_honest_fusion.csv`**（由 `stage2_baselines.py --use-split-json ... --honest-fusion-csv` 生成，含 train+test 行，test 分数为留出预测）。**`baseline_tfidf_full_scores.csv`** 仅适合与全池 LLM/全池基线对齐，易乐观。  
- 仅在 **train** 上拟合，在 **test** 上打印 **AP / best-F1**，避免用测试集调系数。  
- 对 **全部池内窗口** 写出校准概率 CSV，供 `plot_stage2_comparison.py --fused-csv` 使用。

```bash
# 与 TF-IDF 同一 train/test 划分（推荐）
python improve_stage2_scores.py --suspicious suspicious_sequences_xgb.csv \\
  --llm-jsonl llm_predictions_xgb_all.jsonl --rule-csv baseline_rule_scores.csv \\
  --split-json tfidf_eval_split.json --output stage2_fused_lr_scores.csv

# 融合中增加 TF-IDF 分数（全池 TF-IDF 分数文件示例；更严谨时请用与 split 一致的 test/train 产物）
python improve_stage2_scores.py --suspicious suspicious_sequences_xgb.csv \\
  --llm-jsonl llm_predictions_xgb_all.jsonl --rule-csv baseline_rule_scores.csv \\
  --tfidf-csv baseline_tfidf_full_scores.csv \\
  --split-json tfidf_eval_split.json --output stage2_fused_with_tfidf.csv

# 仅 LLM + TF-IDF 协同（公平 TF-IDF 特征 + 全池 LLM + 同一 split）
python improve_stage2_scores.py --suspicious suspicious_sequences_xgb.csv \\
  --llm-jsonl llm_predictions_xgb_all.jsonl --tfidf-csv baseline_tfidf_honest_fusion.csv \\
  --no-stage1 --split-json tfidf_eval_split.json --output stage2_fused_llm_tfidf.csv

# 无 split 文件时：内部 80/20 按用户划分
python improve_stage2_scores.py --suspicious suspicious_sequences_xgb.csv \\
  --llm-jsonl llm_predictions_xgb_all.jsonl --rule-csv baseline_rule_scores.csv \\
  --output stage2_fused_lr_scores.csv --test-size 0.2 --random-state 42

# 画图：融合曲线 + 可选第二条融合（对比是否加入 TF-IDF）
python plot_stage2_comparison.py --suspicious suspicious_sequences_xgb.csv \\
  --rule-csv baseline_rule_scores.csv --tfidf-csv baseline_tfidf_full_scores.csv \\
  --llm-jsonl llm_predictions_xgb_all.jsonl \\
  --fused-csv stage2_fused_with_tfidf.csv --fused-label "Fused (LLM+S1+Rule+TF-IDF)" \\
  --fused-extra-csv stage2_fused_lr_scores.csv --fused-extra-label "Fused (LLM+S1+Rule)" \\
  --out-dir figures
```

不传 `--rule-csv` 时不用 Rule；不传 `--tfidf-csv` 时不用 TF-IDF（仅 **LLM + Stage1** 亦可）。**`--no-stage1`** 时必须至少提供 **`--tfidf-csv` 和/或 `--rule-csv`**。融合训练需要 **train 窗口上的 LLM 分数**：仅有 `llm_predictions_test.jsonl` 时脚本会 **WARN**，主结果建议用 **全池 LLM** + 同一 `tfidf_eval_split.json`。

### 9. 与 Proposal 对齐的补充实验（CoT / 单阶段 / 时间切片）

以下用于补齐 proposal 中 **VI.D（消融）** 与 **VI.E（鲁棒性）** 的可执行项；论文中请写明 **Stage 1 用用户级 5 折**，**Stage 2 因 LLM 成本采用单次用户留出 + 下列对照**。

#### 9.1 Chain-of-Thought vs 非 CoT（同一叙事、同一 keys）

- **CoT（默认）**：`llm_evaluator.py` 不加额外参数（`--prompt-style cot`）。
- **非 CoT（minimal prompt）**：加 **`--prompt-style minimal`**（仍输出同一 JSON 字段，但不要求逐步推理段落）。

```bash
python llm_evaluator.py --input stage2_narratives_xgb.txt --output llm_predictions_cot.jsonl --model gpt-4o-mini
python llm_evaluator.py --input stage2_narratives_xgb.txt --output llm_predictions_no_cot.jsonl --model gpt-4o-mini --prompt-style minimal
```

（若仅用 test 窗口，可加 `--keys-json tfidf_eval_split.json --keys-subset test`。）再用 `llm_eval_metrics.py` / `plot_stage2_comparison.py` 分别评估两条 jsonl。

#### 9.2 单阶段 LLM 对照（VI.D.1：无 Stage 1 排序，预算相同）

在 **与 Top-K 池相同行数** 下，从 **全序列表** 随机抽样窗口（可选 **分层**：恶意/良性数量与 `suspicious_sequences_xgb.csv` 一致）：

```bash
python build_random_pool_csv.py --source ../integrated_sequences_labeled.csv \\
  --reference suspicious_sequences_xgb.csv --stratified --seed 42 \\
  --output suspicious_sequences_random_pool.csv

python stage2_narrative.py --suspicious suspicious_sequences_random_pool.csv \\
  --logs ../integrated_logs_labeled.csv --ldap-dir ../r4.2/LDAP --raw-data-dir ../r4.2 \\
  --output stage2_narratives_random.txt

python llm_evaluator.py --input stage2_narratives_random.txt --output llm_predictions_random_pool.jsonl --model gpt-4o-mini
```

与 **Stage 1 Top-K 叙事 + LLM** 在 **同一 eval 协议**（如同一 `tfidf_eval_split.json` 的 test，或对两池分别算池内指标）下对比，即可支撑「分层是否在用同样 API 预算下改善排序质量」的叙述。

#### 9.3 时间切片鲁棒性（VI.E 的轻量代理）

在 **固定打分模型** 不变的前提下，按 **窗口日期的四分位** 切分 eval 子集，报告各段的 PR-AUC 与 best-F1（反映时间分布变化带来的指标波动，**非**再训练模型）：

```bash
python eval_robustness_time_quartiles.py --suspicious suspicious_sequences_xgb.csv \\
  --keys-json tfidf_eval_split.json --keys-subset test \\
  --llm-jsonl llm_predictions_test.jsonl

# TF-IDF 分数 CSV 同理：
python eval_robustness_time_quartiles.py --keys-json tfidf_eval_split.json --keys-subset test \\
  --scores-csv baseline_tfidf_test_scores.csv
```

#### 9.4 Knowledge Graph 表述建议（与实现一致）

当前实现为 **LDAP + 多源时间线 + 模板叙事** 的 **结构化上下文（graph-informed narrative）**，而非单独持久化的 RDF/属性图数据库。论文中建议与 proposal 措辞对齐为 **「知识图谱式关系摘要 / 图增强叙事」**，避免声称已部署完整 OntoLogX 类流水线，除非另行接入显式图存储。

### 10. 安全提示

若曾将 API Key 写入仓库文件，请在平台 **轮换密钥** 并仅从环境变量读取（例如 `$env:OPENAI_API_KEY`）。
