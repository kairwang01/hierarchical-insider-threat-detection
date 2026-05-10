# Data Preprocessing

Log cleaning, multi-source aggregation, sequence construction, and labelling for the CERT r4.2 insider threat benchmark.

🌐 **Languages:** [English](#english) · [中文](#中文)

---

<a id="english"></a>

## English

### Purpose

Take raw CERT logs and produce two artifacts that downstream stages depend on:

| Output | Consumer | Granularity |
| --- | --- | --- |
| `../features.csv` | Stage 1 screener | one row per `(user, day)` window |
| `../integrated_logs_labeled.csv` | Stage 2 narrative | one row per raw event, malicious flag attached |

All scripts are run from the `data_preprocessing/` directory and write outputs to the project root by default.

### Recommended Pipeline (Sequence-Level, 4-Source Fusion)

`build_sequences.py` aggregates four sources (logon / file / device / email) by `(user, day)`, builds the Stage 1 feature table, and labels each window from the official `answers/insiders.csv`.

```bash
python build_sequences.py \
    --data-dir ../r4.2 --window day \
    --output ../integrated_sequences_labeled.csv \
    --output-features ../features.csv
```

Typical r4.2 output: ~330,285 daily windows, ~1,364 malicious (≈ 0.41%).

**Window features** (16 per window, plus 1 interaction feature added by Stage 1):

| Group | Features |
| --- | --- |
| Logon activity | `n_logon`, `n_logon_after_hours`, `logon_after_hours_ratio` |
| File activity | `n_file`, `n_file_sensitive`, `n_file_sensitive_dir` |
| Device (USB) | `n_device_connect`, `n_device_disconnect`, `n_device_total` |
| Email | `n_email`, `n_email_external`, `n_email_abnormal_attachment` |
| Burst vs. user history | `n_logon_vs_user_avg`, `n_file_vs_user_avg`, `n_device_total_vs_user_avg`, `n_email_vs_user_avg` |

The `*_vs_user_avg` ratio captures *deviation from each user's own baseline*, which matters more than deviation from the org-wide average for insider behavior.

### Record-Level Pipeline (for Stage 2 narratives)

Stage 2's narrative builder needs an event-level table with malicious flags:

```bash
python integrate_logs.py
python label_extraction.py --dataset 4.2 -f ../integrated_logs.csv
```

Outputs `../integrated_logs_labeled.csv`.

### Optional Steps

```bash
python process_ldap.py --summary                       # LDAP role/department summary
python eda_analysis.py --input ../integrated_logs_labeled.csv   # EDA
```

### Files

| File | Role |
| --- | --- |
| `data_cleaning.py` | Shared cleaning utilities (timestamp parsing, dedup, sorting) |
| `build_sequences.py` | **Recommended**: 4-source daily aggregation + sequence-level labelling + features |
| `integrate_logs.py` | Record-level integration (logon + file) |
| `label_extraction.py` | Record- and sequence-level malicious labelling from `answers/insiders.csv` |
| `feature_engineering.py` | Legacy record-level feature extraction |
| `process_ldap.py` | LDAP user/role/department parsing |
| `eda_analysis.py` | Exploratory data analysis |

### Notes

- All paths are relative — always run scripts from inside `data_preprocessing/`.
- A small monkey-patch on `platform.machine()` is applied inside `build_sequences.py` to avoid a slow WMI call on certain Windows environments. It does not change any modeling behavior.
- The label loader treats `dataset` as a string (`'4.2'`) to dodge a pandas float-vs-string comparison bug that previously yielded zero positive labels.

---

<a id="中文"></a>

## 中文

### 目标

把原始 CERT 日志加工成下游两阶段所需的两个核心文件：

| 输出 | 使用方 | 粒度 |
| --- | --- | --- |
| `../features.csv` | Stage 1 筛选器 | 每行 = 一个 `(用户, 日期)` 窗口 |
| `../integrated_logs_labeled.csv` | Stage 2 叙事 | 每行 = 一条原始事件，带恶意标签 |

所有脚本都在 `data_preprocessing/` 目录下执行，输出默认写到项目根目录。

### 推荐主流程（序列级 + 四源融合）

`build_sequences.py` 把四类日志（logon / file / device / email）按 `(用户, 日期)` 聚合，生成 Stage 1 特征表，并基于 `answers/insiders.csv` 做序列级打标。

```bash
python build_sequences.py \
    --data-dir ../r4.2 --window day \
    --output ../integrated_sequences_labeled.csv \
    --output-features ../features.csv
```

r4.2 上典型规模：约 33 万条日窗口，恶意窗口约 1,364 条（≈ 0.41%）。

**每个窗口的特征**（窗口内 16 个，Stage 1 再加 1 个交叉特征）：

| 类别 | 特征 |
| --- | --- |
| 登录活动 | `n_logon`、`n_logon_after_hours`、`logon_after_hours_ratio` |
| 文件活动 | `n_file`、`n_file_sensitive`、`n_file_sensitive_dir` |
| 设备 (USB) | `n_device_connect`、`n_device_disconnect`、`n_device_total` |
| 邮件 | `n_email`、`n_email_external`、`n_email_abnormal_attachment` |
| 相对个人均值的突发性 | `n_logon_vs_user_avg`、`n_file_vs_user_avg`、`n_device_total_vs_user_avg`、`n_email_vs_user_avg` |

`*_vs_user_avg` 比值刻画「相对该用户自身均值的突发程度」，比相对全员均值的偏移更能描述内部威胁。

### 记录级流程（Stage 2 叙事所需）

Stage 2 叙事生成器需要事件级、带标签的整合日志：

```bash
python integrate_logs.py
python label_extraction.py --dataset 4.2 -f ../integrated_logs.csv
```

输出 `../integrated_logs_labeled.csv`。

### 可选步骤

```bash
python process_ldap.py --summary                       # LDAP 部门/角色摘要
python eda_analysis.py --input ../integrated_logs_labeled.csv   # 探索性数据分析
```

### 文件说明

| 文件 | 作用 |
| --- | --- |
| `data_cleaning.py` | 通用清洗（时间解析、去重、按时间排序） |
| `build_sequences.py` | **推荐**：四源日聚合 + 序列级打标 + 特征生成 |
| `integrate_logs.py` | 记录级整合（仅 logon + file） |
| `label_extraction.py` | 基于 `answers/insiders.csv` 的记录级 / 序列级打标 |
| `feature_engineering.py` | 旧版记录级特征提取 |
| `process_ldap.py` | LDAP 用户/角色/部门解析 |
| `eda_analysis.py` | 探索性数据分析 |

### 注意

- 所有路径都是相对路径，必须在 `data_preprocessing/` 目录下运行。
- `build_sequences.py` 内部对 `platform.machine()` 做了极小的 monkey-patch，用于规避某些 Windows 环境上慢速 WMI 调用，不会改变任何建模行为。
- 标签加载器会把 `dataset` 强制按字符串 (`'4.2'`) 处理，避免 pandas 把浮点 `4.2` 解析成 float 后比较失败、出现 0 条正例的历史 bug。
