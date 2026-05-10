# 数据预处理 (Data Preprocessing)

本文件夹包含数据清洗和预处理的脚本。**主流程统一采用「先清洗再聚合/打标/特征」**，由 `data_cleaning.py` 提供统一清洗逻辑。

## 统一数据清洗 (data_cleaning.py)

- **时间格式**：多种格式解析，无效时间转为 NaT 并丢弃对应行。
- **缺失值**：丢弃关键字段（日期、用户）缺失或无效的记录。
- **去重与排序**：去重后按时间升序排列。
- 在 **integrate_logs**、**label_extraction**、**feature_engineering** 三步中，均在处理前调用清洗，保证主流程一致性。

## 工作流程

### 推荐：按「用户 + 时间窗口」的序列级流程（四源融合）

**build_sequences.py** 实现：按「用户 + 时间窗口（天/周）」聚合、**多源融合**（logon + file + device + email）、序列级打标，并可直接产出 `features.csv`。

```bash
# 在 data_preprocessing 目录下
python build_sequences.py --data-dir ../r4.2 --window day --output ../integrated_sequences_labeled.csv --output-features ../features.csv
```

- 邮件「外发」启发式依赖内部域子串，默认 `dtaa`；其它数据集可改 `--internal-domain yourcompany`（传空字符串则仅按是否含 `@` 判断外发）。

- **输入**：`r4.2/logon.csv`, `file.csv`, `device.csv`, `email.csv`（先清洗再聚合）
- **输出**：`integrated_sequences_labeled.csv`（每行 = 一个用户在一个时间窗口内的行为序列）、`features.csv`（供 Stage1 使用）
- **窗口**：`--window day` 按天，`--window week` 按周
- **序列级标签**：若该窗口与 answers/insiders 中该用户的恶意时间段有重叠，则标为 1（`label_extraction.label_sequence_table` 使用 `dataset` 字段严格过滤，例如 `--dataset 4.2`，已修复早期 “dataset 类型不一致导致过滤为 0 条” 的问题）

#### 设计细节与本次优化说明

- **聚合粒度**：  
  - 旧流水线以「记录级」（每条 logon/file 一行）为主，现在改为「用户 + 时间窗口（天/周）」一行，更贴近 Stage 2 叙事和 LLM 的输入粒度。  
  - 在 r4.2 上，`--window day` 的典型规模约为：`~330,000` 行窗口，其中恶意窗口约 `1,364` 行。

- **四源融合特征**（均按 `(user, window_key)` 分组）：  
  - 登录：`n_logon`、`n_logon_after_hours`、`logon_after_hours_ratio`。  
  - 文件：`n_file`、`n_file_sensitive`（敏感扩展名）、`n_file_sensitive_dir`（启发式敏感目录，如包含 backup / C: / 共享路径等）。  
  - 设备（USB）：`n_device_connect`、`n_device_disconnect`、`n_device_total`。  
  - 邮件：`n_email`、`n_email_external`（发往外部域）、`n_email_abnormal_attachment`（size/attachments 超全局 95% 分位的“异常附件”）。

- **突发性（burst）特征**：  
  - 对 `n_logon`、`n_file`、`n_device_total`、`n_email` 计算 `*_vs_user_avg`：  
    \[
      \text{col\_vs\_user\_avg} = 
      \begin{cases}
      \frac{\text{col}}{\text{user\_mean(col)} + 1e-6}, & \text{user\_mean} > 0 \\
      0, & \text{否则}
      \end{cases}
    \]  
  - 用来描述“这个窗口的行为量相对该用户长期均值放大了多少倍”，弱化模型对单一强特征的依赖，鼓励关注「突然放大」的行为模式。

- **序列级打标修复**：  
  - `label_extraction.load_insiders` 现在在过滤 `dataset` 时，统一将列和值都视为字符串（例如 `'4.2'`），避免 pandas 将 `4.2` 读成 float 时比较失败。  
  - 这保证了 `--dataset 4.2` 时可以正确加载约 70 名恶意用户，并在序列表上产生约 `1,364` 条窗口级恶意标记。

- **Windows / pandas 导入兼容性**：  
  - 某些 Windows 环境下，pandas 在导入时会通过 `platform.machine()` 间接调用 WMI 查询系统信息，导致脚本长时间无响应。  
  - 为保证预处理脚本在这类环境中可以稳定运行，本项目在 `build_sequences.py` 内部对 `platform.machine()` 做了极小的 monkeypatch，使其直接返回 `"AMD64"`，避免触发慢速 WMI 调用。  
  - 这一修改仅影响 pandas 的内部环境检测，不改变任何业务逻辑或模型行为。

---

### 记录级流程（仅 logon + file，保留兼容）

按以下顺序运行脚本：

### 1. 日志整合 (integrate_logs.py)
加载 logon/file 后先清洗，再将登录记录与文件操作关联。

```bash
python integrate_logs.py
```

**输出**: `integrated_logs.csv`

### 2. LDAP数据处理 (process_ldap.py)
处理LDAP数据，获取用户角色和部门信息。

```bash
python process_ldap.py --summary
```

**输出**: `ldap_user_summary.csv`

### 3. 标签提取 (label_extraction.py)
- **记录级打标**：每条记录若时间落在某恶意时间段内则标 1（用于 logon/file 等单条记录表）。
- **序列级打标**：若「用户+时间窗口」与 answers 中该用户的恶意时间段有重叠则标 1（用于 build_sequences 产出的序列表）。

```bash
# 记录级（为 integrate_logs 产出打标）
python label_extraction.py --dataset 4.2

# 序列级（为已有序列表打标，如未带标签的 integrated_sequences.csv）
python label_extraction.py --sequences ../integrated_sequences.csv --output ../integrated_sequences_labeled.csv --dataset 4.2
```

**输出**: `integrated_logs_labeled.csv` 或 `integrated_sequences_labeled.csv`

### 4. 特征工程 (feature_engineering.py)
提取机器学习特征。

```bash
python feature_engineering.py --input ../integrated_logs_labeled.csv --output features.csv
```

**输出**: `features.csv`

### 5. EDA分析 (可选) (eda_analysis.py)
探索性数据分析。

```bash
python eda_analysis.py --input ../integrated_logs_labeled.csv
```

## 快速运行（二选一）

在 **`data_preprocessing/`** 下执行。Windows PowerShell 与下列命令一致（注意先 `cd` 到本目录）。

**序列级（推荐，四源融合）：**

```bash
python build_sequences.py --data-dir ../r4.2 --window day --output ../integrated_sequences_labeled.csv --output-features ../features.csv
```

然后到 **`model_training/`** 跑 Stage 1，例如（详见根目录 `README.md`）：

```bash
cd ../model_training
python stage1_screening.py --input ../features.csv --top-k 5 --smote --eval-5fold --top-k-sweep 5,3,2
```

**记录级（仅 logon+file，旧流程）：**

```bash
python integrate_logs.py
python process_ldap.py --summary
python label_extraction.py --dataset 4.2 -f ../integrated_logs.csv
python feature_engineering.py --input ../integrated_logs_labeled.csv --output ../features.csv
```

## 文件说明

- `data_cleaning.py` - **统一清洗模块**（parse_timestamp_safe、clean_dataframe、clean_integrated_logs_df）
- `build_sequences.py` - **行为序列构建**（四源 logon/file/device/email，按用户+时间窗口聚合，序列级打标，可输出 features.csv）
- `integrate_logs.py` - 日志整合脚本（仅 logon+file，先清洗再匹配）
- `process_ldap.py` - LDAP数据处理脚本
- `label_extraction.py` - 标签提取脚本（记录级 + **序列级**打标，先清洗再打标）
- `feature_engineering.py` - 特征工程脚本（先清洗再提特征）
- `eda_analysis.py` - EDA分析脚本

## 注意事项

- 所有脚本使用相对路径，确保在 `data_preprocessing/` 目录下运行
- 输出文件默认保存在项目根目录
- 需要先安装依赖：`pip install -r ../requirements.txt`
