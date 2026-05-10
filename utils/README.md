# Shared Utilities

Helper modules imported by the data preprocessing and model training scripts.

🌐 **Languages:** [English](#english) · [中文](#中文)

---

<a id="english"></a>

## English

### Modules

| Module | Used by | Purpose |
| --- | --- | --- |
| `ldap_helper.py` | `feature_engineering.py`, `stage2_narrative.py` | Parse the CERT LDAP monthly snapshots and resolve a user's department / role at a given date |

### Usage

These modules are normally imported automatically by other scripts. If needed in isolation:

```python
from ldap_helper import LDAPProcessor

processor = LDAPProcessor("../r4.2/LDAP")
user_info = processor.get_user_info("USER_ID", "2010-06")
print(user_info["department"], user_info["role"])
```

`LDAPProcessor` lazily loads each `LDAP/YYYY-MM.csv` snapshot on demand and caches it in memory.

---

<a id="中文"></a>

## 中文

### 模块

| 模块 | 调用方 | 作用 |
| --- | --- | --- |
| `ldap_helper.py` | `feature_engineering.py`、`stage2_narrative.py` | 解析 CERT 的 LDAP 月度快照，按日期解析用户的部门 / 角色 |

### 用法

通常这些模块由其他脚本自动导入。如需单独使用：

```python
from ldap_helper import LDAPProcessor

processor = LDAPProcessor("../r4.2/LDAP")
user_info = processor.get_user_info("USER_ID", "2010-06")
print(user_info["department"], user_info["role"])
```

`LDAPProcessor` 按需懒加载每份 `LDAP/YYYY-MM.csv` 快照并缓存到内存。
