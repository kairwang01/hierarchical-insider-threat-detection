# 工具模块 (Utils)

本文件夹包含辅助模块，供其他脚本使用。

## 模块说明

- `ldap_helper.py` - LDAP数据处理辅助模块（供 `feature_engineering.py`、`stage2_narrative.py` 等使用）

## 使用方法

这些模块会被其他脚本自动导入，通常不需要直接运行。

如果需要单独使用：

```python
from ldap_helper import LDAPProcessor
processor = LDAPProcessor('../r4.2/LDAP')
user_info = processor.get_user_info('USER_ID', '2010-06')
```
