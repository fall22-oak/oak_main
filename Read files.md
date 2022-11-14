---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

## Read files

```python
import pandas as pd
import seaborn as sns
```

```python
from utils import read_parquet_dataset_from_local
```

```python
df = read_parquet_dataset_from_local(path_to_dataset='../train_transactions_contest', num_parts_to_read=1)
```

```python
df.info()
```

```python
df.query('app_id == 0')
```

```python
df.query('app_id == 1')
```

```python
history_len = df.groupby('app_id').size()
```

```python
history_len.max() / 365
```

```python
history_len.min()
```

```python
sns.histplot(history_len)
```

```python
sns.histplot(history_len[history_len < 1000])
```

```python
train_target = pd.read_csv('../train_target.csv')
```

```python
train_target
```

```python
train_target.app_id.duplicated().any()
```

```python
sns.histplot(train_target['product'], discrete=True)
```

```python
sns.histplot(train_target, x='product', hue='flag', discrete=True)
```

```python
g = sns.histplot(train_target, x='product', hue='flag', discrete=True)
g.set_yscale('log')
```
