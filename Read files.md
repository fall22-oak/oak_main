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

Transaction history is taken from the year before the credit is issued

```python
days_before_max = df.groupby('app_id')['days_before'].max()
```

```python
days_before_max.min(), days_before_max.max()
```

```python
sns.histplot(days_before_max)
```

```python
g = sns.histplot(days_before_max)
g.set_yscale('log')
```

How many transactions is made before the issuance of credit?

```python
transaction_num = df.groupby('app_id').size()
```

```python
transaction_num.min(), transaction_num.max()
```

```python
sns.histplot(transaction_num)
```

```python
g = sns.histplot(transaction_num)
g.set_yscale('log')
```

```python
train_target = pd.read_csv('../train_target.csv')
```

```python
train_target
```

Each application is unique

```python
train_target.app_id.duplicated().any()
```

Let's analyze groups of financial products

```python
sns.histplot(train_target['product'], discrete=True)
```

Most of applications are not defaulted, the dataset is unbalanced

```python
sns.histplot(train_target, x='product', hue='flag', discrete=True)
```

```python
g = sns.histplot(train_target, x='product', hue='flag', discrete=True)
g.set_yscale('log')
```

```python

```
