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

```
