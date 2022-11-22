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

```python
import pandas as pd
import seaborn as sns
import tqdm
import os
import sys
import gc
import numpy as np
from utils import read_parquet_dataset_from_local
```

```python
TRAIN_TRANSACTIONS_PATH = '../train_transactions_contest/'
TEST_TRANSACTIONS_PATH = '../test_transactions_contest/'
TRAIN_TARGET_PATH = '../train_target.csv'
TEST_TARGET_PATH = '../test_target_contest.csv'
```

```python
from features import extract_basic_aggregations

def prepare_transactions_dataset(path_to_dataset: str,
                                 num_parts_to_preprocess_at_once: int = 1,
                                 num_parts_total: int=50, 
                                 save_to_path=None,
                                 verbose: bool=False):
    """
    Creates pd.DataFrame with an extracted features from card transaction histories
    
    path_to_dataset: str  
        path to the transaction dataset directory
    num_parts_to_preprocess_at_once: int 
        number of partitions to load and process at once
    num_parts_total: int 
        total number of partitions to process, defaul is all (50)
    save_to_path: str
        directory to save processed data, if None then skip
    verbose: bool
        print info about each processing chunk of data
    """
    preprocessed_frames = []
    block = 0
    for step in tqdm.tqdm_notebook(range(0, num_parts_total, num_parts_to_preprocess_at_once), 
                                   desc="Transforming transactions data"):
        transactions_frame = read_parquet_dataset_from_local(path_to_dataset, step, num_parts_to_preprocess_at_once, 
                                                             verbose=verbose)
        features = extract_basic_aggregations(transactions_frame, 
                                              cat_columns=['mcc_category', 'day_of_week', 'operation_type'])
        if save_to_path:
            block_as_str = str(block)
            if len(block_as_str) == 1:
                block_as_str = '00' + block_as_str
            else:
                block_as_str = '0' + block_as_str
            features.to_parquet(os.path.join(save_to_path, f'processed_chunk_{block_as_str}.parquet'))
            
        preprocessed_frames.append(features)
    return pd.concat(preprocessed_frames, axis=0, ignore_index=True)
```

```python
data = prepare_transactions_dataset(TRAIN_TRANSACTIONS_PATH, num_parts_to_preprocess_at_once=5, num_parts_total=50)
```

```python
data
```

```python
data.info()
```

```python
data.isna().any().any()
```

```python
nunique_s = data.nunique()
```

```python
nunique_s[nunique_s == 1]
```

```python
data = data.loc[:, nunique_s != 1]
```

```python
train_target = pd.read_csv(TRAIN_TARGET_PATH)
```

```python
train_target
```

```python
data = pd.merge(data, train_target, on='app_id', validate='one_to_one')
```

```python
data
```

```python
from sklearn.metrics import roc_auc_score
```

Baseline dummy model using prior probabilities for each product class

```python
from sklearn.base import BaseEstimator, ClassifierMixin
```

```python
class DummyBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    def fit(self, X, y):
        d = pd.concat([X['product'], y], axis=1)
        self.product_probas = d.groupby('product')['flag'].mean()

    def predict_proba(self, X):
        flag1_proba = self.product_probas[X['product']].to_numpy().reshape(-1, 1)
        return np.concatenate((1 - flag1_proba, flag1_proba), axis=1)

    def predict(self, X):
        return np.round(self.predict_proba(X)[:, 1])
```

```python
clf = DummyBinaryClassifier()
```

```python
clf.fit(data[['product']], data['flag'])
```

```python
clf.predict_proba(data[['product']])
```

```python
clf.predict(data[['product']])
```

```python
roc_auc_score(y_true=data['flag'], y_score=clf.predict_proba(data[['product']])[:, 1])
```
