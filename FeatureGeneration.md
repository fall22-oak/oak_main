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
from utils import read_parquet_dataset_from_local
```

```python
TRAIN_TRANSACTIONS_PATH = '../train_transactions_contest/'
TEST_TRANSACTIONS_PATH = '../test_transactions_contest/'
TRAIN_TARGET_PATH = '../train_target.csv'
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

```
