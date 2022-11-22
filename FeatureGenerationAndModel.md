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
test_data = prepare_transactions_dataset(TEST_TRANSACTIONS_PATH, num_parts_to_preprocess_at_once=5, num_parts_total=50)
```

```python
data
```

```python
data.info()
```

```python
test_data.head()
```

```python
test_data.info()
```

```python
data.isna().any().any()
```

```python
test_data.isna().any().any()
```

```python
nunique_data = data.nunique()
```

```python
nunique_data[nunique_data == 1]
```

```python
nunique_test_data = test_data.nunique()
```

```python
nunique_test_data[nunique_test_data == 1]
```

OK we are safe to remove these two columns from both train and test datasets

```python
data = data.loc[:, nunique_data != 1]
test_data = test_data.loc[:, nunique_test_data != 1]
```

```python
(data.nunique() == 1).any(), (test_data.nunique() == 1).any()
```

```python
train_target = pd.read_csv(TRAIN_TARGET_PATH)
```

```python
train_target
```

```python
test_target = pd.read_csv(TEST_TARGET_PATH)
```

```python
test_target.head()
```

```python
data = pd.merge(data, train_target, on='app_id', validate='one_to_one')
```

```python
test_data = pd.merge(test_data, test_target, on='app_id', validate='one_to_one')
```

```python
data
```

```python
test_data
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
baseline_clf = DummyBinaryClassifier()
```

```python
baseline_clf.fit(data[['product']], data['flag'])
```

```python
baseline_clf.predict_proba(data[['product']])
```

```python
baseline_clf.predict(data[['product']])
```

```python
roc_auc_score(y_true=data['flag'], y_score=baseline_clf.predict_proba(data[['product']])[:, 1])
```

```python
from sklearn.model_selection import train_test_split
```

```python
train_data, val_data = train_test_split(data.copy(), test_size=0.1, random_state=42)
```

```python
baseline_clf.fit(train_data[['product']], train_data['flag'])
```

```python
roc_auc_score(y_true=val_data['flag'], y_score=baseline_clf.predict_proba(val_data[['product']])[:, 1])
```

Let's create list of all features we are going to use

```python
features = train_data.columns.difference(['app_id', 'flag'])
```

```python
from xgboost import XGBClassifier
```

```python
# WARNING: CPU training is much slower than GPU!
XGB_CPU_TRAIN = False
if XGB_CPU_TRAIN:
    xgb_method_param = {'tree_method': 'auto'}
else:
    xgb_method_param = {'tree_method': 'gpu_hist'}
```

```python
xgb_clf = XGBClassifier(max_depth=2, **xgb_method_param)
```

```python
xgb_clf.fit(train_data[features], train_data['flag'])
```

```python
roc_auc_score(y_true=val_data['flag'], y_score=xgb_clf.predict_proba(val_data[features])[:, 1])
```

```python
from sklearn.model_selection import GridSearchCV
```

```python
n_splits = 5
```

```python
gs_xgb = GridSearchCV(estimator=XGBClassifier(early_stopping_rounds=10,
                                              **xgb_method_param,
                                              n_estimators=100,
                                              random_seed=42,
                                              eval_metric='auc'),
                      param_grid={"max_depth": [2, 8, 16],
                                  'learning_rate': [0.01, 0.1, 1]},
                      scoring='roc_auc',
                      cv=n_splits)
gs_xgb.fit(X=train_data[features], y=train_data['flag'], eval_set=[(val_data[features], val_data['flag'])])
```

```python
gs_xgb.best_params_
```

```python
gs_xgb.best_score_
```

```python
gs_xgb.cv_results_
```

```python
gs_xgb2 = GridSearchCV(estimator=XGBClassifier(early_stopping_rounds=10,
                                              **xgb_method_param,
                                              n_estimators=100,
                                              random_seed=42,
                                              eval_metric='auc'),
                      param_grid={"max_depth": [6, 8, 10],
                                  'learning_rate': [0.1]},
                      scoring='roc_auc',
                      cv=n_splits)
gs_xgb2.fit(X=train_data[features], y=train_data['flag'], eval_set=[(val_data[features], val_data['flag'])])
```

```python
gs_xgb2.best_params_
```

```python
gs_xgb2.best_score_
```

```python
gs_xgb3 = GridSearchCV(estimator=XGBClassifier(early_stopping_rounds=10,
                                              **xgb_method_param,
                                              n_estimators=100,
                                              random_seed=42,
                                              eval_metric='auc'),
                      param_grid={"max_depth": [7, 8, 9],
                                  'learning_rate': [0.08, 0.1, 0.2]},
                      scoring='roc_auc',
                      cv=n_splits)
gs_xgb3.fit(X=train_data[features], y=train_data['flag'], eval_set=[(val_data[features], val_data['flag'])])
```

```python
gs_xgb3.best_params_
```

```python
gs_xgb3.best_score_
```

```python
xgb_clf = XGBClassifier(early_stopping_rounds=10,
                        learning_rate = 0.1,
                        max_depth = 9,
                        **xgb_method_param,
                        n_estimators=300,
                        random_seed=42,
                        eval_metric='auc')
```

```python
xgb_clf.fit(X=train_data[features], y=train_data['flag'], eval_set=[(val_data[features], val_data['flag'])])
```

```python
from sklearn.model_selection import cross_val_score
```

```python
xgb_final = XGBClassifier(learning_rate = 0.1,
                        max_depth = 9,
                        **xgb_method_param,
                        n_estimators=211,
                        random_seed=42)
xgb_final_cv_score = cross_val_score(estimator=xgb_final, X=train_data[features], y=train_data['flag'], scoring='roc_auc', cv=n_splits)
```

```python
xgb_final_cv_score
```

```python
xgb_final_cv_score.mean()
```

Well, maybe when we do cross-validation, we have less training data and the increased n_estimators does not help?

```python
xgb_final.fit(X=data[features], y=data['flag'])
```

Now let's run the final model on the test dataset

```python
default_proba = xgb_final.predict_proba(X=test_data[features])[:,1]
```

```python
submission = pd.DataFrame(data={'app_id': test_data.app_id, 'score': default_proba})
```

```python
submission.to_csv('../xgb_submission.csv', index=False)
```
