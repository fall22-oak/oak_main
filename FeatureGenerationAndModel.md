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
import matplotlib.pyplot as plt
import tqdm
import os
import sys
import gc
import numpy as np
from utils import read_parquet_dataset_from_local
from joblib import dump
```

```python
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
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
                                 num_parts_to_preprocess_at_once: int=1,
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

Let's check for any NaN values

```python
data.isna().any().any()
```

```python
test_data.isna().any().any()
```

Let's check for any non-informative columns having only one unique value

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

We need to merge two dataframes because product is a feature

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

At this moment we cannot submit test predictions so we need to create our own test dataset

```python
from sklearn.model_selection import train_test_split
```

```python
train_data, inner_test_data = train_test_split(data.copy(),
                                               test_size=0.1,
                                               random_state=42,
                                               stratify=data['flag'].values)
```

This is our metric

```python
from sklearn.metrics import roc_auc_score
```

```python
cv_scores = {}
nfolds = 5
```

```python
from sklearn.model_selection import cross_val_score
```

Baseline dummy model using prior probabilities for each product class

```python
from sklearn.base import BaseEstimator, ClassifierMixin
```

```python
class DummyProductClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self):
        self.classes_ = [0, 1]
        pass

    def fit(self, X, y):
        d = pd.concat([X['product'], y], axis=1)
        self.product_probas = d.groupby('product')['flag'].mean()

    def predict_proba(self, X):
        flag1_proba = self.product_probas[X['product']].to_numpy().reshape(-1, 1)
        return np.concatenate((1 - flag1_proba, flag1_proba), axis=1)

    def decision_function(self, X):
        return self.predict_proba(X)[:, 1]

    def predict(self, X):
        return np.round(self.predict_proba(X)[:, 1])
```

```python
dummy_product_clf = DummyProductClassifier()
```

```python
cv_scores['dummy_product_clf'] = cross_val_score(dummy_product_clf,
                                                 X=train_data[['product']],
                                                 y=train_data['flag'],
                                                 scoring='roc_auc',
                                                 cv=nfolds).mean()
```

```python
cv_scores
```

```python
dummy_product_clf.fit(train_data[['product']], train_data['flag'])
```

```python

```

dummy model which gives 0 default probability

```python
from sklearn.dummy import DummyClassifier
```

```python
baseline_zero_proba_clf = DummyClassifier(strategy='constant', constant=0)
```

```python
cv_scores['baseline_zero_proba_clf'] = cross_val_score(baseline_zero_proba_clf,
                                                         X=train_data[['product']],
                                                         y=train_data['flag'],
                                                         scoring='roc_auc',
                                                         cv=nfolds).mean()
```

```python
baseline_zero_proba_clf.fit(train_data[['product']], train_data['flag'])
```

```python
cv_scores
```

```python

```

dummy model which gives 0.5 default probability

```python
baseline_05_proba_clf = DummyClassifier(strategy='uniform')
```

```python
cv_scores['baseline_05_proba_clf'] = cross_val_score(baseline_05_proba_clf,
                                                         X=train_data[['product']],
                                                         y=train_data['flag'],
                                                         scoring='roc_auc',
                                                         cv=nfolds).mean()
```

```python
baseline_05_proba_clf.fit(train_data[['product']], train_data['flag'])
```

```python
cv_scores
```

```python

```

Let's do a bit more EDA

```python
features = train_data.columns.difference(['app_id', 'flag'])
```

```python
for feature in features[:2]:
    g = sns.histplot(data=train_data, x=feature, hue='flag')
    g.set_yscale('log')
    plt.show()
```

Let's analyze correlations between features

```python
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 60))
corr = spearmanr(train_data[features.drop(['product'])]).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=features.drop(['product']).to_list(), ax=ax1, orientation='right'
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
fig.tight_layout()
plt.show()
```

```python
check_ft = ['amnt_median', 'amnt_mean', 'amnt_mean_day_of_week_1', 'amnt_mean_day_of_week_3']
check_ft = ['amnt_mean_operation_type_6', 'amnt_count_operation_type_6', 'amnt_sum_operation_type_6']
train_data[check_ft].corr()
```

```python
from sklearn.pipeline import make_pipeline
```

```python
from sklearn.decomposition import PCA
```

```python
from sklearn.preprocessing import StandardScaler
```

```python
pca = make_pipeline(StandardScaler(), PCA(n_components=0.9))
```

we need to remove the only categorical feature we have first

```python
pca_features = pca.fit_transform(train_data[features.drop('product')])
```

```python
pca['pca'].explained_variance_ratio_
```

```python
pca['pca'].explained_variance_ratio_.sum()
```

```python
pca['pca'].n_features_
```

```python
len(pca['pca'].explained_variance_ratio_)
```

We managed to reduce number of features in more than two times

```python
from sklearn.linear_model import LogisticRegression
```

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
```

```python
cat_features = ['product']
num_features = features.drop('product').to_list()
```

```python
pca_log_reg = make_pipeline(ColumnTransformer(transformers=
                                              [("num", make_pipeline(StandardScaler(), PCA(n_components=80)), num_features),
                                               ("cat", OneHotEncoder(handle_unknown='ignore'), cat_features)
                                              ]),
                            LogisticRegression())
```

```python
cv_scores['pca_log_reg'] = cross_val_score(pca_log_reg,
                                             X=train_data[features],
                                             y=train_data['flag'],
                                             scoring='roc_auc',
                                             cv=nfolds).mean()
```

```python
pca_log_reg.fit(train_data[features], train_data['flag'])
```

```python
cv_scores
```

```python

```

```python
from xgboost import XGBClassifier
```

for xgboost classifier we need additional validation set for stop criteria

```python
train_xgb_data, val_xgb_data = train_test_split(train_data.copy(),
                                                test_size=0.1,
                                                random_state=42,
                                                stratify=train_data['flag'].values)
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
xgb_clf.fit(train_xgb_data[features], train_xgb_data['flag'])
```

```python
roc_auc_score(y_true=val_xgb_data['flag'], y_score=xgb_clf.predict_proba(val_xgb_data[features])[:, 1])
```

We can find better parameters

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

gs_xgb.fit(X=train_xgb_data[features],
           y=train_xgb_data['flag'],
           eval_set=[(val_xgb_data[features],
                      val_xgb_data['flag'])])
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
gs_xgb2.fit(X=train_xgb_data[features], y=train_xgb_data['flag'], eval_set=[(val_xgb_data[features], val_xgb_data['flag'])])
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
gs_xgb3.fit(X=train_xgb_data[features], y=train_xgb_data['flag'], eval_set=[(val_xgb_data[features], val_xgb_data['flag'])])
```

```python
gs_xgb3.best_params_
```

```python
gs_xgb3.best_score_
```

```python
xgb_clf = XGBClassifier(early_stopping_rounds=10,
                        learning_rate=0.1,
                        max_depth=8,
                        **xgb_method_param,
                        n_estimators=300,
                        random_seed=42,
                        eval_metric='auc')
```

```python
xgb_clf.fit(X=train_xgb_data[features],
            y=train_xgb_data['flag'],
            eval_set=[(val_xgb_data[features],
                       val_xgb_data['flag'])])
```

```python
xgb_final = XGBClassifier(learning_rate=0.1,
                          max_depth=8,
                          **xgb_method_param,
                          n_estimators=129,
                          random_seed=42)
```

```python
xgb_final_cv_score = cross_val_score(estimator=xgb_final,
                                     X=train_data[features],
                                     y=train_data['flag'],
                                     scoring='roc_auc',
                                     cv=n_splits)
```

```python
xgb_final_cv_score
```

```python
cv_scores['xgb_tuned'] = xgb_final_cv_score.mean()
```

```python
cv_scores
```

Now let's train the final model on train_data test dataset

```python
xgb_final.fit(X=train_data[features], y=train_data['flag'])
```

```python
cv_scores
```

Let's also create cpu model for Hugging Face spaces

```python
xgb_final_cpu = XGBClassifier(learning_rate=0.1,
                          max_depth=8,
                          n_estimators=129,
                          random_seed=42)
```

```python
xgb_final_cpu.fit(X=train_data[features], y=train_data['flag'])
```

Study feature importance using xgb_final model

```python
from xgboost import plot_importance
```

```python
fig, ax = plt.subplots(figsize=(10, 40))
plot_importance(xgb_final, ax=ax, importance_type='weight')
```

```python
fig, ax = plt.subplots(figsize=(10, 40))
plot_importance(xgb_final, ax=ax, importance_type='gain')
```

```python
fig, ax = plt.subplots(figsize=(10, 40))
plot_importance(xgb_final, ax=ax, importance_type='cover')
```

SHAP importance analysis

```python
import shap
```

```python
explainer = shap.TreeExplainer(xgb_final_cpu)
```

```python
shap_values = explainer.shap_values(val_xgb_data[features])
```

```python
shap_values.shape
```

```python
shap.summary_plot(shap_values, val_xgb_data[features])
```

<!-- #region -->
Permutation importances

check https://scikit-learn.org/stable/modules/permutation_importance.html


**Warning**

Features that are deemed of low importance for a bad model (low cross-validation score) could be very important for a good model. Therefore it is always important to evaluate the predictive power of a model using a held-out set (or better with cross-validation) prior to computing importances. Permutation importance does not reflect to the intrinsic predictive value of a feature by itself but how important this feature is for a particular model. 

Permutation importances can be computed either on the training set or on a held-out testing or validation set. Using a held-out set makes it possible to highlight which features contribute the most to the generalization power of the inspected model. Features that are important on the training set but not on the held-out set might cause the model to overfit.
<!-- #endregion -->

When two features are correlated and one of the features is permuted, the model will still have access to the feature through its correlated feature. This will result in a lower importance value for both features, where they might actually be important.

One way to handle this is to cluster features that are correlated and only keep one feature from each cluster. **This is our case, TODO**

```python

```

Let's see how every model performs on inner_test_data and how good are they from business perspective

```python
model_df = pd.DataFrame(sorted(cv_scores.items(), key=lambda x: x[1]), columns=['model', 'cv_score'])
```

```python
model_df
```

```python
model_df.plot.barh(y='cv_score', x='model')
```

```python
classifiers = {"baseline_05_proba_clf": (baseline_05_proba_clf, ['product']),
              "baseline_zero_proba_clf": (baseline_zero_proba_clf, ['product']),
              "dummy_product_clf": (dummy_product_clf, ['product']),
              "pca_log_reg": (pca_log_reg, features),
              "xgb_tuned": (xgb_final, features),
              }
```

```python
test_scores = {name: roc_auc_score(inner_test_data['flag'], clf.predict_proba(inner_test_data[fts])[:, 1])
               for name, (clf, fts) in classifiers.items()}
```

```python
model_df = model_df.merge(pd.DataFrame(sorted(test_scores.items(), key=lambda x: x[1]), columns=['model', 'test_score']), on='model')
```

```python
model_df
```

```python
ax = sns.barplot(data=model_df.melt(id_vars='model', value_name='score', var_name=' '),
            x='score', y='model', hue=' ')
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')
ax.set_xbound(upper=0.9)
```

```python
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
```

```python
fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))
for name, (clf, fts) in classifiers.items():
    RocCurveDisplay.from_predictions(inner_test_data['flag'],
                                    clf.predict_proba(inner_test_data[fts])[:, 1],
                                    ax=ax_roc, name=name)
    DetCurveDisplay.from_predictions(inner_test_data['flag'],
                                    clf.predict_proba(inner_test_data[fts])[:, 1],
                                    ax=ax_det, name=name)
ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
ax_det.set_title("Detection Error Tradeoff (DET) curves")
ax_roc.grid(linestyle="--")
ax_det.grid(linestyle="--")
plt.legend()
```

**False Positive Rate**: percent of missed clients among good
* we want it as low as possible

**False Negative Rate**: percent of accepted clients among bad
* we want it as low as possible

**True Positive Rate**: percent of accepted clients among good
* we want it as high as possible

```python
fig = plt.figure()
ax = plt.gca()
for name, (clf, fts) in classifiers.items():
    DetCurveDisplay.from_predictions(inner_test_data['flag'],
                                    clf.predict_proba(inner_test_data[fts])[:, 1],
                                    ax=ax, name=name)
ax.set_title("Detection Error Tradeoff (DET) curves", fontdict = {'fontsize': 15})
ax.grid(linestyle="--")
ax.set_xlabel("Accepted clients among bad", fontdict = {'fontsize': 15})
ax.set_ylabel("Missed clients among good", fontdict = {'fontsize': 15})
plt.legend()
```

for the submission let's train the model on the whole dataset

```python
xgb_final.fit(X=data[features], y=data['flag'])
```

```python
default_proba = xgb_final.predict_proba(X=test_data[features])[:,1]
```

```python
submission = pd.DataFrame(data={'app_id': test_data.app_id, 'score': default_proba})
```

```python
submission.to_csv('../xgb_submission.csv', index=False)
```

Let's save the model

```python
dump(xgb_final, 'xgb_gpu.joblib')
```

```python
xgb_final_cpu.fit(X=data[features], y=data['flag'])
```

```python
dump(xgb_final_cpu, 'xgb_cpu.joblib')
```
