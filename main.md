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

# Erdos Institute Fall 2022 Project

## Credit risk assessment of individuals using time series models, along with the application of recursive neural networks (RNNs)

## by Team Oak

[Source link](https://ods.ai/competitions/dl-fintech-card-transactions) Get dataset and description there. Download each file separately. Do not use `Download all materials` button, the resulting zip file is corrupted.

[Link to submit test prediction](https://boosters.pro/championship/alfabattle2_sand/overview) I could not join the competition there at this moment, there's an issue with phone verification. But this is the only way to score test prediction. Maybe I can ask the organizer to provide us with the test ground truth...

[Best publicly shared solution](https://github.com/aizakharov94/AlfaBattle_task2)

[Organizer's github repo](https://github.com/smirnovevgeny/AlfaBattle2.0) Baselines, helpers and parquet loader from the organizer

Related [video](https://www.youtube.com/watch?v=yzV5ZQB850s) and [article](https://habr.com/ru/company/alfa/blog/551130/) from the organizer in Russian

**Tags**: `deep learning`, `neural networks`, `RNN`, `transformers`, `credit scoring`, `card transactions`


`Motivation`: A bank stores its clients card transaction data. One day a client comes in and asks for a loan. Should the bank lend to the client? Another client comes in and asks for a credit card. We can analyze the stored card transaction data and decide should we provide a financial product to a client or not.

`Dataset`: card transactions from single bank and multiple clients

`Train dataset` (train_transactions_contest): N days

`Test dataset` (test_transactions_contest): following K days

`Goal`: predict client's default on a financial product using their card transactions history

`Main metric`: AUC ROC

`Baseline models`: logistic regression, gradient boosting


### Stakeholders
Banks, other commercial institutions, government
### Company KPIs we pursuit to improve
Banks: TODO

Other commercial institutions: TODO

Government: TODO

### Timeline

October 28:
* Data cleaning + preprocessing
* Look for missing values and duplicates
* Basic data manipulation & preliminary feature engineering

November 4:
* Exploratory data analysis + visualizations
* Distributions of variables, looking for outliers, etc.
* Descriptive statistics

November 11:
* Written proposal of modeling approach
* Test linearity assumptions
* Dimensionality reductions (if necessary)
* Describe your planned modeling approach, based on the exploratory data analysis from the last two weeks (< 1 page, bullet points)
* Your modeling based on the KPIs you identified a few weeks ago

November 13:
* First pass of machine learning model or equivalent
* Preliminary results (visualizations and/or metrics)
* List of successes and pitfalls (so we can help you address them!)

November 20:
* Second pass of machine learning model or equivalent + GitHub
* Further results from last week (visualizations and/or metrics)
* List of successes and pitfalls (so we can help you address them!)

December 9: Final Project Due
* Presentation video and accompanying Google Slides 
* Presentations will be 5-min pre-recorded video based on Google Slides
* Slides should cover data gathering, exploratory data analysis, modeling approach, finalized results (visualizations and/or metrics), and implications / recommendations for stakeholders
* Annotated GitHub
* Nice markdown file
* For each notebook, include text so people can understand what you are doing and why 
* Please note that the GitHub repo should be public
* Executive Summary (1 page maximum)
* More than your actual code and modeling, companies want to see how your results will help them. This is the type of information they are looking for in job interviews
* Questions to address: How do your results impact the KPIs of the business? What recommendations do you have for the stakeholders? 


<!-- #region -->
###  File descriptions
1. train_transactions_contest: train data in parquet format, **20 GB as pd.DataFrame**
2. test_transactions_contest: test data in parquet format

    Features

* `app_id` - request id. Requests are enumerated wrt time, starting with the earliest date. `app` may stand for application for a financial product or applicant, so we may treat `app_id` as `client_id`. We need to check if `app_id` has a unique `product` (see train_target.csv and test_target_contest.csv).
* `amnt` - normalized transaction sum. 0.0 - missed data (?)
* `currency` - currency id
* `operation_type` - card operation type id
* `operation_type_group` - id of a group of a card operation type (e.g., debit card or credit card)
* `ecommerce_flag` - ecommerce flag
* `payment_system` - Payment system type identifier
* `income_flag` - Flag of write-off / deposit of funds to the card
* `mcc` - Unique identifier for the store type
* `country` - transaction country id
* `city` - transaction city id
* `mcc_category` - Transaction store category id
* `day_of_week` - transaction day of week
* `hour` - transaction hour
* `days_before` - day before issuance of credit
* `weekofyear` - transaction week of year
* `hour_diff` - time difference in hours since previous transaction of the same customer
* `transaction_number` - Sequence number of the customer's transaction


3. train_target.csv: train input target
* `app_id`
* `product` - financial product id
* `flag` - target, 1 - default happened for the product

4. test_target_contest.csv: test input (no target provided!)
* `app_id`
* `product`
5. sample_submission.csv: submission example file
* `app_id`
* `score` - model prediction
<!-- #endregion -->

```python

```
