import pandas as pd

CAT_COLUMNS = ['currency', 'operation_kind', 'card_type',
               'operation_type', 'operation_type_group', 'ecommerce_flag',
               'payment_system', 'income_flag', 'mcc', 'country', 'city',
               'mcc_category', 'day_of_week', 'hour','weekofyear']

NUMERIC_COLUMNS = ['days_before', 'hour_diff']

REAL_COLUMNS = ['amnt']


def __amnt_pivot_table_by_column_as_frame(frame, column, agg_funcs=None) -> pd.DataFrame:
    """
    Generates pivot table for `app_id` and a specified column by aggregating `amnt` column

    :param frame: pd.DataFrame containing card transactions
    :param column: column with keys to group by on the pivot table column
    :param agg_funcs: list of aggregation functions, default is ['sum', 'mean', 'count']
    :return: pd.DataFrame pivot table
    """
    if agg_funcs is None:
        agg_funcs = ['sum', 'mean', 'count']
    aggs = pd.pivot_table(frame, values='amnt',
                          index=['app_id'], columns=[column],
                          aggfunc={'amnt': agg_funcs},
                          fill_value=0)
    aggs.columns = [f'amnt_{col[0]}_{column}_{col[1]}' for col in aggs.columns.values]
    return aggs


def extract_basic_aggregations(transactions_frame: pd.DataFrame, cat_columns=None, agg_funcs=None) -> pd.DataFrame:
    """
    Extracts basic features from a card transaction dataframe

    :param transactions_frame: pd.DataFrame containing card transactions
    :param cat_columns: list of categorical columns for which we want to aggregate `amnt`, default is all
    :param agg_funcs: list of aggregation functions for cat_columns, default is
    ['sum', 'mean', 'count']
    :return: pd.DataFrame with extracted features
    """
    if not cat_columns:
        cat_columns = CAT_COLUMNS

    if not agg_funcs:
        agg_funcs = ['sum', 'mean', 'count']

    pivot_tables = []
    for col in cat_columns:
        pivot_tables.append(__amnt_pivot_table_by_column_as_frame(transactions_frame, column=col,
                                                                  agg_funcs=agg_funcs))
    pivot_tables = pd.concat(pivot_tables, axis=1)

    # we will also generate total statistics grouped by app_id
    aggs = {
        # transation amount
        'amnt': ['max', 'min', 'mean', 'median', 'sum', 'std'],
        # time difference between transactions
        'hour_diff': ['max', 'mean', 'median', 'var', 'std'],
        # days left before application at the moment when transaction took place
        'days_before': ['min', 'max', 'median']}

    numeric_stats = transactions_frame.groupby(['app_id']).agg(aggs)
    numeric_stats.columns = numeric_stats.columns.map('_'.join)

    return pd.concat([pivot_tables, numeric_stats], axis=1).reset_index()
