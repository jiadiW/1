import numpy as np
import pandas as pd
import gc
from collections import defaultdict
from joblib import Parallel, delayed
N_Jobs = 32


def calc_ic_ir(cols, df):
    corr = defaultdict(dict)
    target_value = df.pivot(index= 'stock_id', columns = 'time_id', values = 'target')
    for col in cols:
        factor_value = df.pivot(index='stock_id', columns='time_id', values=col)
        IC = factor_value.corrwith(target_value, method='kendall', axis=0)  # 每个time_id求一个IC取平均
        mean_ic = np.nanmean(IC)
        ir = mean_ic / np.nanstd(IC)
        corr[col]['mean_ic'], corr[col]['ir'] = mean_ic, ir
        print(f'{col}, Mean IC: {mean_ic: .4f}, IR: {ir: .4f}')
    metric_df = pd.DataFrame.from_dict(corr).T
    metric_df['abs_mean_ic'] = np.abs(metric_df['mean_ic'])
    metric_df['abs_ir'] = np.abs(metric_df['ir'])
    return metric_df


def calc_ic_ir_parallel(df, exclude_cols = ['stock_id', 'time_id', 'date_id', 'seconds_in_bucket', 'target', 'GICS_SECTOR']):
    full_vars = df.columns.to_list()
    used_cols = [col for col in full_vars if (col not in exclude_cols) or not col.startswith('imbalance_buy_sell_flag')]
    step = int(len(used_cols) / N_Jobs) + 1
    params = [used_cols[i * step:(i + 1) * step] for i in range(N_Jobs)]
    p = Parallel(n_jobs=N_Jobs)(delayed(calc_ic_ir)(_param, df) for _param in params)
    metric_df = pd.DataFrame()
    for res in p:
        metric_df = pd.concat([metric_df, res[0]])
    gc.collect()
    metric_df = metric_df.sort_values(by = ['abs_ir', 'abs_mean_ic'], ascending = False)
    return metric_df


# df_train_feats['time_id'] = df_train['time_id']
# df_train_feats['target'] = df_train['target']
# df_train_feats['stock_id'] = df_train['stock_id']
# metric_df = calc_ic_ir_parallel(df_train_feats)

#df_train_feats columns 按照IC/IR排序
# df_train_feats2 = df_train_feats[metric_df.index]
# corr = df_train_feats2.corr()

def filter_cols_by_corr(df):
    """
    :param df: dataframe, columns ordered by IC/IR importance
    """
    corr = df.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= 0.99 or corr.iloc[i, j]<=-0.99:
                if columns[j]:
                    columns[j] = False

    feature_columns = corr.columns[columns].values
    drop_columns = corr.columns[columns == False].values
    print(feature_columns)
    print('-'*73)
    print(drop_columns)
