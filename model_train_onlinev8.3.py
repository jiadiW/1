import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb  #3.3.3
import gc
import joblib  #1.3.1
import os
import logging
import time
import multiprocessing
import traceback

from sklearn.metrics import mean_absolute_error
from itertools import combinations
from lightgbm import log_evaluation, early_stopping
from warnings import simplefilter
from joblib import Parallel, delayed
from numba import njit, prange
from contextlib import contextmanager
from collections import defaultdict

warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


is_offline = False
is_train = True
is_infer = True
split_day = 435
debug = False
N_Jobs = min(multiprocessing.cpu_count(), 32)
is_special_purged_cv = False
feature_id = 8.3
"""
develop model--10 mins data
0. drop stock id
1. add sector
2. tune more global features. 
"""

df = pd.read_csv("train.csv")
df = df.dropna(subset=["target"])
df.reset_index(drop=True, inplace=True)
df_shape = df.shape


def setup_logger(path,name='logging'):
    path_ = '%s/' %(path)
    if not os.path.exists(path_): os.makedirs(path_)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fhlr = logging.FileHandler(path_ +f'/{name}.log')
    fhlr.setFormatter(formatter)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger,path_

logger_path = 'log'
if not os.path.exists(logger_path):
    os.makedirs(logger_path)
logger, logger_path = setup_logger(logger_path, name=f"IsOffline_{is_offline}_FeatureId_{feature_id}")
lgb.register_logger(logger)

@contextmanager
def timer(name: str):
    s = time.time()
    yield
    elapsed = time.time() - s
    print(f'[{name}] {elapsed: .3f}sec')


def reduce_mem_usage(df):

    for col in df.columns:
        col_type = df[col].dtype
        if (col_type != object) and (col_type != 'category'):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
    return df


@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))
    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            if mid_val - min_val < 1e-6: #add clip
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)
    return imbalance_features


@njit
def ma(arr: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    result = np.full(arr.shape, np.nan)  # Fill with nan
    if min_periods is None:
        min_periods = window
    for i in prange(arr.shape[0]):
        windowed_data = arr[max(0, i - window + 1) : i + 1]
        valid_count = np.sum(~np.isnan(windowed_data))
        if valid_count >= min_periods:
            result[i] = np.nanmean(windowed_data)  # Compute mean considering possible NaN values
    return result


def calc_ma(x, lookback=6):
    '''
    已经验证完全一致
    df['temp2'] = df.groupby(['date_id', 'stock_id'], group_keys=False)['wap'].apply(lambda x: calc_ma(x, 10))
    df['temp1'] = df.groupby(['date_id', 'stock_id'], group_keys=False)['wap'].rolling(10, 1).mean().droplevel([0, 1])
    '''
    x2 = x.to_numpy()
    return pd.Series(ma(x2, lookback), index=x.index)


@njit
def nanstd(arr: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    result = np.full(arr.shape, np.nan)  # Fill with nan
    if min_periods is None:
        min_periods = window
    for i in prange(arr.shape[0]):
        windowed_data = arr[max(0, i - window + 1) : i + 1]
        valid_count = np.sum(~np.isnan(windowed_data))
        if valid_count >= min_periods:
            result[i] = np.nanstd(windowed_data)  # Compute mean considering possible NaN values
    return result


def calc_nanstd(x, lookback = 6):
    x2 = x.to_numpy()
    return pd.Series(nanstd(x2, lookback), index=x.index)


@njit
def realized_volatility(arr: np.ndarray, window: int, min_periods: int = 1)-> np.ndarray:
    result = np.full(arr.shape, np.nan)  # Fill with nan
    if min_periods is None:
        min_periods = window
    for i in prange(arr.shape[0]):
        windowed_data = arr[max(0, i - window + 1) : i + 1]
        valid_count = np.sum(~np.isnan(windowed_data))
        if valid_count >= min_periods:
            result[i] = np.sqrt(np.nanmean(windowed_data**2))
    return result


def calc_realized_vol(x, lookback = 6):
    x2 = x.to_numpy()
    return pd.Series(realized_volatility(x2, lookback), index=x.index)


def calculate_triplet_imbalance_numba(price, df):
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]
    features_array = compute_triplet_imbalance(df_values, comb_indices)
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)
    return features


def imbalance_features(df):
    # Define lists of price and size-related column names
    df['GICS_SECTOR'] = df['stock_id'].map(sector_map)
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    df["volume"] = df['ask_size'] + df['bid_size']
    df["matched_imbalance"] = (df['imbalance_size'] - df['matched_size']) / (df['matched_size'] + df['imbalance_size'])
    df["size_imbalance"] = df['bid_size']/df['ask_size']
    # df["size_spread"] = (df['bid_size'] - df['ask_size'])/ (df['bid_size'] + df['ask_size']) size_spread=1-slope ask
    for c in combinations(prices, 2):
        if c not in [('near_price', 'ask_price'), ('ask_price', 'bid_price'), ('near_price', 'bid_price'), ('far_price', 'ask_price'),
                     ('far_price', 'bid_price'), ('reference_price', 'near_price'), ('reference_price', 'far_price')]:
            df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values

    #v3 features
    df["stock_weights"] = df["stock_id"].map(weights)
    df["weighted_wap"] = df["stock_weights"] * df["wap"]
    df['wap_momentum'] = df.groupby('stock_id')['weighted_wap'].pct_change(periods=6)

    #v5 features
    df['if_wap_notna'] = df['wap'].apply(lambda x: 0 if np.isnan(x) else 1)
    df['stock_weights_used'] = df["stock_weights"] * df['if_wap_notna']
    index_wap = (df.groupby(["date_id", "seconds_in_bucket"])['weighted_wap'].sum())/df.groupby(["date_id", "seconds_in_bucket"])['stock_weights_used'].sum()
    df['index_wap'] = df.set_index(["date_id", "seconds_in_bucket"]).index.map(index_wap.to_dict())
    df["wap_gap"] = df["wap"] - df['index_wap']
    df["reference_price_wap_ratio_imb"] = df["reference_price"]/df["wap"]

    for window in [1,3,5,10]:
        df[f'active_ret_lt{window*10}'] = df.groupby(["date_id", "stock_id"])["wap"].pct_change(window) \
                                          - df.groupby(["date_id", "stock_id"])["index_wap"].pct_change(window)
    for window in [3,5,10]:
        df[f'wap_gap_rrank_{window}'] = df.groupby(["stock_id"]
                                                  )['wap_gap'].rolling(window, min_periods=1).rank(pct = True).droplevel([0])

    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])
    df['slope_ask'] = df['ask_size'] / (df["volume"] / 2 + 1)
    df['swap_price'] = (df['bid_price'] * df['bid_size'] + df['ask_price'] * df['ask_size']) / df["volume"]

    #v3 features
    df['spread_depth_ratio'] = (df['ask_price'] - df['bid_price']) / (df['bid_size'] + df['ask_size'])
    df['mid_price_movement'] = ((df['ask_price'] + df['bid_price']) / 2).diff(periods=5).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df['relative_spread'] = (df['ask_price'] - df['bid_price']) / df['wap']

    # Calculate various statistical aggregation features
    for func in ["skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)

    for col in ['matched_size', 'imbalance_size', 'reference_price', 'wap_gap']:
        for window in [1, 3, 5, 10]:
            if (col, window) not in [('matched_size', 1)]:
                df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)

    for col in  ['matched_size', 'imbalance_size', 'reference_price', 'wap']:
        for window in [1, 3, 5, 10]:
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window).clip(-1e4, 1e4)

    # Calculate diff features for specific columns
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size', 'market_urgency', 'imbalance_momentum','size_imbalance',
                'weighted_wap', 'price_spread']:
        for window in [1, 3, 5, 10]:
            if (col, window) not in [('price_spread', 1)]:
                df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)

    for col in ['imbalance_buy_sell_flag']:
        for window in [1, 3, 5, 10]:
            df[f"{col}_shift_{window}"] = df.groupby("stock_id")[col].shift(window)  # 0,1
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)

    # v3 features
    for window in [3,5,10]:
        df[f'size_change_diff_{window}'] = df[f'bid_size_diff_{window}'] - df[f'ask_size_diff_{window}']
    df['mid_price*volume'] = df['mid_price_movement'] * df['volume']
    df['harmonic_imbalance'] = df.eval('2 / ((1 / bid_size) + (1 / ask_size))')

    df.drop(columns=['stock_weights', 'price_spread', "if_wap_notna", "stock_weights_used"
    ], inplace=True)
    return df


def other_features(df):
    df["seconds"] = df["seconds_in_bucket"] % 60
    df["minute"] = df["seconds_in_bucket"] // 60
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())
    return df


def generate_rolling_features(roll_cols, group, rolling_window):
    df = pd.DataFrame()
    new_cols = []
    for col in roll_cols:
        for window in rolling_window:
            print(f"{col}, {window}")
            if (col, window) not in [('wap', 5), ('bid_price', 3), ('bid_price', 5), ('ask_price', 5), ('bid_price', 10), ('ask_price', 3)]:
                df[f"{col}_roll{window}_mean"] = group[col].apply(lambda x: calc_ma(x, window))
                new_cols += [f"{col}_roll{window}_mean"]
            df[f"{col}_roll{window}_nanstd"] = group[col].apply(lambda x: calc_nanstd(x, window))
            new_cols += [ f"{col}_roll{window}_nanstd"]
    return df, new_cols


def generate_rolling_vol_features(roll_cols, group, rolling_window):
    df = pd.DataFrame()
    new_cols = []
    for col in roll_cols:
        for window in rolling_window:
            print(f"{col}, {window}")
            df[f"{col}_roll{window}_realizedvol"] = group[col].apply(lambda x: calc_realized_vol(x, window))
            new_cols += [f"{col}_roll{window}_realizedvol"]
    return df, new_cols


def calc_rolling_features_parallel(df):
    group = df.groupby(['stock_id', 'date_id'], group_keys=False)
    step = int(len(roll_mean_std_cols) / N_Jobs) + 1
    params = [roll_mean_std_cols[i * step:(i + 1) * step] for i in range(N_Jobs)]
    p = Parallel(n_jobs=N_Jobs)(delayed(generate_rolling_features)(_param, group, windows) for _param in params)
    for res in p:
        df_part, new_cols = res[0], res[1]
        df[new_cols] = df_part
    gc.collect()
    step_v = int(len(roll_vol_cols) / N_Jobs) + 1
    params = [roll_vol_cols[i * step_v:(i + 1) * step_v] for i in range(N_Jobs)]
    pv = Parallel(n_jobs=N_Jobs)(delayed(generate_rolling_vol_features)(_param, group, windows) for _param in params)
    for res in pv:
        df_part, new_cols = res[0], res[1]
        df[new_cols] = df_part
    gc.collect()
    return df



def generate_all_features(df):
    # Select relevant columns for feature generation
    cols = [c for c in df.columns if c not in ["row_id", "target", "time_id"]]
    df = df[cols]

    df = imbalance_features(df)
    df = other_features(df)
    df = calc_rolling_features_parallel(df)

    gc.collect()
    feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id", "stock_id"]]
    cat_features = ['GICS_SECTOR']
    for col in cat_features:
        df[col] = df[col].astype('category')
    return df[feature_name]


if is_offline:
    if debug:
        df_train = df[(df["date_id"] <= split_day)&(df['stock_id']<3)]
        df_test = df[(df["date_id"] > split_day)&(df['stock_id']<3)]
    else:
        df_train = df[df["date_id"] <= split_day]
        df_test = df[df["date_id"] > split_day]
    logger.info("Offline mode")
    logger.info(f"train : {df_train.shape}, valid : {df_test.shape}")
else:
    df_train = df
    logger.info("Online mode")


roll_mean_std_cols = ['market_urgency', 'slope_ask', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap']
for window in [3,5,10]:
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size']:
        roll_mean_std_cols += [f"{col}_diff_{window}"]

roll_vol_cols = ['reference_price_ret_1', 'reference_price_ret_3', 'reference_price_ret_5', 'reference_price_ret_10']
windows = [3,5,10]


weights = [
    0.004, 0.001, 0.002, 0.006, 0.004, 0.004, 0.002, 0.006, 0.006, 0.002, 0.002, 0.008,
    0.006, 0.002, 0.008, 0.006, 0.002, 0.006, 0.004, 0.002, 0.004, 0.001, 0.006, 0.004,
    0.002, 0.002, 0.004, 0.002, 0.004, 0.004, 0.001, 0.001, 0.002, 0.002, 0.006, 0.004,
    0.004, 0.004, 0.006, 0.002, 0.002, 0.04 , 0.002, 0.002, 0.004, 0.04 , 0.002, 0.001,
    0.006, 0.004, 0.004, 0.006, 0.001, 0.004, 0.004, 0.002, 0.006, 0.004, 0.006, 0.004,
    0.006, 0.004, 0.002, 0.001, 0.002, 0.004, 0.002, 0.008, 0.004, 0.004, 0.002, 0.004,
    0.006, 0.002, 0.004, 0.004, 0.002, 0.004, 0.004, 0.004, 0.001, 0.002, 0.002, 0.008,
    0.02 , 0.004, 0.006, 0.002, 0.02 , 0.002, 0.002, 0.006, 0.004, 0.002, 0.001, 0.02,
    0.006, 0.001, 0.002, 0.004, 0.001, 0.002, 0.006, 0.006, 0.004, 0.006, 0.001, 0.002,
    0.004, 0.006, 0.006, 0.001, 0.04 , 0.006, 0.002, 0.004, 0.002, 0.002, 0.006, 0.002,
    0.002, 0.004, 0.006, 0.006, 0.002, 0.002, 0.008, 0.006, 0.004, 0.002, 0.006, 0.002,
    0.004, 0.006, 0.002, 0.004, 0.001, 0.004, 0.002, 0.004, 0.008, 0.006, 0.008, 0.002,
    0.004, 0.002, 0.001, 0.004, 0.004, 0.004, 0.006, 0.008, 0.004, 0.001, 0.001, 0.002,
    0.006, 0.004, 0.001, 0.002, 0.006, 0.004, 0.006, 0.008, 0.002, 0.002, 0.004, 0.002,
    0.04 , 0.002, 0.002, 0.004, 0.002, 0.002, 0.006, 0.02 , 0.004, 0.002, 0.006, 0.02,
    0.001, 0.002, 0.006, 0.004, 0.006, 0.004, 0.004, 0.004, 0.004, 0.002, 0.004, 0.04,
    0.002, 0.008, 0.002, 0.004, 0.001, 0.004, 0.006, 0.004,
]
weights = {int(k):v for k,v in enumerate(weights)}
sector_df = pd.read_excel(r'D:\jiadi\optiver-trading-at-the-close\feature\GICS_map.xlsx', sheet_name="GICS")[['stock_id', 'GICS_SECTOR']].set_index('stock_id')
sector_map = sector_df.to_dict('dict')['GICS_SECTOR']
if is_train:
    global_stock_id_feats = {
        #max, min update online
        "bid_size_median": df_train.groupby("stock_id")["bid_size"].median(),
        "ask_size_median": df_train.groupby("stock_id")["ask_size"].median(),
        "bid_size_min": df_train.groupby("stock_id")["bid_size"].min(),
        "ask_size_min": df_train.groupby("stock_id")["ask_size"].min(),
        "ask_size_max": df_train.groupby("stock_id")["ask_size"].max(),
        "bid_size_std": df_train.groupby("stock_id")["bid_size"].std(),
        "ask_size_std": df_train.groupby("stock_id")["ask_size"].std(),
        "std_size": df_train.groupby("stock_id")["bid_size"].std() + df_train.groupby("stock_id")["ask_size"].std(),
        "ptp_size": df_train.groupby("stock_id")["bid_size"].max() - df_train.groupby("stock_id")["bid_size"].min(),
        "median_price": df_train.groupby("stock_id")["bid_price"].median() + df_train.groupby("stock_id")["ask_price"].median(),
        "std_price": df_train.groupby("stock_id")["bid_price"].std() + df_train.groupby("stock_id")["ask_price"].std(),
        "ptp_price": df_train.groupby("stock_id")["bid_price"].max() - df_train.groupby("stock_id")["ask_price"].min(),
    }


    if is_offline:
        df_train_feats = generate_all_features(df_train)
        logger.info("Build Train Feats Finished.")
        df_test_feats = generate_all_features(df_test)
        logger.info("Build Test Feats Finished.")
    else:
        df_train_feats = generate_all_features(df_train)
        logger.info("Build Online Train Feats Finished.")

#fix the n_estimators, don't use it.
# lgb_params = {
#     "objective": "mae",
#     "learning_rate": 0.1,
#     'metric': 'mae',
#     'bagging_freq': 1,
#     'seed': 42,
#     'extra_trees': True,
#     'feature_fraction': 0.5,
#     "num_leaves": 511,
#     'max_depth': 15,
#     'min_gain_to_split': 0,
#     "bagging_fraction": 0.98,
#     'min_sum_hessian_in_leaf': 0,
#     'lambda_l2': 0,
#     'lambda_l1': 0,
#     "importance_type": "gain",
#     "device": "gpu",
#     "n_estimators": 6000,
# }
lgb_params = {
    "objective": "mae",
    "n_estimators": 6000,
    "num_leaves": 256,
    "subsample": 0.6,
    "colsample_bytree": 0.8,
    "learning_rate": 0.01,
    'max_depth': 11,
    "n_jobs": -1,
    "device": "gpu",
    "verbosity": -1,
    "importance_type": "gain",
}

feature_save_path = 'feature'
if not os.path.exists(feature_save_path):
    os.makedirs(feature_save_path)
feature_name = list(df_train_feats.columns)
logger.info(f"Feature length = {len(feature_name)}")
joblib.dump(feature_name, f"{feature_save_path}/feats_name_{feature_id}")

# feats5 = joblib.load(f"{feature_save_path}/feats_name_{feature_id}")
# feats4 = joblib.load(f"{feature_save_path}/feats_name_{4}")
# new_feats = [fea for fea in feats5 if fea not in feats4]

num_folds = 5
model_num = num_folds + 1
if is_offline:
    fold_size = 435 // num_folds
else:
    fold_size = 480 // num_folds
gap = 5

model_save_path = 'model_from_start'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

date_ids = df_train['date_id'].values
#normal 5 fold CV
if not is_special_purged_cv:
    if is_offline:
        fold_set = {0: [0, 86], 1: [87, 173], 2: [174, 260], 3: [261, 347], 4: [348, 435]}
    else: #online mode
        if not debug:
            fold_set = {0: [0, 96], 1: [97, 193], 2: [194, 290], 3: [291, 387], 4: [388, 480]}

    data_index = {0:{'train_idx':[], 'valid_idx':[]}}
    for i in range(5):
        vmin, vmax = fold_set[i][0], fold_set[i][1]
        data_index[0]['train_idx'].append(df_train_feats.loc[(date_ids < vmin) | (date_ids > vmax)].index.to_list())
        data_index[0]['valid_idx'].append(df_train_feats.loc[(date_ids >= vmin) & (date_ids <= vmax)].index.to_list())
    # cv_fold0 = zip(train_idx0, validation_idx0)
    # cv_fold1 = zip(train_idx1, validation_idx1)

model_dict = defaultdict(list)
if is_offline:
    test_preds = np.zeros(shape = len(df_test))
for fold_id in range(num_folds):
    if not is_special_purged_cv:  # normal 5 fold cv
        train_indices, valid_indices = data_index[0]['train_idx'][fold_id], data_index[0]['valid_idx'][fold_id]
    else:  # special purged cv
        start = fold_id * fold_size
        end = start + fold_size
        if fold_id < num_folds - 1:  # No need to purge after the last fold
            purged_start = end - 2
            purged_end = end + gap + 2
            train_indices_f = (date_ids >= start) & (date_ids < purged_start) | (date_ids > purged_end)
        else:
            train_indices_f = (date_ids >= start) & (date_ids < end)
        valid_indices_f = (date_ids >= end) & (date_ids < end + fold_size)
        gc.collect()
    df_fold_train = df_train_feats.loc[train_indices, feature_name]
    df_fold_train_target = df_train['target'][train_indices]
    df_fold_valid = df_train_feats.loc[valid_indices, feature_name]
    df_fold_valid_target = df_train['target'][valid_indices]
    train_weight = np.log(df_train.loc[train_indices, 'time_id'] + 2)
    validation_weight = np.log(df_train.loc[valid_indices, 'time_id'] + 2)

    logger.info(f"Fold {fold_id} Training")
    extra_params = {'random_seed': [1, 42, 60, 66, 80, 128], 'learning_rate': [0.01, 0.015, 0.015, 0.015, 0.015, 0.02]}
    i = 0
    while i < 6:
        logger.info(f"i: {i}")
        try:
            lgb_params['random_seed'], lgb_params['learning_rate'] = extra_params['random_seed'][i],  extra_params['learning_rate'][i]
            gbm = lgb.LGBMRegressor(**lgb_params)
            gbm.fit(df_fold_train, df_fold_train_target, sample_weight=train_weight,
                    eval_set=[(df_fold_valid, df_fold_valid_target)], eval_sample_weight=[validation_weight],
                    eval_metric='mae', callbacks=[log_evaluation(period=100), early_stopping(stopping_rounds=100)])
            break
        except Exception as e:
            logger.info('模型训练报错：' + str(e))
            logger.info('traceback.format_exc(): \n%s' % (traceback.format_exc()))
            i += 1
            continue
    if i == 6:  # try 6 times all fail.
        continue
    pred_train_y = gbm.predict(df_fold_train)
    train_mae = mean_absolute_error(df_fold_train_target, pred_train_y)
    valid_mae = mean_absolute_error(df_fold_valid_target, gbm.predict(df_fold_valid))
    if is_offline:
        test_preds += gbm.predict(df_test_feats[feature_name])

    logger.info('Fold ID: %d Finished. train MAE: %.2f, valid MAE: %.2f ' % (fold_id, train_mae, valid_mae))
    one_fold_md = {'model': gbm, 'train_mae': train_mae, 'valid_mae': valid_mae}
    joblib.dump(one_fold_md,f"{model_save_path}/lgb_IsOffline_{is_offline}_featureID_{feature_id}_foldID_{fold_id}")
    model_dict['model'].append(gbm)
    model_dict['train_mae'].append(train_mae)
    model_dict['valid_mae'].append(valid_mae)
    del df_fold_train, df_fold_train_target, df_fold_valid, df_fold_valid_target
    gc.collect()

average_best_iteration = int(np.mean([model.best_iteration_ for model in model_dict['model']]))
final_model_params = lgb_params.copy()
final_model_params['n_estimators'] = average_best_iteration
logger.info(f"Training final model with average best iteration: {average_best_iteration}")
final_model = lgb.LGBMRegressor(**final_model_params)
final_model.fit(
    df_train_feats[feature_name], df_train['target'],
    callbacks=[lgb.callback.log_evaluation(period=100), ],
    sample_weight = np.log(df_train['time_id'] + 2)
)
pred_train_y = final_model.predict(df_train_feats[feature_name])
train_mae = mean_absolute_error(df_train['target'], pred_train_y)
model_dict['model'].append(final_model)
model_dict['train_mae'].append(train_mae)

if is_offline:
    test_preds += final_model.predict(df_test_feats[feature_name])
    test_mae = mean_absolute_error(df_test['target'], test_preds/model_num)
    model_dict['test_mae'].append(test_mae)
logger.info(
    f"Offline: {is_offline}. Average Validation MAE across all folds: {np.mean(model_dict['valid_mae'])}")
if is_offline:
    logger.info(f"Offline: {is_offline}.  Average Test MAE across all folds: {test_mae}")

joblib.dump(model_dict, f"{model_save_path}/lgb_IsOffline_{is_offline}_featureID_{feature_id}_5fold")