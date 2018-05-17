#encoding:utf-8

import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import datetime
import os
import shutil
from geopy.geocoders import Nominatim
import math
from sklearn.model_selection import RandomizedSearchCV
import gc
from sklearn.model_selection import train_test_split


# ---------submit------------

path_train = '/data/dm/train.csv'
path_test = '/data/dm/test.csv'
path_test_out = "model/"

# --------local test---------
# path_train = '/home/yw/study/Competition/pingan/train.csv'  # 训练文件
# path_test = '/home/yw/study/Competition/pingan/test.csv'  # 测试文件
# path_test_out = "model/"


train_dtypes = {'TERMINALNO': 'int32',
                'TIME': 'int32',
                'TRIP_ID': 'int16',
                'LONGITUDE': 'float32',
                'LATITUDE': 'float32',
                'DIRECTION': 'int16',
                'HEIGHT': 'float32',
                'SPEED': 'float32',
                'CALLSTATE': 'int8',
                'Y': 'float32'}

test_dtypes = {'TERMINALNO': 'int32',
                'TIME': 'int32',
                'TRIP_ID': 'int16',
                'LONGITUDE': 'float32',
                'LATITUDE': 'float32',
                'DIRECTION': 'int16',
                'HEIGHT': 'float32',
                'SPEED': 'float32',
                'CALLSTATE': 'int8'}


def compute_num_statistics(time_df):
    # 行程量最多的时段
    time_sta = time_df.value_counts()
    busy_time = time_sta.index[0]
    free_time = time_sta.index[-1]
    # 各时段录量均值、方差、最大、最小值
    time_mean_num = time_sta.mean()
    time_std_num = time_sta.std()
    time_max_num = time_sta.max()
    time_min_num = time_sta.min()
    return busy_time, free_time, time_mean_num, time_std_num, time_max_num, time_min_num


def compute_index_statistics():
    pass


def extract_user_features(term):
    term = term.copy()
    term = term.drop_duplicates()
    features = []

    # [1] 行程量统计量
    # 总的行程记录数量
    record_num = term.shape[0]
    features.append(record_num)

    term['month'] = term['TIME'].apply(lambda t: datetime.datetime.fromtimestamp(t).month)
    term['day'] = term['TIME'].apply(lambda t: datetime.datetime.fromtimestamp(t).day)
    term['weekday'] = term['TIME'].apply(lambda t: datetime.datetime.fromtimestamp(t).weekday())
    term['hour'] = term['TIME'].apply(lambda t: datetime.datetime.fromtimestamp(t).hour)
    # 行程量最多、最少的月份,各月记录量均值、方差、最大、最小值
    features.extend(compute_num_statistics(term['month']))

    # 行程最多、最少的天,各天记录量均值、方差、最大、最小值
    features.extend(compute_num_statistics(term['day']))

    # 行程最多、最少的周天,各周天记录量均值、方差、最大、最小值
    features.extend(compute_num_statistics(term['weekday']))

    # 行程最多、最少的时辰,各时辰记录量均值、方差、最大、最小值
    features.extend(compute_num_statistics(term['hour']))

    # 各周天的行程量
    weekday_sta = term['weekday'].value_counts()
    weekday_num = np.zeros(7, dtype=np.float32)
    weekday_num[weekday_sta.index] = weekday_sta
    features.extend(weekday_num)
    del weekday_sta, weekday_num

    # 各时辰的行程量
    hour_sta = term['hour'].value_counts()
    hour_num = np.zeros(24, dtype=np.float32)
    hour_num[hour_sta.index] = hour_sta
    features.extend(hour_num)
    del hour_sta, hour_num

    # [2] 速度特征
    speed_mean = term['SPEED'].mean()
    speed_max = term['SPEED'].max()
    speed_std = term['SPEED'].std()
    speed_median = term['SPEED'].median()
    features.extend([speed_mean, speed_max, speed_std, speed_median])

    speed_hour_group = term['SPEED'].groupby(term['hour'])
    # 各时辰速度均值
    speed_mean_hour_sta = speed_hour_group.mean()
    speed_mean_hours = np.zeros(24, dtype=np.float32)
    speed_mean_hours[speed_mean_hour_sta.index] = speed_mean_hour_sta
    features.extend(speed_mean_hours)
    del speed_mean_hour_sta

    # 各时辰速度标准差
    speed_std_hour_sta = speed_hour_group.std()
    speed_std_hours = np.zeros(24, dtype=np.float32)
    speed_std_hours[speed_std_hour_sta.index] = speed_std_hour_sta
    features.extend(speed_std_hours)
    del speed_std_hour_sta

    # [3] 方向特征
    unknow_direc = (term['DIRECTION'] < 0).sum() / record_num
    features.append(unknow_direc)

    # [4]海拔特征
    height_mean = term['HEIGHT'].mean()
    height_max = term['HEIGHT'].max()
    height_min = term['HEIGHT'].min()
    height_std = term['HEIGHT'].std()
    height_median = term['HEIGHT'].median()
    features.extend([height_mean, height_max, height_min, height_std, height_median])

    height_hour_group = term['HEIGHT'].groupby(term['hour'])
    # 各时辰海拔均值
    height_mean_hour_sta = height_hour_group.mean()
    height_mean_hours = np.zeros(24, dtype=np.float32)
    height_mean_hours[height_mean_hour_sta.index] = height_mean_hour_sta
    features.extend(height_mean_hours)
    del height_mean_hour_sta

    # 各时辰海拔标准差
    height_std_hour_sta = height_hour_group.std()
    height_std_hours = np.zeros(24, dtype=np.float32)
    height_std_hours[height_std_hour_sta.index] = height_std_hour_sta
    features.extend(height_std_hours)
    del height_std_hour_sta

    # [5] 状态特征
    state_ratio_sta = term['CALLSTATE'].value_counts() / record_num
    state_ratio = np.zeros(5, dtype=np.float32)
    state_ratio[state_ratio_sta.index] = state_ratio_sta
    features.extend(state_ratio)
    del state_ratio_sta

    # 经纬度特征
    max_lon = term['LONGITUDE'].max()
    min_lon = term['LONGITUDE'].min()
    max_lat = term['LATITUDE'].max()
    min_lat = term['LATITUDE'].min()
    time_dur = (term['TIME'].max() - term['TIME'].min()) / 3600.0+1.0
    lon_ratio = (max_lon - min_lon) / time_dur
    lat_ratio = (max_lat - min_lat) / time_dur
    features.extend([max_lon, min_lon, max_lat, min_lat, lon_ratio, lat_ratio])

    return features

feature_names=['record_num','busy_month', 'free_month', 'month_mean_num', 'month_std_num', 'month_max_num', 'month_min_num',
              'busy_day', 'free_day', 'day_mean_num', 'day_std_num', 'day_max_num', 'day_min_num','busy_weekday', 'free_weekday',
               'weekday_mean_num', 'weekday_std_num', 'weekday_max_num', 'weekday_min_num','busy_hour', 'free_hour', 'hour_mean_num',
               'hour_std_num', 'hour_max_num', 'hour_min_num','weekday0_num','weekday1_num','weekday2_num','weekday3_num','weekday4_num',
               'weekday5_num','weekday6_num','hour0_num', 'hour1_num', 'hour2_num', 'hour3_num', 'hour4_num', 'hour5_num', 'hour6_num',
               'hour7_num', 'hour8_num', 'hour9_num', 'hour10_num', 'hour11_num', 'hour12_num', 'hour13_num', 'hour14_num', 'hour15_num',
               'hour16_num', 'hour17_num', 'hour18_num', 'hour19_num', 'hour20_num', 'hour21_num', 'hour22_num', 'hour23_num','speed_mean',
              'speed_max','speed_std','speed_median','hour0_speed_mean', 'hour1_speed_mean', 'hour2_speed_mean', 'hour3_speed_mean',
               'hour4_speed_mean', 'hour5_speed_mean', 'hour6_speed_mean', 'hour7_speed_mean', 'hour8_speed_mean', 'hour9_speed_mean',
               'hour10_speed_mean', 'hour11_speed_mean', 'hour12_speed_mean', 'hour13_speed_mean', 'hour14_speed_mean', 'hour15_speed_mean',
               'hour16_speed_mean', 'hour17_speed_mean', 'hour18_speed_mean', 'hour19_speed_mean', 'hour20_speed_mean', 'hour21_speed_mean',
               'hour22_speed_mean', 'hour23_speed_mean','hour0_speed_std', 'hour1_speed_std', 'hour2_speed_std', 'hour3_speed_std', 'hour4_speed_std',
               'hour5_speed_std', 'hour6_speed_std', 'hour7_speed_std', 'hour8_speed_std', 'hour9_speed_std', 'hour10_speed_std', 'hour11_speed_std',
               'hour12_speed_std', 'hour13_speed_std', 'hour14_speed_std', 'hour15_speed_std', 'hour16_speed_std', 'hour17_speed_std', 'hour18_speed_std',
               'hour19_speed_std', 'hour20_speed_std', 'hour21_speed_std', 'hour22_speed_std', 'hour23_speed_std','unknow_direc','height_mean',
               'height_max','height_min','height_std','height_median','hour0_height_mean', 'hour1_height_mean', 'hour2_height_mean', 'hour3_height_mean',
               'hour4_height_mean', 'hour5_height_mean', 'hour6_height_mean', 'hour7_height_mean', 'hour8_height_mean', 'hour9_height_mean',
               'hour10_height_mean', 'hour11_height_mean', 'hour12_height_mean', 'hour13_height_mean', 'hour14_height_mean', 'hour15_height_mean',
               'hour16_height_mean', 'hour17_height_mean', 'hour18_height_mean', 'hour19_height_mean', 'hour20_height_mean', 'hour21_height_mean',
               'hour22_height_mean', 'hour23_height_mean','hour0_height_std', 'hour1_height_std', 'hour2_height_std', 'hour3_height_std',
               'hour4_height_std', 'hour5_height_std', 'hour6_height_std', 'hour7_height_std', 'hour8_height_std', 'hour9_height_std', 'hour10_height_std',
               'hour11_height_std', 'hour12_height_std', 'hour13_height_std', 'hour14_height_std', 'hour15_height_std', 'hour16_height_std',
               'hour17_height_std', 'hour18_height_std', 'hour19_height_std', 'hour20_height_std', 'hour21_height_std', 'hour22_height_std',
               'hour23_height_std','state0_ratio', 'state1_ratio', 'state2_ratio', 'state3_ratio', 'state4_ratio','max_lon','min_lon','max_lat',
               'min_lat','lon_ratio','lat_ratio']

lgb_params = {'boosting_type': 'gbdt',
              'colsample_bytree': 0.8,
              'learning_rate': 0.2,
              'max_bin': 168,
              # 'max_depth': 5,
              'min_child_samples': 6,
              'min_child_weight': 0.1,
              'min_split_gain': 0.17,
              'n_estimators': 100,
              'n_jobs': -1,
              'num_leaves': 9,
              'objective': 'regression',
              'random_state': 9,
              'reg_alpha': 0.005,
              'reg_lambda': 20,
              'subsample': 0.85,
              'subsample_freq': 1}

param_dist = {'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
              'max_bin': [50, 100, 168, 200, 255],
              # 'max_depth': 5,
              'min_child_samples': [1, 6, 10, 20, 30, 50, 80],
              'min_child_weight': [0.01, 0.1, 1, 2],
              'min_split_gain': [0.01, 0.17, 0.5],
              'num_leaves': [5, 10, 15, 20, 30, 40, 50, 60],
              'reg_alpha': [0.005, 0.01, 0.1, 0.5, 1],
              'reg_lambda': [0.1, 1, 5, 10, 20],
              'subsample': [0.7, 0.8, 0.9],
              'subsample_freq': [1, 3, 5, 8]}


def process():
    print('[1]>> Extracting train features...')
    start = time.time()
    train = pd.read_csv(path_train, dtype=train_dtypes)
    train_x = []
    targets = []

    for uid in train['TERMINALNO'].unique():
        term = train.loc[train['TERMINALNO'] == uid]
        train_x.append(extract_user_features(term))
        targets.append(term['Y'].iloc[0])
    del train
    train_x = pd.DataFrame(train_x, columns=feature_names, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    gc.collect()
    end = time.time()
    print('time:{0}'.format((end-start)/60.0))

    print('[2]>> Finding the best parameters...')
    start = time.time()
    lgbr = lgb.LGBMRegressor(**lgb_params)
    n_iter_search = 60
    random_search = RandomizedSearchCV(lgbr, param_distributions=param_dist, n_iter=n_iter_search,
                                       scoring='neg_mean_squared_error', n_jobs=-1, cv=2)

    random_search.fit(train_x, targets)

    print(random_search.best_params_)

    gc.collect()

    end = time.time()
    print('time:{0}'.format((end - start) / 60.0))

    feat_imp = pd.Series(random_search.best_estimator_.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(feat_imp.iloc[0:10])

    print('[4]>> Extracting test features...')
    start = time.time()
    test = pd.read_csv(path_test, dtype=test_dtypes)
    test_x = []
    items = []
    for uid in test['TERMINALNO'].unique():
        term = test.loc[test['TERMINALNO'] == uid]
        test_x.append(extract_user_features(term))
        items.append(uid)
    del test
    test_x = pd.DataFrame(test_x, columns=feature_names, dtype=np.float32)
    gc.collect()

    end = time.time()
    print('time:{0}'.format((end - start) / 60.0))

    print('[5] Predicting...')
    start = time.time()
    preds = random_search.predict(test_x)
    pred_csv = pd.DataFrame(columns=['Id', 'Pred'])
    pred_csv['Id'] = items
    pred_csv['Pred'] = preds
    pred_csv.to_csv(path_test_out + 'pred.csv', index=False)
    end = time.time()
    print('time:{0}'.format((end - start) / 60.0))

if __name__ == '__main__':
    process()