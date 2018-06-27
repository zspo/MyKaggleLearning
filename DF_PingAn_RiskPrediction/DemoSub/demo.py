#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-21 14:03:31
# @Author  : songpo.zhang (songpo.zhang@foxmail.com)
# @Link    : https://github.com/zspo
# @Version : $Id$

# import os
# import pandas as pd

# df = pd.DataFrame({'A': [0, 1, 2, 3, 4],
#     'B': [0,1,2,3,4],
#     'C': ['a', 'b', 'c', 'd', 'e']})

# print(df)
# df['D'] = df['B'].replace([-1,2,3,4],[0,1,1,0]).astype('int8')
# print(df)
#encoding:utf-8
"""
@project : Evaluation
@file : 20180514_02LightGBM
@author : Drxan
@create_time : 18-5-14 下午7:38
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import datetime
import os
import shutil
import math
from sklearn.decomposition import TruncatedSVD
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import copy
from sklearn import metrics
from math import radians, cos, sin, asin, sqrt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

"""
  去掉trip_id列,增加地理编码特征
"""
# ---------submit------------

path_train = 'train.csv'
path_test = 'test.csv'
path_test_out = "model/"


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


def haversine1(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000


def extract_user_features(term):
    term = term.loc[term['SPEED'] >= 0]
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

    term['period'] = 0
    term.loc[(term['hour'] >= 1) & (term['hour'] <= 3), 'period'] = 1
    term.loc[(term['hour'] >= 13) & (term['hour'] <= 15), 'period'] = 2
    term.loc[(term['hour'] >= 17) & (term['hour'] <= 19), 'period'] = 3

    # 行程量最多、最少的时段,各时段记录量均值、方差、最大、最小值
    features.extend(compute_num_statistics(term['period']))

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

    speed_period_group = term['SPEED'].groupby(term['period'])
    # 各时段速度均值
    speed_mean_period_sta = speed_period_group.mean()
    speed_mean_periods = np.zeros(4, dtype=np.float32)
    speed_mean_periods[speed_mean_period_sta.index] = speed_mean_period_sta
    features.extend(speed_mean_periods)
    del speed_mean_period_sta

    # 各时段速度标准差
    speed_std_period_sta = speed_period_group.std()
    speed_std_periods = np.zeros(4, dtype=np.float32)
    speed_std_periods[speed_std_period_sta.index] = speed_std_period_sta
    features.extend(speed_std_periods)
    del speed_std_period_sta

    speed_callstate_group = term['SPEED'].groupby(term['CALLSTATE'])
    # 各状态速度均值
    speed_mean_callstate_sta = speed_callstate_group.mean()
    speed_mean_callstates = np.zeros(5, dtype=np.float32)
    speed_mean_callstates[speed_mean_callstate_sta.index] = speed_mean_callstate_sta
    features.extend(speed_mean_callstates)
    del speed_mean_callstate_sta

    # 各状态速度标准差
    speed_std_callstate_sta = speed_callstate_group.std()
    speed_std_callstates = np.zeros(5, dtype=np.float32)
    speed_std_callstates[speed_std_callstate_sta.index] = speed_std_callstate_sta
    features.extend(speed_std_callstates)
    del speed_std_callstate_sta

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
    term = term.sort_values(by='TIME')
    startlong = term.iloc[0]['LONGITUDE']
    startlat = term.iloc[0]['LATITUDE']
    dis_start = haversine1(startlong, startlat, 113.9177317, 22.54334333)  # 距离某一点的距离

    # 将经纬度取整，拼接起来作为地块编码
    term['geo_code'] = term[['LONGITUDE', 'LATITUDE']].apply(lambda p: int(p[0])*100+int(p[1]), axis=1)
    geo_sta = term['geo_code'].value_counts()
    loc_most = geo_sta.index[0]
    geo_sta = geo_sta / term.shape[0]
    loc_most_freq = geo_sta.iloc[0]
    loc_entropy = ((-1)*geo_sta*np.log2(geo_sta)).sum()
    loc_num = len(geo_sta)


    features.extend([max_lon, min_lon, max_lat, min_lat, lon_ratio, lat_ratio, dis_start, loc_most, loc_most_freq,loc_entropy, loc_num])

    return features


features = ['record_num', 'busy_period', 'free_period', 'period_mean_num', 'period_std_num',
                 'period_max_num', 'period_min_num','busy_month', 'free_month', 'month_mean_num', 'month_std_num',
                 'month_max_num', 'month_min_num','busy_day', 'free_day', 'day_mean_num', 'day_std_num', 'day_max_num',
                 'day_min_num','busy_weekday', 'free_weekday','weekday_mean_num', 'weekday_std_num', 'weekday_max_num',
                 'weekday_min_num','busy_hour', 'free_hour', 'hour_mean_num', 'hour_std_num', 'hour_max_num',
            'hour_min_num', 'weekday0_num', 'weekday1_num', 'weekday2_num', 'weekday3_num', 'weekday4_num',
            'weekday5_num', 'weekday6_num', 'hour0_num', 'hour1_num', 'hour2_num', 'hour3_num', 'hour4_num',
            'hour5_num', 'hour6_num', 'hour7_num', 'hour8_num', 'hour9_num', 'hour10_num', 'hour11_num',
            'hour12_num', 'hour13_num', 'hour14_num', 'hour15_num', 'hour16_num', 'hour17_num', 'hour18_num',
            'hour19_num', 'hour20_num', 'hour21_num', 'hour22_num', 'hour23_num', 'speed_mean', 'speed_max',
            'speed_std', 'speed_median', 'hour0_speed_mean', 'hour1_speed_mean', 'hour2_speed_mean',
            'hour3_speed_mean', 'hour4_speed_mean', 'hour5_speed_mean', 'hour6_speed_mean', 'hour7_speed_mean',
            'hour8_speed_mean', 'hour9_speed_mean', 'hour10_speed_mean', 'hour11_speed_mean', 'hour12_speed_mean',
            'hour13_speed_mean', 'hour14_speed_mean', 'hour15_speed_mean', 'hour16_speed_mean', 'hour17_speed_mean',
            'hour18_speed_mean', 'hour19_speed_mean', 'hour20_speed_mean', 'hour21_speed_mean', 'hour22_speed_mean',
            'hour23_speed_mean', 'hour0_speed_std', 'hour1_speed_std', 'hour2_speed_std', 'hour3_speed_std', 'hour4_speed_std',
            'hour5_speed_std', 'hour6_speed_std', 'hour7_speed_std', 'hour8_speed_std', 'hour9_speed_std', 'hour10_speed_std', 'hour11_speed_std',
            'hour12_speed_std', 'hour13_speed_std', 'hour14_speed_std', 'hour15_speed_std', 'hour16_speed_std', 'hour17_speed_std', 'hour18_speed_std',
            'hour19_speed_std', 'hour20_speed_std', 'hour21_speed_std', 'hour22_speed_std', 'hour23_speed_std', 'period0_speed_mean',
            'period1_speed_mean', 'period2_speed_mean', 'period3_speed_mean', 'period0_speed_std', 'period1_speed_std', 'period2_speed_std', 'period3_speed_std',
            'callstate0_speed_mean', 'callstate1_speed_mean', 'callstate2_speed_mean', 'callstate3_speed_mean', 'callstate4_speed_mean',
            'callstate0_speed_std', 'callstate1_speed_std', 'callstate2_speed_std', 'callstate3_speed_std', 'callstate4_speed_std', 'unknow_direc', 'height_mean',
            'height_max', 'height_min', 'height_std', 'height_median', 'hour0_height_mean', 'hour1_height_mean', 'hour2_height_mean', 'hour3_height_mean',
            'hour4_height_mean', 'hour5_height_mean', 'hour6_height_mean', 'hour7_height_mean', 'hour8_height_mean', 'hour9_height_mean',
            'hour10_height_mean', 'hour11_height_mean', 'hour12_height_mean', 'hour13_height_mean', 'hour14_height_mean', 'hour15_height_mean',
            'hour16_height_mean', 'hour17_height_mean', 'hour18_height_mean', 'hour19_height_mean', 'hour20_height_mean', 'hour21_height_mean',
            'hour22_height_mean', 'hour23_height_mean', 'hour0_height_std', 'hour1_height_std', 'hour2_height_std', 'hour3_height_std',
            'hour4_height_std', 'hour5_height_std', 'hour6_height_std', 'hour7_height_std', 'hour8_height_std', 'hour9_height_std', 'hour10_height_std',
            'hour11_height_std', 'hour12_height_std', 'hour13_height_std', 'hour14_height_std', 'hour15_height_std', 'hour16_height_std',
            'hour17_height_std', 'hour18_height_std', 'hour19_height_std', 'hour20_height_std', 'hour21_height_std', 'hour22_height_std',
            'hour23_height_std', 'state0_ratio', 'state1_ratio', 'state2_ratio', 'state3_ratio', 'state4_ratio', 'max_lon', 'min_lon', 'max_lat',
            'min_lat', 'lon_ratio', 'lat_ratio', 'dis_start', 'loc_most', 'loc_most_freq', 'loc_entropy', 'loc_num']
cat_features = ['busy_period', 'free_period', 'busy_month', 'free_month', 'busy_day', 'free_day',
                'busy_weekday', 'free_weekday','busy_hour', 'free_hour', 'loc_most']


def get_feature_dummies(df_cat, cat_features, transformer=None):
    df_cat = df_cat.copy()
    result_df = pd.DataFrame(index=df_cat.index)
    if transformer is None:
        transformer = dict()
        for cf in cat_features:
            feat_transformer = dict()
            le = LabelEncoder()
            tmp_data = list(df_cat[cf])
            tmp_data.append(-1)
            le_feat = le.fit_transform(tmp_data)[:-1]
            feat_transformer['le'] = le
            ohe = OneHotEncoder(dtype =np.int8,sparse=False,handle_unknown='ignore')
            ohe_feat = ohe.fit_transform(le_feat.reshape((-1, 1)))[:, 1:]
            feat_transformer['ohe'] = ohe
            feat_names = [cf+'_'+str(i) for i in range(ohe_feat.shape[1])]
            result_df = pd.concat([result_df,pd.DataFrame(ohe_feat, columns=feat_names, index=df_cat.index)],axis=1)
            transformer[cf] = feat_transformer
    else:
        for cf in cat_features:
            df_cat.loc[~df_cat[cf].isin(transformer[cf]['le'].classes_), cf] = -1
            le_feat = transformer[cf]['le'].transform(df_cat[cf])
            ohe_feat = transformer[cf]['ohe'].transform(le_feat.reshape((-1, 1)))[:, 1:]
            feat_names = [cf+'_'+str(i) for i in range(ohe_feat.shape[1])]
            result_df = pd.concat([result_df,pd.DataFrame(ohe_feat, columns=feat_names, index=df_cat.index)], axis=1)
    return result_df, transformer

def process():
    conti_features = [c for c in features if c not in cat_features]
    print('[1]>> Extracting train features...')
    start = time.time()
    train = pd.read_csv(path_train, dtype=train_dtypes)
    train.drop('TRIP_ID', inplace=True, axis=1)
    train_x = []
    targets = []

    for uid in train['TERMINALNO'].unique():
        term = train.loc[train['TERMINALNO'] == uid]
        train_x.append(extract_user_features(term))
        targets.append(term['Y'].iloc[0])
    del train
    train_x = pd.DataFrame(train_x, columns=features, dtype=np.float32)
    train_x = train_x.fillna(-1)

    dummy_feats, transformers = get_feature_dummies(train_x[cat_features], cat_features, transformer=None)

    train_x = pd.concat([train_x[conti_features], dummy_feats], axis=1)

    gc.collect()
    end = time.time()
    print('time:{0}'.format((end-start)/60.0))

    print('[2]>> Fitting the  model...')
    model_ridge = Ridge()
    alpha_param = {'alpha':[1.5,1.9,2.2,2.7,3.0]}
    ridge_grid = GridSearchCV(estimator=model_ridge,param_grid=alpha_param,cv=5,n_jobs=-1)
    ridge_grid.fit(train_x,targets)
    print('The parameters of the best model are: ')
    print(ridge_grid.best_params_)

    # print('[3]>> Extracting test features...')
    # start = time.time()
    # test = pd.read_csv(path_test, dtype=test_dtypes)
    # test.drop('TRIP_ID', inplace=True, axis=1)
    # test_x = []
    # items = []
    # for uid in test['TERMINALNO'].unique():
    #     term = test.loc[test['TERMINALNO'] == uid]
    #     test_x.append(extract_user_features(term))
    #     items.append(uid)
    # del test
    # test_x = pd.DataFrame(test_x, columns=features, dtype=np.float32)
    # test_x = test_x.fillna(-1)
    # dummy_feats, transformers = get_feature_dummies(test_x[cat_features], cat_features, transformer=transformers)

    # test_x = pd.concat([test_x[conti_features], dummy_feats], axis=1)
    # gc.collect()

    # end = time.time()
    # print('time:{0}'.format((end - start) / 60.0))

    # print('[4] Predicting...')
    # start = time.time()
    # preds = ridge_grid.predict(test_x)
    # pred_csv = pd.DataFrame(columns=['Id', 'Pred'])
    # pred_csv['Id'] = items
    # pred_csv['Pred'] = preds
    # pred_csv.to_csv(path_test_out + 'pred.csv', index=False)
    # end = time.time()
    # print('time:{0}'.format((end - start) / 60.0))

if __name__ == '__main__':
    process()