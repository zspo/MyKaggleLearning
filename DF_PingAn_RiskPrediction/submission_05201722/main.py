# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
import math
import time
import gc
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
# path_train = "PINGAN-2018-train_demo.csv"  # 训练文件
# path_test = "PINGAN-2018-train_demo.csv"  # 测试文件

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

def time_convert(timestamp, type):
    #转换成localtime
    time_local = time.localtime(timestamp)
    if type == 'hour':
        #转换成新的时间格式(2016-05-05 20:28:54)
        # dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
        dt = time.strftime("%H", time_local)
    else:
        dt = time.strftime("%w", time_local)
    return int(dt)

def conver_time(data):
    data['hour'] = data['TIME'].apply(lambda x: time_convert(x, 'hour'))
    data['week'] = data['TIME'].apply(lambda x: time_convert(x, 'week'))
    data[['hour', 'week']] = data[['hour', 'week']].apply(pd.to_numeric)
    return data

def label_process(data):
    pre_label = data.drop_duplicates()
    return pre_label['Y'].values

def print_na(data):
    for each in data.columns.tolist():
        print(each + ' na num is %d' %sum(data[each].isna()))

def feature_process(data):
    set_data = set(data['TERMINALNO'])
    columns=['p_id','maxTime','phonerisk','dir_risk','height_risk',
             'speed_max','speed_mean','speed_std','speed_median','height_max','height_min','height_mean','height_std','height_median','sp_he_mean',
             'zao','wan','shenye','weekday_ratio','weekend_ratio',
             'weizhi_ratio','huchu_ratio','huru_ratio','liantong_ratio','duanlian_ratio',
             'callstate0_speed_mean','callstate1_speed_mean', 'callstate2_speed_mean','callstate3_speed_mean','callstate4_speed_mean',
             'callstate0_speed_std','callstate1_speed_std', 'callstate2_speed_std','callstate3_speed_std','callstate4_speed_std',
             'callstate0_height_mean','callstate1_height_mean', 'callstate2_height_mean','callstate3_height_mean','callstate4_height_mean',
             'callstate0_height_std','callstate1_height_std', 'callstate2_height_std','callstate3_height_std','callstate4_height_std',
             'callstate0_speed_diff_mean','callstate1_speed_diff_mean', 'callstate2_speed_diff_mean','callstate3_speed_diff_mean','callstate4_speed_diff_mean',
             'callstate0_speed_diff_std','callstate1_speed_diff_std', 'callstate2_speed_diff_std','callstate3_speed_diff_std','callstate4_speed_diff_std',
             'callstate0_height_diff_mean','callstate1_height_diff_mean', 'callstate2_height_diff_mean','callstate3_height_diff_mean','callstate4_height_diff_mean',
             'callstate0_height_diff_std','callstate1_height_diff_std', 'callstate2_height_diff_std','callstate3_height_diff_std','callstate4_height_diff_std',
            ]
    feature = pd.DataFrame(columns=columns)
    
    for p_id in set_data:
        tempData = data.loc[data['TERMINALNO'] == p_id]
        tempData = tempData.sort_values(["TIME"])

            ### 数据预处理----------------------------------------------------
        mean_value = tempData['SPEED'][tempData.SPEED != -1].mean()  # 去掉缺失值之后的均值
        tempData.SPEED = tempData['SPEED'].replace([-1], [mean_value])  # 均值速度填充缺失值

        tempTime = tempData["TIME"].iloc[0]
        tempSpeed = tempData["SPEED"].iloc[0]
        tempDir = tempData["DIRECTION"].iloc[0]
        tempHeight = tempData["HEIGHT"].iloc[0]

        maxTime = 0
        maxTimelist = []

        phonerisk = 0

        dir_risk = 0

        height_risk = 0
        zao=0
        wan=0
        shenye=0
        
        isWeekday=0
        isWeekend=0

        weizhi = 0
        huchu = 0
        huru = 0
        liantong = 0
        duanlian = 0

        for index, row in tempData.iterrows():
            
            hour = row['hour']
            if 7 <= hour <= 9:
                zao = 1
            elif 17 <= hour <= 19:
                wan = 1
            elif 0 <= hour < 7:
                shenye = 1

            if (row['week'] > 0 and row['week'] < 6):
                isWeekday += 1 
            else:
                isWeekend += 1

            if tempSpeed > 0 and row['CALLSTATE'] != 4:
                if row["CALLSTATE"] == 0:
                    phonerisk += math.exp(tempSpeed / 10) * 0.02
                else:
                    phonerisk += math.exp(tempSpeed / 10)
       
            if row["TIME"] - tempTime == 60:
                maxTime += 60
                tempTime = row["TIME"]

                dir_change = (min(abs(row["DIRECTION"] - tempDir), abs(360 + tempDir - row["DIRECTION"])) / 90.0)
                if tempSpeed != 0 and row["SPEED"] > 0:
                    dir_risk += math.pow((row["SPEED"] / 10), dir_change)

                height_risk += math.pow(abs(row["SPEED"] - tempSpeed) / 10,(abs(row["HEIGHT"] - tempHeight) / 100))
                
                tempHeight = row["HEIGHT"]

            elif row["TIME"] - tempTime > 60:
                maxTimelist.append(maxTime)
                maxTime = 0
                tempTime = row["TIME"]

                tempDir = row["DIRECTION"]
                tempHeight = row["HEIGHT"]
                tempSpeed = row["SPEED"]
                
            if row["CALLSTATE"] == 0:
                weizhi += 1
            elif row["CALLSTATE"] == 1:
                huchu += 1
            elif row["CALLSTATE"] == 2:
                huru += 1
            elif row["CALLSTATE"] == 3:
                liantong += 1
            elif row["CALLSTATE"] == 4:
                duanlian += 1

        total_Week = len(tempData['week'])
        weekday_ratio = isWeekday / float(total_Week)
        weekend_ratio = isWeekend / float(total_Week)

        speed_max = tempData["SPEED"].max()
        speed_mean = tempData["SPEED"].mean()
        speed_var = tempData["SPEED"].std()
        speed_median = tempData["SPEED"].median()
        
        height_max = tempData["HEIGHT"].max()
        height_min = tempData['HEIGHT'].min()
        height_mean = tempData["HEIGHT"].mean()
        height_var = tempData['HEIGHT'].std()
        height_median = tempData["HEIGHT"].median()

        sp_he_mean = speed_mean * height_mean

        maxTimelist.append(maxTime)
        maxTime = max(maxTimelist)
        
        total_callstate = len(tempData["CALLSTATE"])
        weizhi_ratio = weizhi / float(total_callstate)
        huchu_ratio = huchu / float(total_callstate)
        huru_ratio = huru / float(total_callstate)
        liantong_ratio = liantong / float(total_callstate)
        duanlian_ratio = duanlian / float(total_callstate)


        speed_callstate_group = tempData['SPEED'].groupby(tempData['CALLSTATE'])
        # 各状态的速度平均值
        speed_mean_callstate_sta = speed_callstate_group.mean()
        speed_mean_callstates = np.zeros(5, dtype=np.float32)
        speed_mean_callstates[speed_mean_callstate_sta.index] = speed_mean_callstate_sta
        del speed_mean_callstate_sta

        # 各状态的速标准差
        speed_std_callstate_sta = speed_callstate_group.std()
        speed_std_callstates = np.zeros(5, dtype=np.float32)
        speed_std_callstates[speed_std_callstate_sta.index] = speed_std_callstate_sta
        del speed_std_callstate_sta

        height_callstate_group = tempData['HEIGHT'].groupby(tempData['CALLSTATE'])
        # 各状态海拔均值
        height_mean_callstate_sta = height_callstate_group.mean()
        height_mean_callstates = np.zeros(5, dtype=np.float32)
        height_mean_callstates[height_mean_callstate_sta.index] = height_mean_callstate_sta
        del height_mean_callstate_sta

        # 各状态海拔标准差
        height_std_callstate_sta = height_callstate_group.std()
        height_std_callstates = np.zeros(5, dtype=np.float32)
        height_std_callstates[height_std_callstate_sta.index] = height_std_callstate_sta
        del height_std_callstate_sta


        tempData[['SPEED_DIF','HEIGHT_DIF']] = tempData[['SPEED','HEIGHT']].diff().astype('float16')
        # tempData['DIR_DIF'] = tempData['DIRECTION'].diff().astype('float32')
        tempData = tempData.fillna(0)

        speed_diff_callstate_group = tempData['SPEED_DIF'].groupby(tempData['CALLSTATE'])
        # 各状态的速度diff平均值
        speed_diff_mean_callstate_sta = speed_diff_callstate_group.mean()
        speed_diff_mean_callstates = np.zeros(5, dtype=np.float32)
        speed_diff_mean_callstates[speed_diff_mean_callstate_sta.index] = speed_diff_mean_callstate_sta
        del speed_diff_mean_callstate_sta

        # 各状态的速度diff标准差
        speed_diff_std_callstate_sta = speed_callstate_group.std()
        speed_diff_std_callstates = np.zeros(5, dtype=np.float32)
        speed_diff_std_callstates[speed_diff_std_callstate_sta.index] = speed_diff_std_callstate_sta
        del speed_diff_std_callstate_sta

        height_diff_callstate_group = tempData['HEIGHT_DIF'].groupby(tempData['CALLSTATE'])
        # 各状态的高度diff平均值
        height_diff_mean_callstate_sta = height_diff_callstate_group.mean()
        height_diff_mean_callstates = np.zeros(5, dtype=np.float32)
        height_diff_mean_callstates[height_diff_mean_callstate_sta.index] = height_diff_mean_callstate_sta
        del height_diff_mean_callstate_sta

        # 各状态的高度diff标准差
        height_diff_std_callstate_sta = height_callstate_group.std()
        height_diff_std_callstates = np.zeros(5, dtype=np.float32)
        height_diff_std_callstates[height_diff_std_callstate_sta.index] = height_diff_std_callstate_sta
        del height_diff_std_callstate_sta


        tempfeature = pd.DataFrame([[p_id,maxTime,phonerisk,dir_risk,height_risk,
                                    speed_max,speed_mean,speed_var,speed_median,height_max,height_min,height_mean,height_var,height_median,sp_he_mean,
                                    zao,wan,shenye,weekday_ratio,weekend_ratio,
                                    weizhi_ratio,huchu_ratio,huru_ratio,liantong_ratio,duanlian_ratio,
                                    speed_mean_callstates[0],speed_mean_callstates[1],speed_mean_callstates[2],speed_mean_callstates[3],speed_mean_callstates[4],
                                    speed_std_callstates[0],speed_std_callstates[1],speed_std_callstates[2],speed_std_callstates[3],speed_std_callstates[4],
                                    height_mean_callstates[0],height_mean_callstates[1],height_mean_callstates[2],height_mean_callstates[3],height_mean_callstates[4],
                                    height_std_callstates[0],height_std_callstates[1],height_std_callstates[2],height_std_callstates[3],height_std_callstates[4],
                                    speed_diff_mean_callstates[0],speed_diff_mean_callstates[1],speed_diff_mean_callstates[2],speed_diff_mean_callstates[3],speed_diff_mean_callstates[4],
                                    speed_diff_std_callstates[0],speed_diff_std_callstates[1],speed_diff_std_callstates[2],speed_diff_std_callstates[3],speed_diff_std_callstates[4],
                                    height_diff_mean_callstates[0],height_diff_mean_callstates[1],height_diff_mean_callstates[2],height_diff_mean_callstates[3],height_diff_mean_callstates[4],
                                    height_diff_std_callstates[0],height_diff_std_callstates[1],height_diff_std_callstates[2],height_diff_std_callstates[3],height_diff_std_callstates[4],
                                    ]],
                                    index=['0'],
                                    columns=columns)
        feature = feature.append(tempfeature,ignore_index=True)
    print_na(feature)
    # feature = feature.values
    return feature


def data_process(data):
    data = conver_time(data)
    feature_data = feature_process(data)
    feature_data = feature_data.fillna(0)
    return feature_data

def main():
    print('***************** Load Data *********************')
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)

    pre_label = label_process(train[["TERMINALNO","Y"]])

    train = train.drop('Y',axis=1)

    print('***************** Process Data *********************')

    feature_train = data_process(train)
    feature_train = feature_train.values

    feature_test = data_process(test)
    feature_test = feature_test.values
    
    train_x = feature_train[:,1:]
    targets = pre_label
    test_x = feature_test[:,1:]
    print('***************** Training Data *********************')
    # ----------------------- 线性模型 ------------------------
    # model_ridge = Ridge()
    # alpha_param = {'alpha':[1.5,1.7,1.9,2.1,2.2]}
    # ridge_grid = GridSearchCV(estimator=model_ridge,param_grid=alpha_param,cv=5,n_jobs=-1)
    # ridge_grid.fit(feature_train[:,1:],pre_label)
    # print('The parameters of the best model are: ')
    # print(ridge_grid.best_params_)
    # predict_y = ridge_grid.predict(feature_test[:,1:])
    new_feature_names = list(train_x.columns)
    x_train, x_val, y_train, y_val = train_test_split(train_x, targets, test_size=0.25, random_state=9)

    lgb_params = {'boosting_type': 'gbdt',
              'colsample_bytree': 0.8,
              'learning_rate': 0.05,
              'max_bin': 55,
              # 'max_depth': 5,
              'min_child_samples': 1,
              'min_child_weight': 0,
              'min_split_gain': 0,
              'n_estimators': 10000,
              'n_jobs': -1,
              'num_leaves': 32,
              'objective': 'regression',
              'random_state': 9,
              'reg_alpha': 0,
              'reg_lambda': 0,
              'subsample': 0.85,
              'subsample_freq': 1
              #'min_data_in_bin': 1,
              #'min_data': 1
              }
    lgbr = lgb.LGBMRegressor(**lgb_params)
    lgbr.fit(x_train, y_train, eval_metric='rmse', feature_name=new_feature_names, 
            eval_set=[(x_train, y_train), (x_val, y_val)],
         eval_names=['train', 'val'], verbose=0, early_stopping_rounds=100)

    feat_imp = pd.Series(lgbr.feature_importances_, index=new_feature_names).sort_values(ascending=False)
    print(feat_imp.iloc[0:10])

    predict_y = lgbr.predict(test_x)
    # *********************************************************
    print('***************** Sub Data *********************')
    submission = pd.DataFrame(columns=['Id','Pred'])
    submission['Id'] = feature_test[:,0]
    submission['Pred'] = predict_y
    submission.to_csv(path_test_out+ 'sub.csv',index=False)

if __name__ == "__main__":
    print("****************** start **********************")
    start = time.time()
    # 程序入口
    main()
    end = time.time()
    print('time:{0}'.format((end - start) / 60.0))
    print('****************** end ************************')