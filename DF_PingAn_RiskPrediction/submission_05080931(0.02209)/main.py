# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
import math
import time

from sklearn.linear_model import Ridge
import xgboost as xgb

# path_train = "PINGAN-2018-train_demo.csv"  # 训练文件
# path_test = "PINGAN-2018-train_demo.csv"  # 测试文件

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def conver_time(data):
    data['Conver_TIME'] = data.TIME.apply(timestamp_datetime)
#     data['month'] = data.Conver_TIME.apply(lambda x: int(x[5:7]))
    data['hour'] = data.Conver_TIME.apply(lambda x: int(x[11:13]))
#     data = data.drop('TIME',axis=1)
#     data = data.drop('Conver_TIME',axis=1)
    return data

def label_process(data):
    pre_label = data.drop_duplicates()
    return pre_label['Y'].values

def feature_process(data):
    set_data = set(data['TERMINALNO'])
    columns=['id','maxTime', 'phonerisk', 'dir_risk', 'height_risk', 'speed_max', 'speed_mean', 'height_mean','zao','wan','shenye']
    feature = pd.DataFrame(columns=columns)
    for p_id in set_data:
        tempData = data.loc[data['TERMINALNO'] == p_id]
        tempData = tempData.sort_values(["TIME"])
    
        tempTime = tempData["TIME"].iloc[0]
        tempSpeed = tempData["SPEED"].iloc[0]
        tempDir = tempData["DIRECTION"].iloc[0]
        tempHeight = tempData["HEIGHT"].iloc[0]
        
        # 根据时间信息判断最长时间
        maxTime = 0
        maxTimelist = []

        # 用户行驶过程中，打电话危机上升
        phonerisk = 0

        # Direction 突变超过
        dir_risk = 0

        # Height 高度的危险值
        height_risk = 0
        zao=0
        # wu=0
        wan=0
        shenye=0
        for index, row in tempData.iterrows():
            hour = row['hour']
            # if hour>=3 and hour <=5:
            #     zao=1
            # elif hour>=6 and hour<=8:
            #     wu=1
            # elif hour>=9 and hour<=11:
            #     wan=1
            # else:
            #     shenye=1
            if 7 <= hour <= 9:
                zao = 1
            elif 17 <= hour <= 19:
                wan = 1
            elif 0 <= hour < 7:
                shenye = 1

            if tempSpeed > 0 and row['CALLSTATE'] != 4:
                if row["CALLSTATE"] == 0:
                    phonerisk += math.exp(tempSpeed / 10) * 0.02
                else:
                    phonerisk += math.exp(tempSpeed / 10)

        
            if row["TIME"] - tempTime == 60:
                maxTime += 60
                tempTime = row["TIME"]

                # 判断方向变化程度与具有车速之间的危险系数
                dir_change = (min(abs(row["DIRECTION"] - tempDir), abs(360 + tempDir - row["DIRECTION"])) / 90.0)
                if tempSpeed != 0 and row["SPEED"] > 0:
                    dir_risk += math.pow((row["SPEED"] / 10), dir_change)
                
                # 海拔变化大的情况下和速度的危险系数
                height_risk += math.pow(abs(row["SPEED"] - tempSpeed) / 10,(abs(row["HEIGHT"] - tempHeight) / 100))
                
                tempHeight = row["HEIGHT"]

            elif row["TIME"] - tempTime > 60:
                maxTimelist.append(maxTime)
                maxTime = 0
                tempTime = row["TIME"]

                tempDir = row["DIRECTION"]
                tempHeight = row["HEIGHT"]
                tempSpeed = row["SPEED"]
                
        speed_max = tempData["SPEED"].max()
        speed_mean = tempData["SPEED"].mean()

        height_mean = tempData["HEIGHT"].mean()

        maxTimelist.append(maxTime)
        maxTime = max(maxTimelist)
        
        tempfeature = pd.DataFrame({'id':p_id,
                                    'maxTime':maxTime,
                                    'phonerisk':phonerisk, 
                                    'dir_risk':dir_risk, 
                                    'height_risk':height_risk, 
                                    'speed_max':speed_max, 
                                    'speed_mean':speed_mean, 
                                    'height_mean':height_mean,
                                    'zao':zao,
                                    'wan':wan,
                                    'shenye':shenye},index=['0'],columns=columns)
        feature = feature.append(tempfeature,ignore_index=True)

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

    train = conver_time(train)
    test = conver_time(test)
    print('***************** Process Data *********************')
    feature_train = data_process(train)
    feature_train = feature_train.values
    X_TRAIN = feature_train[:,1:]
    Y_TRAIN = pre_label
    feature_test = data_process(test)
    feature_test = feature_test.values
    X_TEST = feature_test[:,1:]
    
    print('***************** Training Data *********************')
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             nthread = -1)
    model_xgb.fit(X_TRAIN, Y_TRAIN)
    xgb_pred = model_xgb.predict(X_TEST)
    # *********************************************************
    print('***************** Sub Data *********************')
    submission = pd.DataFrame(columns=['Id','Pred'])
    submission['Id'] = feature_test[:,0]
    submission['Pred'] = xgb_pred
    submission.to_csv(path_test_out+ 'sub.csv',index=False)

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    main()
    print('****************** end ************************')