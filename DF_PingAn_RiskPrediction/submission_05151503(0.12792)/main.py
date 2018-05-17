# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
import math
import time

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

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
    columns=['p_id',
             'maxTime',
             'phonerisk',
             'dir_risk',
             'height_risk',
             'speed_max',
             'speed_mean',
             'speed_var',
             'height_max',
             'height_mean',
             'height_var',
             'sp_he_mean',
             'zao',
             'wan',
             'shenye',
             'weizhi_ratio',
             'huchu_ratio',
             'huru_ratio',
             'liantong_ratio',
             'duanlian_ratio'
            ]
    feature = pd.DataFrame(columns=columns)
    
    for p_id in set_data:
        tempData = data.loc[data['TERMINALNO'] == p_id]
        tempData = tempData.sort_values(["TIME"])

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

        speed_max = tempData["SPEED"].max()
        speed_mean = tempData["SPEED"].mean()
        speed_var = tempData["SPEED"].var()
        
        height_max = tempData["HEIGHT"].max()
        height_mean = tempData["HEIGHT"].mean()
        height_var = tempData['HEIGHT'].var()

        sp_he_mean = speed_mean * height_mean

        maxTimelist.append(maxTime)
        maxTime = max(maxTimelist)
        
        total_callstate = len(tempData["CALLSTATE"])
        weizhi_ratio = weizhi / float(total_callstate)
        huchu_ratio = huchu / float(total_callstate)
        huru_ratio = huru / float(total_callstate)
        liantong_ratio = liantong / float(total_callstate)
        duanlian_ratio = duanlian / float(total_callstate)
        
        tempfeature = pd.DataFrame({'p_id':p_id,
                                    'maxTime':maxTime,
                                    'phonerisk':phonerisk,
                                    'dir_risk':dir_risk,
                                    'height_risk':height_risk,
                                    'speed_max':speed_max,
                                    'speed_mean':speed_mean,
                                    'speed_var':speed_var,
                                    'height_max':height_max,
                                    'height_mean':height_mean,
                                    'height_var':height_var,
                                    'sp_he_mean':sp_he_mean,
                                    'zao':zao,
                                    'wan':wan,
                                    'shenye':shenye,
                                    'weizhi_ratio':weizhi_ratio,
                                    'huchu_ratio':huchu_ratio,
                                    'huru_ratio':huru_ratio,
                                    'liantong_ratio':liantong_ratio,
                                    'duanlian_ratio':duanlian_ratio
                                    },
                                    index=['0'],
                                    columns=columns)
        feature = feature.append(tempfeature,ignore_index=True)

        
    # feature = feature.values
    return feature

def data_process(data):
    data = conver_time(data)
    feature_data = feature_process(data)
    feature_data = feature_data.fillna(method='pad')
    return feature_data

def main():
    print('***************** Load Data *********************')
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)

    pre_label = label_process(train[["TERMINALNO","Y"]])

    train = train.drop('Y',axis=1)

    # train = conver_time(train)
    # test = conver_time(test)
    print('***************** Process Data *********************')
    # feature_train = feature_process(train)
    # feature_test = feature_process(test)

    feature_train = data_process(train)
    feature_train = feature_train.values

    feature_test = data_process(test)
    feature_test = feature_test.values
    

    print('***************** Training Data *********************')
    # ----------------------- 线性模型 ------------------------
    model_ridge = Ridge()
    alpha_param = {'alpha':[1.5,1.7,1.9,2.1,2.2]}
    ridge_grid = GridSearchCV(estimator=model_ridge,param_grid=alpha_param,cv=5,n_jobs=-1)
    ridge_grid.fit(feature_train[:,1:],pre_label)
    print('The parameters of the best model are: ')
    print(ridge_grid.best_params_)
    predict_y = ridge_grid.predict(feature_test[:,1:])
    # *********************************************************
    print('***************** Sub Data *********************')
    submission = pd.DataFrame(columns=['Id','Pred'])
    submission['Id'] = feature_test[:,0]
    submission['Pred'] = predict_y
    submission.to_csv(path_test_out+ 'sub.csv',index=False)

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    main()
    print('****************** end ************************')