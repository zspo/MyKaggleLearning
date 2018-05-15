# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
import math
import time

from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


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

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


def main():
    print('***************** Load Data *********************')
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)

    pre_label = label_process(train[["TERMINALNO","Y"]])

    train = train.drop('Y',axis=1)

    # train = conver_time(train)
    # test = conver_time(test)
    print('***************** Process Data *********************')
    feature_train = data_process(train)
    feature_train = feature_train.values
    X_TRAIN = feature_train[:,1:]
    Y_TRAIN = pre_label
    feature_test = data_process(test)
    feature_test = feature_test.values
    X_TEST = feature_test[:,1:]

    print('***************** Training Data *********************')
    # ----------------------- 线性模型 ------------------------
    # linreg = Ridge(normalize=True,max_iter=2000,solver="sparse_cg")
    # linreg.fit(feature_train[:,1:],pre_label)
    # predict_y = linreg.predict(feature_test[:,1:])

    lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             nthread = -1)
    model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

    stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

    stacked_averaged_models.fit(X_TRAIN, Y_TRAIN)
    stacked_pred = stacked_averaged_models.predict(X_TEST)

    model_xgb.fit(X_TRAIN, Y_TRAIN)
    xgb_pred = model_xgb.predict(X_TEST)

    model_lgb.fit(X_TRAIN, Y_TRAIN)
    lgb_pred = model_lgb.predict(X_TEST)

    ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
    # *********************************************************
    print('***************** Sub Data *********************')
    submission = pd.DataFrame(columns=['Id','Pred'])
    submission['Id'] = feature_test[:,0]
    submission['Pred'] = ensemble
    submission.to_csv(path_test_out+ 'sub.csv',index=False)

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    main()
    print('****************** end ************************')