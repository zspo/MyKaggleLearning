import numpy as np

import pandas as pd
import xgboost as xgb
#from xgboost import XGBClassifier
#from xgboost import XGBRegressor
from select_para import *
import warnings
import gc
warnings.filterwarnings("ignore")
class Model_class(object):

    def __init__(self,X_train=0, X_test=0, y_train=0, y_test=0,model="xgb"):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def train(self,train_data,label):
        label = label.ravel()
        #best_para = select_parameter(train_data, label)

        #self.model = XGBClassifier(**best_para)
        #self.model = XGBClassifier()
        #self.model.fit(train_data, label)
        params = {'objective': 'binary:logistic','max_depth': 10,'eval_metric': 'auc'}

        xgbtrain = xgb.DMatrix(train_data, label)

        self.model = xgb.train(params=params,dtrain=xgbtrain,verbose_eval=False)


    def predict(self,test_set):
        xgbtest = xgb.DMatrix(test_set)
        y_pred = self.model.predict(xgbtest)

        return y_pred

class Model_regre(object):

    def __init__(self, X_train=0, X_test=0, y_train=0, y_test=0, model="xgb"):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # if model == "xgb":
        #     self.reg = XGBRegressor()

    def train(self, X_train, y_train):
        params = {'objective': 'reg:linear','eval_metric': 'rmse', 'verbose_eval': False}
        xgbtrain = xgb.DMatrix(X_train, y_train)

        self.model = xgb.train(params=params, dtrain=xgbtrain, verbose_eval=False)



    def predict(self,test_set):
        xgbtest = xgb.DMatrix(test_set)
        y_pred = self.model.predict(xgbtest)
        return y_pred

