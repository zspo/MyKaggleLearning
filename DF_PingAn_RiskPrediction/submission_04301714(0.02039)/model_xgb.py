import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
import lightgbm as lgbm
import numpy as np
#from loss import evalerror
import xgboost as xgb
learning_rate = 0.1
num_leaves = 15
min_data_in_leaf = 2000
feature_fraction = 0.6
num_boost_round = 10000
NFOLDS = 5
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=2018)



class ModelXGB():
    def __init__(self):

        self.params = {"objective": "regression",
                               "boosting_type": "gbdt",
                               "learning_rate": learning_rate,
                               'metric': {'l2'},
                               "feature_fraction": feature_fraction,
                               "verbosity": 0,
                               "min_child_samples": 10,
                               "subsample": 0.9
                               }

    def train(self, train_data, train_label, test_data):
        xlf = xgb.XGBRegressor(seed=2018)
        xlf.fit(train_data, train_label, eval_metric='rmse', verbose=False)
        y = xlf.predict(test_data)
        return y