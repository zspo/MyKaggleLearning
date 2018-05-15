import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
import lightgbm as lgbm
import numpy as np
import xgboost as xgb
learning_rate = 0.1
num_leaves = 15
min_data_in_leaf = 2000
feature_fraction = 0.6
num_boost_round = 10000
NFOLDS = 5
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=2018)

class ModelLgb():
    def __init__(self,num=10):
        self.num = num

        self.params = {"objective": "regression",
                               "boosting_type": "gbdt",
                               "learning_rate": learning_rate,
                               'metric': {'l2'},
                               "feature_fraction": feature_fraction,
                               "verbosity": 0,
                               "min_child_samples": 10,
                               "subsample": 0.9,
                               "num_leaves":8
                               }

    def train(self, train_data, train_label, test_data):
        cv_pred = np.zeros(test_data.shape[0])
        kf = kfold.split(train_data, train_label)
        for sd in range(self.num):

            for i, (train_fold, validate) in enumerate(kf):
                X_train, X_validate, label_train, label_validate = \
                    train_data[train_fold, :], train_data[validate, :], train_label[train_fold], train_label[validate]
                xlf = xgb.XGBRegressor(seed=sd)
                xlf.fit(X_train, label_train, eval_metric='rmse', verbose=False,eval_set=[(X_validate, label_validate)], early_stopping_rounds=10)
                cv_pred += xlf.predict(test_data)
            cv_pred /= NFOLDS
        cv_pred /= self.num
        return cv_pred