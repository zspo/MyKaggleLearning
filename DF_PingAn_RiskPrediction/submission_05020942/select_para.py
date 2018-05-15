import pprint
import sklearn.preprocessing as preprocessing
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import numpy as np
import time


skfold = StratifiedKFold(n_splits=10, shuffle=True)

def xgb_cv(clf, train_data, label, cv_folds=5, early_stopping_rounds=50, metric='map', \
           eval_metric=metrics.accuracy_score, is_print_f_i=False):
    train_X, test_X, train_y, test_y = train_test_split(train_data, label, test_size=0.1, random_state=2017)
    param = clf.get_xgb_params()
    train_data = xgb.DMatrix(train_X, train_y)
    train_test = xgb.DMatrix(test_X, test_y)
    cv_res = xgb.cv(param, train_data, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds, metrics=metric,
                    early_stopping_rounds=early_stopping_rounds)
    clf.set_params(n_estimators=cv_res.shape[0])

    clf.set_params(n_estimators=cv_res.shape[0])

    print("the best estimators %d" % cv_res.shape[0])
    clf.fit(train_X, train_y, eval_metric=metric)
    y_train_pred = clf.predict(train_X)

    print("the train set Accuracy is : %f" % metrics.accuracy_score(train_y, y_train_pred))
    y_test_pred = clf.predict(test_X)
    print('The test set Accuracy is: %f' % metrics.accuracy_score(test_y, y_test_pred))
    print('\n')

    feature_importances = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
    if is_print_f_i:
        print(feature_importances)
    return cv_res.shape[0]


def grid_search_para(train_data, label, best_para=0, grid_param=0, is_search_estimator=False, search_lr=0.1,
                     scoring='accuracy', search_estimators=10000, iid=False, cv=skfold):
    if not is_search_estimator:
        for key, value in grid_param.items():
            print('start GridSearchCV {} in range {}'.format(key, value))

        xgb_ = XGBClassifier(**best_para)

        grid_search = GridSearchCV(estimator=xgb_, param_grid=grid_param, scoring=scoring, iid=iid, cv=cv)

        grid_search.fit(train_data, label)

        best_para.update(grid_search.best_params_)

        print('the best parameter is ', grid_search.best_params_)
        print('the best score is %f' % grid_search.best_score_)


    else:
        xgb_ = XGBClassifier()
        if best_para == 0:
            best_para = xgb_.get_params()
        best_para['n_estimators'] = search_estimators
        best_para['learning_rate'] = search_lr
        xgb_ = XGBClassifier(**best_para)

        best_estimator = xgb_cv(xgb_, train_data, label)

        best_para['n_estimators'] = best_estimator

    return best_para


def select_parameter(train_data, label, test_size=0.1, scoring='accuracy'):
    best_para = grid_search_para(train_data, label, best_para=0, is_search_estimator=True)

    grid_param = {'max_depth': list(range(3, 10, 1)), 'min_child_weight': list(range(1, 12, 1))}

    best_para = grid_search_para(train_data, label, best_para, grid_param=grid_param)

    grid_param = {'gamma': [i / 10.0 for i in range(0, 5)]}
    best_para = grid_search_para(train_data, label, best_para, grid_param=grid_param)

    base_para = grid_search_para(train_data, label, best_para, is_search_estimator=True)

    grid_param = {'subsample': [i / 10.0 for i in range(6, 10)], 'colsample_bytree': [i / 10.0 for i in range(6, 10)]}
    best_para = grid_search_para(train_data, label, best_para, grid_param=grid_param)

    grid_param = {'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 1e-5, 1e-2, 0.1, 1, 100]}
    best_para = grid_search_para(train_data, label, best_para, grid_param=grid_param)

    best_para = grid_search_para(train_data, label, best_para, is_search_estimator=True, search_lr=0.1)

    pprint.pprint("The best parameter is \n {}".format(base_para))

    return best_para



if __name__ == "__main__":
    pass