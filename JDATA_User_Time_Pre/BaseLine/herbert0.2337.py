# coding=utf-8
# @author: herbert-chen
# github: https://github.com/Herbert95/JDATA_
import pandas as pd
import numpy as np
import datetime
import copy
import xgboost as xgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# sku = pd.read_csv('../DataSets/jdata_sku_basic_info.csv').rename(columns={'\ufeffsku_id':'sku_id'},inplace=True)
# action = pd.read_csv('../DataSets/jdata_user_action.csv', parse_dates=['a_date']).rename(columns={'\ufeffuser_id':'user_id'},inplace=True)
# basic_info = pd.read_csv('../DataSets/jdata_user_basic_info.csv').rename(columns={'\ufeffuser_id':'user_id'},inplace=True)
# comment_score = pd.read_csv('../DataSets/jdata_user_comment_score.csv', parse_dates=['comment_create_tm']).rename(columns={'\ufeffuser_id':'user_id'},inplace=True)
# order = pd.read_csv('../DataSets/jdata_user_order.csv', parse_dates=['o_date']).rename(columns={'\ufeffuser_id':'user_id'},inplace=True)
sku = pd.read_csv('../DataSets/jdata_sku_basic_info.csv')
sku['sku_id'] = sku['\ufeffsku_id']
sku = sku.drop('\ufeffsku_id',axis=1)

action = pd.read_csv('../DataSets/jdata_user_action.csv', parse_dates=['a_date'])
action['user_id'] = action['\ufeffuser_id']
action = action.drop('\ufeffuser_id',axis=1)

basic_info = pd.read_csv('../DataSets/jdata_user_basic_info.csv')
basic_info['user_id'] = basic_info['\ufeffuser_id']
basic_info = basic_info.drop('\ufeffuser_id',axis=1)

comment_score = pd.read_csv('../DataSets/jdata_user_comment_score.csv', parse_dates=['comment_create_tm'])
comment_score['user_id'] = comment_score['\ufeffuser_id']
comment_score = comment_score.drop('\ufeffuser_id',axis=1)

order = pd.read_csv('../DataSets/jdata_user_order.csv', parse_dates=['o_date'])
order['user_id'] = order['\ufeffuser_id']
order = order.drop('\ufeffuser_id',axis=1)


order = pd.merge(order, sku, on='sku_id', how='left')
order = pd.merge(order, basic_info, on='user_id', how='left')
action = pd.merge(action, sku, how='left', on='sku_id')
order['month'] = order['o_date'].apply(lambda x: x.month)
action['month'] = action['a_date'].apply(lambda x: x.month)

# 评分函数
def score(pred, real):
    # pred: user_id, pre_date | real: user_id, o_date
    # wi与oi的定义与官网相同
    pred['pred_day'] = pd.to_datetime(pred['pred_date']).dt.day
    pred['index'] = np.arange(pred.shape[0]) + 1
    pred['wi'] = 1 / (1 + np.log(pred['index']))

    real['real_day'] = pd.to_datetime(real['o_date']).dt.day
    real['oi'] = 1

    compare = pd.merge(pred, real, how='left', on='user_id')
    compare.fillna(0, inplace=True)  # 实际上没有购买的用户，correct_for_S1列的值为nan，将其赋为0
    S1 = np.sum(compare['oi'] * compare['wi']) / np.sum(compare['wi'])

    compare_for_S2 = compare[compare['oi'] == 1]
    S2 = np.sum(10 / (10 + np.square(compare_for_S2['pred_day'] - compare_for_S2['real_day']))) / real.shape[0]

    S = 0.4 * S1 + 0.6 * S2
    print("S1=", S1, "| S2 ", S2)
    print("S =", S)

# 三月训练，四月验证；四月训练，五月验证
# def main(evaluate=False):
evaluate = False
train_month = 4 - int(evaluate)
if evaluate:
    train_action = action[action['month'] != 3][action['month'] != 4]
else:
    train_action = action[action['month'] != 4]
    
# 构建训练集：是首次购买日期，所以训练集只取训练月份最早购买的那一天
train_data = order[order['month'] == train_month][['user_id', 'o_date', 'cate']].sort_values(by=['user_id', 'o_date']).drop_duplicates()
train_data = train_data.drop(train_data[train_data[['user_id', 'cate']].duplicated()].index, axis=0)
train_data = pd.merge(train_data, basic_info, on='user_id', how='left')

# 增加训练集负样本
print('creating train dataset')
all_negative_train_data = pd.DataFrame()
len_of_original_train_data = train_data.shape[0]
for day_shift in [-2, -1, 1, 2]:
    negative_train_data = copy.deepcopy(train_data)
    negative_train_data['o_date'] = negative_train_data['o_date'].apply(lambda x: x + datetime.timedelta(days=day_shift))
    all_negative_train_data = pd.concat([all_negative_train_data, negative_train_data])
train_data['label'] = 1
all_negative_train_data['label'] = 0
all_negative_train_data['month'] = all_negative_train_data['o_date'].apply(lambda x: x.month)
all_negative_train_data = all_negative_train_data[all_negative_train_data['month'] == train_month]

# 删去错误的负样本（删除条件：若当天有购买，不应设其label为0）
wrong_data_index = pd.concat([train_data[['user_id', 'cate', 'o_date']],
                              all_negative_train_data[['user_id', 'cate', 'o_date']]]).reset_index(drop=True).duplicated()
wrong_data_index[:len_of_original_train_data] = False
train_data = pd.concat([train_data, all_negative_train_data]).reset_index(drop=True)
train_data = train_data.drop(train_data[wrong_data_index].index, axis=0)
train_data = train_data.sample(frac=1, random_state=777)
train_data['day'] = train_data['o_date'].apply(lambda x: x.day)

# 构建测试集
print('creating test dataset')
test_data_cate_101, test_data_cate_30 = basic_info[['user_id']], basic_info[['user_id']]
test_data_cate_101['cate'] = 101
test_data_cate_30['cate'] = 30
original_test_data = pd.concat([test_data_cate_101, test_data_cate_30], axis=0)
original_test_data = pd.merge(original_test_data, basic_info, on='user_id', how='left')
test_data = pd.DataFrame()
for i in tqdm(range({4:30, 5:31}[train_month+1])):
    test_data_per_day = copy.deepcopy(original_test_data)
    test_data_per_day['o_date'] = datetime.datetime(2017, train_month + 1, i + 1)
    test_data_per_day['day'] = i + 1
    test_data = pd.concat([test_data, test_data_per_day])

# 构建action特征
base_id = ['user_id', 'cate', 'o_date']

def add_action_feature(data, action_data, mode):
    first_day = {'train': datetime.datetime(2017, train_month, 1), 'test': datetime.datetime(2017, train_month + 1, 1)}[mode]
    current_month = {'train': train_month, 'test': train_month + 1}[mode]
    pre_data = pd.DataFrame()

    for action_index, action in enumerate(['look', 'star']):
        # 用户从上次浏览或关注到现在的时间
        latest_action_date, days_from_latest_action_date = 'latest_%s_date' % action, 'days_from_latest_%s_date' % action
        print('adding %s feature for %s data' % (days_from_latest_action_date, mode))
        action_data_i = action_data[action_data['a_type'] == action_index + 1]
        action_data_i = action_data_i.groupby(['user_id', 'cate']).a_date.agg({latest_action_date: max}).reset_index()
        data = pd.merge(data, action_data_i, how='left', on=['user_id', 'cate'])
        data[latest_action_date].fillna(datetime.datetime(2000, 1, 1), inplace=True)
        data_for_compute_action_days = data[[latest_action_date]].drop_duplicates()
        data_for_compute_action_days[days_from_latest_action_date] = data_for_compute_action_days[
            latest_action_date].apply(lambda x: (first_day - x).days)
        data = pd.merge(data, data_for_compute_action_days, how='left', on=latest_action_date)
        data[days_from_latest_action_date] = data[days_from_latest_action_date] + data['day'] - 1

    return data

train_data = add_action_feature(train_data, train_action, 'train')
test_data = add_action_feature(test_data, action, 'test')

# one hot
len_of_total_train_data = train_data.shape[0]
all_data = pd.concat([train_data, test_data])
one_hot_feature = ['cate', 'sex']
all_data = pd.get_dummies(all_data, columns=one_hot_feature)

# 处理缺失数据
all_data['age'] = all_data['age'].replace(-1, all_data['age'].mode()[0])
all_data.fillna(0, inplace=True)

train_data = all_data[:len_of_total_train_data]
test_data = all_data[len_of_total_train_data:]

# 筛选特征
cols_to_delete = ['user_id', 'sku_id', 'o_id', 'o_date', 'o_area', 'o_sku_num', 'month', 'latest_look_date', 'latest_star_date', 'day']
feature_list = [feature for feature in test_data.columns if feature not in cols_to_delete]

# xgboost参数
params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'gamma': 0.025,
          'min_child_weight': 6,
          'max_depth': 7,
          'lambda': 1,
          'subsample': 0.7,
          'colsample_bytree': 0.6,
          'eta': 0.1,
          'seed': 0,
          'max_delta_step': 0.5,
          'silent': 0,
          'scale_pos_weight': 4,
          }
          
# 输出
train_xgb = xgb.DMatrix(train_data[feature_list].values, train_data['label'])
test_xgb = xgb.DMatrix(test_data[feature_list].values)
model_xgb = xgb.train(params, train_xgb, num_boost_round=500)
test_y_xgb = model_xgb.predict(test_xgb)

result = test_data[['user_id']]
result['pred_date'] = test_data['o_date']
result['prob'] = test_y_xgb
result = result.sort_values(by=['prob', 'user_id'], ascending=False)
result = result.drop(result[result[['user_id']].duplicated()].index, axis=0)

if evaluate:
    real_result = order[order['month'] == train_month + 1][['user_id', 'o_date']].sort_values(by=['user_id', 'o_date']).drop_duplicates()
    real_result = real_result.drop(real_result[real_result[['user_id']].duplicated()].index, axis=0)
    score(result[:50000], real_result)
else:
    result[['user_id', 'pred_date']][:50000].to_csv('./result.csv', index=None)
