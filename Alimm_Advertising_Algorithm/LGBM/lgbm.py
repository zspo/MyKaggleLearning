import time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss


def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def get_time_feature(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    return data


if __name__ == "__main__":
    raw_train_data_path = r'../DataSet/TrainData/round1_ijcai_18_result_demo_20180301.txt'
    online = True  # 这里用来标记是 线下验证 还是 在线提交

    # 读取训练数据
    print('读取训练数据...')
    data = pd.read_csv(raw_train_data_path, sep=' ')
    data.drop_duplicates(inplace=True)

    # 提取时间特征
    print('提取时间特征...')
    data = get_time_feature(data)

    if online == False:
        train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
        test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
    elif online == True:
        train = data.copy()
        test = pd.read_csv(r'../DataSet/TestData/round1_ijcai_18_test_a_20180301.txt', sep=' ')
        test = get_time_feature(test)

    features = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
                'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                ]
    target = ['is_trade']

    if online == False:
        clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
        clf.fit(train[features], train[target], feature_name=features, categorical_feature=['user_gender_id'])
        test['lgb_predict'] = clf.predict_proba(test[features], )[:, 1]
        positive = test.loc[test.lgb_predict < 0.5]
        print(len(positive))
        print(log_loss(test[target], test['lgb_predict']))
    else:
        clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
        clf.fit(train[features], train[target],
                categorical_feature=['user_gender_id', ])
        test['predicted_score'] = clf.predict_proba(test[features])[:, 1]
        positive = test.loc[test.predicted_score >= 0.5]
        print(len(positive))

        test[['instance_id', 'predicted_score']].to_csv('lgbm_predict.csv', index=False, sep=' ')  # 保存在线提交结果