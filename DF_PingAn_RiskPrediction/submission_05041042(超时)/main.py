# -*- coding:utf-8 -*-
import data_helper_mulprocess, models
from data_helper_mulprocess import generate_xy, generate_x
from keras import losses
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import os
import time
from keras import metrics

# ---------submit------------

path_train = '/data/dm/train.csv'
path_test = '/data/dm/test.csv'
path_test_out = "model/"

# --------local test---------
# path_train = 'train.csv'  # 训练文件
# path_test = 'test.csv'  # 测试文件
# path_test_out = "model/"


CURRENT_PATH = os.getcwd()
BATCH_SIZE = 256
EPOCHES = 1000

train_dtypes = {'TERMINALNO': 'int32',
                'TIME': 'int32',
                'TRIP_ID': 'int16',
                'LONGITUDE': 'float32',
                'LATITUDE': 'float32',
                'DIRECTION': 'int16',
                'HEIGHT': 'float32',
                'SPEED': 'float32',
                'CALLSTATE': 'int8',
                'Y': 'float32'}

test_dtypes = {'TERMINALNO': 'int32',
                'TIME': 'int32',
                'TRIP_ID': 'int16',
                'LONGITUDE': 'float32',
                'LATITUDE': 'float32',
                'DIRECTION': 'int16',
                'HEIGHT': 'float32',
                'SPEED': 'float32',
                'CALLSTATE': 'int8'}


def process():
    #######
    train_data_path = os.path.join(CURRENT_PATH, 'data/train')
    test_data_path = os.path.join(CURRENT_PATH, 'data/test')
    ##########
    print('>>>[1].Preprocessing train data...')
    start_time = time.time()
    train_data_path = os.path.join(CURRENT_PATH, 'data/train')
    lens, params = data_helper_mulprocess.extract_feature(path_train, train_dtypes, train_data_path, target='Y')
    os.chdir(CURRENT_PATH)
    print('time1:', time.time() - start_time)

    print('>>>[2].Preprocessing test data...')
    start_time = time.time()
    test_data_path = os.path.join(CURRENT_PATH, 'data/test')
    _ = data_helper_mulprocess.extract_feature(path_test, test_dtypes, test_data_path, data_process_params=params, target=None)
    os.chdir(CURRENT_PATH)
    print('time2:', time.time() - start_time)

    print('>>>[3].Split data into the train and validate...')
    train_data, val_data = data_helper_mulprocess.train_test_split(train_data_path, test_ratio=0.25, random_state=9)
    target_file = os.path.join(train_data_path, 'targets.npy')
    train_user_feat_file = os.path.join(train_data_path, 'ufeatures.npy')

    max_len = int(np.percentile(lens, 85))
    # max_len =33 # int(np.percentile(lens, 80))
    print('max_len:', max_len)
    x_trip_dim = 29
    x_user_dim = 15

    print('>>>[4].Creating model...')
    model = models.create_cnn_dense((max_len, x_trip_dim), (x_user_dim,))

    model.compile(optimizer='adam', loss=losses.mse)
    print(model.summary())

    print('val steps:', len(val_data)//BATCH_SIZE)
    print('>>>[5].Training model...')
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    val_batch_size = int(len(val_data)/10)
    val_steps = data_helper_mulprocess.get_step(len(val_data), val_batch_size)
    train_steps = data_helper_mulprocess.get_step(len(train_data), BATCH_SIZE)
    num_input = 1
    start_time = time.clock()
    hist = model.fit_generator(generate_xy(train_data, train_user_feat_file, target_file, x_trip_dim, x_user_dim, batch_size=BATCH_SIZE, max_len=max_len, x_num=num_input),
                               steps_per_epoch=train_steps,
                               epochs=EPOCHES,
                               callbacks=[early_stop],
                               validation_data=generate_xy(val_data, train_user_feat_file, target_file, x_trip_dim, x_user_dim, batch_size=val_batch_size, max_len=max_len, x_num=num_input),
                               validation_steps=val_steps,
                               initial_epoch=0,
                               verbose=2)

    print('time:', time.clock() - start_time)

    print('>>>[6].Predicting...')
    pred_batch_size = 512
    id_preds = (np.load(os.path.join(test_data_path, 'targets.npy'))).astype(np.float32)
    test_data, _ = data_helper_mulprocess.train_test_split(test_data_path, test_ratio=0)
    test_user_feat_file = os.path.join(test_data_path, 'ufeatures.npy')

    test_data_len = len(test_data)
    pred_steps = data_helper_mulprocess.get_step(test_data_len, pred_batch_size)

    start_time = time.clock()
    predicts = model.predict_generator(generate_x(test_data, test_user_feat_file, x_trip_dim, x_user_dim, batch_size=pred_batch_size, max_len=max_len, x_num=num_input), steps=pred_steps)
    print('time:', time.clock() - start_time)

    print('>>>[7].Saving results...')
    predicts = np.array(predicts).reshape(-1)
    for idx, file_name in enumerate(test_data):
        user_idx = int(os.path.split(file_name)[1].split(r'.')[0])
        id_preds[user_idx, 1] = predicts[idx]

    pred_csv = pd.DataFrame(columns=['Id', 'Pred'])
    pred_csv['Id'] = id_preds[:, 0].astype(np.int64)
    pred_csv['Pred'] = id_preds[:, 1]
    pred_csv.to_csv(path_test_out+'pred.csv', index=False)


if __name__ == "__main__":
    print("****************** start **********************")
    process()
