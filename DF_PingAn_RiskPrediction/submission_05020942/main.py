# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
from get_feature import*
from model_xgb import *
from model_lgb import *
from processing_feature import *
from utils import *
import threading
from extract_train import *
from extract_test import *
path_train = "train.csv"  # 训练文件
path_test = "test.csv"  # 测试文件
import warnings
import gc
warnings.filterwarnings("ignore")

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

#COLUMNS = ['TERMINALNO', 'TIME', 'DIRECTION','HEIGHT', 'SPEED', 'CALLSTATE', 'Y']
def read_csv(file_path,isTrainData=True):
    """
    文件读取模块，头文件见columns.
    :return:
    """
    # for filename in os.listdir(path_train):
    if isTrainData:

        df = pd.read_csv(file_path,usecols=['TERMINALNO', 'TRIP_ID', 'TIME', 'DIRECTION','HEIGHT', 'SPEED', 'CALLSTATE', 'Y'])
    else:
        df = pd.read_csv(file_path,usecols=['TERMINALNO', 'TRIP_ID', 'TIME', 'DIRECTION','HEIGHT', 'SPEED', 'CALLSTATE'])

    #     df = pd.read_csv(file_path,usecols=['TERMINALNO', 'TIME', 'DIRECTION','HEIGHT', 'SPEED', 'CALLSTATE', 'Y'])
    # else:
    #     df =  pd.read_csv(file_path,usecols=['TERMINALNO', 'TIME', 'DIRECTION','HEIGHT', 'SPEED', 'CALLSTATE'])

    return df

def train_model():
    train_df = pd.read_csv(path_train)
    ### 删除异常值
    #train_df = train_df.loc[

    test_df = pd.read_csv(path_test)
    test_df["TERMINALNO"] = test_df["TERMINALNO"].map(lambda x: str(x) + "_t")


    train_thread = threading.Thread(target=extracTrainFea,args=(train_df,0.5), name='train')
    test_thread = threading.Thread(target=extracTestFea,args=(test_df,), name='test')

    train_thread.start()
    test_thread.start()

    train_thread.join()
    test_thread.join()

    del train_df,test_df
    gc.collect()

    test_df = pd.read_csv("test_feature.csv", index_col="TERMINALNO")


    train_df = pd.read_csv("train_feature.csv", index_col="TERMINALNO")

    
    print(test_df.head(5))
    print('*'*20)
    print(train_df.head(5))


    total_df = pd.concat([train_df, test_df], axis=0, ignore_index=False)
    # feature_name = ["dist", "deta_alt", "ave_alt", "max_alt", "min_alt",
    #                 "ave_speed", "max_speed", "std_speed", "acc_speed",
    #                 "wavy_terrain", "driv_time"]
    feature_name = ["max_speed","aver_speed","speed_std","HEIGHT"]

    norm_feature(total_df, feature_name)

    test_df, train_df = sep_train_test(total_df)
    train_label = train_df["Y"].copy()

    del total_df, train_df["Y"], test_df["Y"]
    gc.collect()

    clf = ModelLgb()
    test_df["Y"] = clf.train(train_df.values, train_label.values, test_df.values)

    y_pred = pd.DataFrame(test_df["Y"])

    del test_df,train_df,train_label,clf
    gc.collect()


    y_pred.index = y_pred.index.map(lambda x: x[:-2])

    y_pred.index.name = "Id"
    y_pred.columns = ["Pred"]

    y_pred.to_csv("model/result.csv")

def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return:
    """
    train_model()

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    st = time.time()
    process()
    print(time.time() - st)
    print("****************** end **********************")
