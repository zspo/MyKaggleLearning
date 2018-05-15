# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
import math
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,HuberRegressor,Ridge,Lasso,PassiveAggressiveRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

# path_train = "PINGAN-2018-train_demo.csv"  # 训练文件
# path_test = "PINGAN-2018-train_demo.csv"  # 测试文件

path_train = "train.csv"  # 训练文件
path_test = "test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

def timestamp_datetime(value):
    format = '%H'
    # value为传入的值为时间戳(整形)，如：1332888820
    value = time.localtime(value)
    ## 经过localtime转换后变成
    ## time.struct_time(tm_year=2012, tm_mon=3, tm_mday=28, tm_hour=6, tm_min=53, tm_sec=40, tm_wday=2, tm_yday=88, tm_isdst=0)
    # 最后再经过strftime函数转换为正常日期格式。
    dt = time.strftime(format, value)
    return int(dt)

def data_y_process(data):

    new_data = data.drop_duplicates()
    return np.array(new_data["Y"])


def data_x_process(data):
    """
    文件读取模块，头文件见columns.
    :return:
    """
    # for filename in os.listdir(path_train):
    setTE = set(data["TERMINALNO"])

    new_a = np.array([[0, 0, 0, 0, 0, 0, 0, 0,0,0,0]])

    for i in setTE:
        tempdata = data.loc[data["TERMINALNO"] == i]
        tempdata = tempdata.sort_values(["TIME"])

        # 初始化 时间，方向变化
        tempTime = tempdata["TIME"].iloc[0]
        tempSpeed = tempdata["SPEED"].iloc[0]
        tempdir = tempdata["DIRECTION"].iloc[0]
        tempheight = tempdata["HEIGHT"].iloc[0]

        # 根据时间信息判断最长时间
        maxTime = 0
        maxTimelist = []

        # 用户行驶过程中，打电话危机上升
        phonerisk = 0

        # Direction 突变超过
        dir_risk = 0

        # Height 高度的危险值
        height_risk = 0
        Zao = 0
        Wan = 0
        Sheye = 0

        for index, row in tempdata.iterrows():

            p_time = timestamp_datetime(row["TIME"])
            if 7 <= p_time <= 9:
                Zao = 1
            elif 17 <= p_time <= 19:
                Wan = 1
            elif 0 <= p_time < 7:
                Sheye = 1


            # 如果具有速度，且在打电话
            if tempSpeed > 0 and row["CALLSTATE"] != 4:

                # 人设打电话状态未知情况下，他的危机指数为 0.05
                if row["CALLSTATE"] == 0:
                    phonerisk += math.exp(tempSpeed / 10) * 0.02
                else:
                    phonerisk += math.exp(tempSpeed / 10)

            # 根据时间行驶判断
            if row["TIME"] - tempTime == 60:
                maxTime += 60
                tempTime = row["TIME"]

                # 判断方向变化程度与具有车速之间的危险系数
                dir_change = (min(abs(row["DIRECTION"] - tempdir), abs(360 + tempdir - row["DIRECTION"])) / 90.0)
                if tempSpeed != 0 and row["SPEED"] > 0:
                    dir_risk += math.pow((row["SPEED"] / 10), dir_change)

                # 海拔变化大的情况下和速度的危险系数
                height_risk += math.pow(abs(row["SPEED"] - tempSpeed) / 10,(abs(row["HEIGHT"] - tempheight) / 100))
                tempheight = row["HEIGHT"]

            elif row["TIME"] - tempTime > 60:
                maxTimelist.append(maxTime)
                maxTime = 0
                tempTime = row["TIME"]

                tempdir = row["DIRECTION"]
                tempheight = row["HEIGHT"]
                tempSpeed = row["SPEED"]

        speed_max = tempdata["SPEED"].max()
        speed_mean = tempdata["SPEED"].mean()

        height_mean = tempdata["HEIGHT"].mean()

        maxTimelist.append(maxTime)
        maxTime = max(maxTimelist)
        # print(i,maxTime,phonerisk,dir_risk,height_risk)

        new_a = np.row_stack((new_a, [i,maxTime, phonerisk, dir_risk, height_risk, speed_max, speed_mean, height_mean,Zao,Wan,Sheye]))

    return new_a[1:]



def process(xlist,ylist):
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
    pp = len(xlist)
    print(len(xlist),len(ylist))
    with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
        writer = csv.writer(outer)
        writer.writerow(["Id", "Pred"])
        for a in range(pp):
            writer.writerow([int(xlist[a]), ylist[a]])

def f(x,median_num):
    if x < median_num:
        return 0
    else:
        return x


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口

    train_data = pd.read_csv(path_train)
    test_data = pd.read_csv(path_test)

    train_data_x = data_x_process(train_data.iloc[:,:-1])
    train_data_y = data_y_process(train_data[["TERMINALNO","Y"]])

    test_data_x = data_x_process(test_data)
    # print(test_data_x)
    print(train_data_x[0,1:])
    print(train_data_y)
    print(test_data_x[0,1:])
    # ----------------------- 回归树模型 -----------------------
    # tree = DecisionTreeRegressor()
    # tree.fit(train_data_x[:,1:],train_data_y)
    # predict_y = tree.predict(test_data_x[:,1:])
    # print(predict_y)
    # *********************************************************

    # -------------------- 随机森林回归模型 ---------------------
    # regr = RandomForestRegressor(max_features="log2",max_depth=4,n_jobs=-1)
    # regr.fit(train_data_x[:,1:],train_data_y)
    # predict_y = regr.predict(test_data_x[:, 1:])
    # *********************************************************

    # ----------------------- 线性模型 ------------------------
    linreg = Ridge(normalize=True,max_iter=2000,solver="sparse_cg")
    linreg.fit(train_data_x[:,1:],train_data_y)
    predict_y = linreg.predict(test_data_x[:, 1:])
    # *********************************************************

    predic_x = test_data_x[:,0]

    # print(predic_x)
    process(predic_x,predict_y)
