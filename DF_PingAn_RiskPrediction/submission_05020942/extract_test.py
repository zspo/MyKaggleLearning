import pandas as pd
import numpy as np

from get_feature import *
from utils import *
import gc

def extrac_test_time_feature(df,column_name="TIME"):
    df["hour"] = pd.to_datetime(df[column_name]).dt.hour  # 时
    df["weekday"] = pd.to_datetime(df[column_name]).dt.weekday  # 星期
    print("feature of time is ok")


def get_test_feature(df,userId,feature_df):


    data = df.to_records(index=False)

    del df
    gc.collect()

    for userId in userId:

        user = data[data["TERMINALNO"] == userId]

        user.sort(order="second_time")

        user["delta_time"][1:] = user["second_time"][1:] - user["second_time"][:-1]
        ### 开车时间间隔大于一个小时就是不同次驾驶，不同行程
        diff_div = np.where(user["delta_time"] > 3600, 1, 0)


        days = np.unique(user["un_days"]).shape[0]
        ### 平均每天出车的次数
        freq = diff_div.sum() * 1.0 / days
        feature_df.loc[userId, "freq_drive"] = freq
        #### 开车的平均速度和最大速度
        feature_df.loc[userId, "aver_speed"] = user["SPEED"].mean()
        feature_df.loc[userId, "max_speed"] = user["SPEED"].max()

        feature_df.loc[userId, "speed_std"] = user["SPEED"].std()
        # feature_df.loc[userId, "speed_std"] = user.groupby(["TRIP_ID"])["SPEED"].std().max()

        feature_df.loc[userId, "speed_std"] = user["SPEED"].std()

        ### 以下应该是判断每一次驾驶是否是高峰期驾驶

        st_time = np.argwhere(diff_div==1)[:,0]

        st = 0
        peak_times = 0
        week_times = 0
        days_times = 0
        for end in st_time:
            h = user["hour"][st:end-1]
            if set([7,8,9,18,19,20]) & set(h):
                peak_times += 1
            w = user["weekday"][st:end-1]
            if set([5,6]) & set(w):
                week_times += 1
            if set(list(range(7,19))) & set(h):
                days_times += 1
            st = end

        days_times = days_times / diff_div.sum()
        feature_df.loc[userId, "in_day"] = days_times
        feature_df.loc[userId, "in_night"] = 1 - days_times

        ### 早高峰和晚高峰的影响
        # peak = np.where((user["hour"] > 7) & (user["hour"] < 9), 1,
        #                 np.where((user["hour"] > 18) & (user["hour"] < 21), 1, 0)).sum()

        peak_times = peak_times / diff_div.sum()
        feature_df.loc[userId, "peak_times"] = peak_times
        ### 平均海拔
        high = user["HEIGHT"].mean()
        feature_df.loc[userId, "HEIGHT"] = high

        week_times = week_times / diff_div.sum()
        feature_df.loc[userId, "in_week"] = 1 - week_times
        feature_df.loc[userId, "in_weekend"] = week_times

    del data
    gc.collect()
    print("extrac feature  is ok")

def extracTestFea(df):
    df["second_time"] = df["TIME"].values

    df["delta_time"] = 0

    df["TIME"] = df["TIME"].apply(unix_datatime)

    df["un_days"] = df["second_time"].apply(unix_data)
    userId = df["TERMINALNO"].unique()

    feature_df = pd.DataFrame()
    extrac_test_time_feature(df)
    get_test_feature(df, userId,feature_df)

    feature_df.index.name = "TERMINALNO"

    feature_df.to_csv("test_feature.csv", index=True)

