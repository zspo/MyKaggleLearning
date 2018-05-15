import time

import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool
NUM_PART = 10
NUM_CORES = multiprocessing.cpu_count()

format1 = '%Y-%m-%d %H:%M:%S'
format2 = '%Y-%m-%d'
def unix_datatime(unit):
    '''

    :param unit: 输入时间戳
    :return: 返回标准时间格式
    '''

    v = time.localtime(unit)
    d = time.strftime(format1,v)
    return d

def unix_data(unit):
    '''

    :param unit: 输入时间戳
    :return: 返回标准时间格式
    '''

    v = time.localtime(unit)
    d = time.strftime(format2,v)
    return d


def apply_day_night(x):
    if (x > 7) and (x < 19):
        return 1
    else:
        return 0

def unix_para(data):
    data = data.apply(unix_datatime)
    return data

def cout_delta_time(data):
    data.map(lambda x: x.total_seconds())
    return data

def parallelize_dataframe(df, func=unix_para):
    p = np.array_split(df, 10)
    pool = Pool(NUM_CORES)
    df = pd.concat(pool.map(func, p))
    pool.close()
    pool.join()
    return df



def sep_train_test(df):
    ind_test = ["_t" in str(i) for i in df.index.tolist()]
    ind_train = [not i for i in ind_test]

    test_df = df.loc[ind_test]
    train_df = df.loc[ind_train]
    return test_df,train_df

if __name__ == '__main__':
    pass


