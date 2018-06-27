#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-28 21:37:38
# @Author  : songpo.zhang (songpo.zhang@foxmail.com)
# @Link    : https://github.com/zspo
# @Version : $Id$

import numpy as np
import pandas as pd
import random

def create_rand(c):
    global seed
    return random.uniform(0,seed)

def process_y(data):
    data0 = data[data.Pred==0]
    global seed
    seed = data[data.Pred!=0].min()
    y = data0[['Pred']].apply(create_rand, axis=1)
    data0.Pred = y
    new_data = pd.concat([data0,data[data.Pred!=0]])
    return new_data

pre = np.array([5,10,15,20,0,0,5,1,0,40])
print(pre)
df = pd.DataFrame(pre, columns=["Pred"])
print(df)
df_pro = process_y(df)
print(df_pro)