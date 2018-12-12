#!/usr/bin/env python
# -*- coding = utf-8 -*-

"""
 @ Create Time: 2018/11/15
 @ Author: songpo.zhang
 @ Target:
"""

import config
from model import *
from data_preprocessing import *
import pickle

d = DataPreProcessing()
d.load_data()

train_X = pickle.load(open(config.TRAIN_DATA_DIR, 'rb'))
train_y = pickle.load(open(config.TRAIN_Y_DIR, 'rb'))
val_X = pickle.load(open(config.VALID_DATA_DIR, 'rb'))
val_y = pickle.load(open(config.VALID_Y_DIR, 'rb'))
test_X = pickle.load(open(config.TEST_DATA_DIR, 'rb'))

GLOVE_EM = pickle.load(open(config.GLOVE_EM_DIR, 'rb'))
GOOGLE_NEWS_EM = pickle.load(open(config.GOOGLE_NEWS_EM_DIR, 'rb'))
PARAGRAM_EM = pickle.load(open(config.PARAGRAM_EM_DIR, 'rb'))
WIKI_NEWS_EM = pickle.load(open(config.WIKI_NEWS_EM_DIR, 'rb'))

my_model = MyModel(GLOVE_EM=GLOVE_EM,
                   GOOGLE_NEWS_EM=GOOGLE_NEWS_EM,
                   PARAGRAM_EM=PARAGRAM_EM,
                   WIKI_NEWS_EM=WIKI_NEWS_EM,
                   word_index=d.word_index,
                   data_len=d.data_len,
                   train_X=train_X,
                   train_y=train_y,
                   val_X=val_X,
                   val_y=val_y
                   )

my_model.trainModel()
