#!/usr/bin/env python
# -*- coding = utf-8 -*-

"""
 @ Create Time: 2018/11/15
 @ Author: songpo.zhang
 @ Target:
"""
import time


def get_current_time(str):
    '''
    get current time
    :param str:
    :return:
    '''
    current_time = time.strftime('%Y-%m-%d %H:%M:%S ====> '.format(time.localtime(time.time()))) + str
    return current_time


# ---------- 项目路径 ----------
PROJECTROOT_DIR = 'D:/_Zsp_Space/Python/Kaggle/Kaggle_QIQC'

# ---------- 原始数据路径 ----------
TRAIN_DIR = PROJECTROOT_DIR + '/datasets/train.csv'
TEST_DIR = PROJECTROOT_DIR + '/datasets/test.csv'
SAMPLE_SUB_DIR = PROJECTROOT_DIR + '/datasets/sample_submission.csv'

# ---------- 原始embeddings 路径 ----------
GLOVE_DIR = PROJECTROOT_DIR + '/datasets/embeddings/glove.840B.300d/glove.840B.300d.txt'
GOOGLE_NEWS_DIR = PROJECTROOT_DIR + '/datasets/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
PARAGRAM_DIR = PROJECTROOT_DIR + '/datasets/embeddings/paragram_300_sl999/paragram_300_sl999_1.txt'
WIKI_NEWS_DIR = PROJECTROOT_DIR + '/datasets/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

# ---------- 处理后数据路径 ----------
TRAIN_DATA_DIR = PROJECTROOT_DIR + '/inputdatas/train_data.pkl'
TRAIN_Y_DIR = PROJECTROOT_DIR + '/inputdatas/train_y.pkl'
VALID_DATA_DIR = PROJECTROOT_DIR + '/inputdatas/valid_data.pkl'
VALID_Y_DIR = PROJECTROOT_DIR + '/inputdatas/valid_y.pkl'
TEST_DATA_DIR = PROJECTROOT_DIR + '/inputdatas/test_data.pkl'

# ---------- embedding matrix 路径 ----------
GLOVE_EM_DIR = PROJECTROOT_DIR + '/embedding_matrixs/glove.pkl'
GOOGLE_NEWS_EM_DIR = PROJECTROOT_DIR + '/embedding_matrixs/GoogleNews.pkl'
PARAGRAM_EM_DIR = PROJECTROOT_DIR + '/embedding_matrixs/paragram.pkl'
WIKI_NEWS_EM_DIR = PROJECTROOT_DIR + '/embedding_matrixs/wiki-news.pkl'

# ---------- 模型保存路径 ----------
MODEL_SAVE_DIR = PROJECTROOT_DIR + '/model_save_here/'


# ---------- 日志保存路径 ----------
