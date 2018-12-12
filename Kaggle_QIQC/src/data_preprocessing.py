#!/usr/bin/env python
# -*- coding = utf-8 -*-

"""
 @ Create Time: 2018/11/15
 @ Author: songpo.zhang
 @ Target:
"""

import random
import time
import os
import pickle
import re
import gc

import jieba
import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import config


class DataPreProcessing():

    def __init__(self):
        self.word_index = None
        self.data_len = 70
        self.MAX_NB_WORDS = 50000
        self.EMBEDDING_DIM = 300

        self.GLOVE_EM = None
        self.GOOGLE_NEWS_EM = None
        self.PARAGRAM_EM = None
        self.WIKI_NEWS_EM = None

    def load_data(self):
        print(config.get_current_time("load row data"))
        train = pd.read_csv(config.TRAIN_DIR)
        test = pd.read_csv(config.TEST_DIR)

        ## split to train and val
        train_data, val_data = train_test_split(train, test_size=0.08, random_state=2018)
        print("Train data: {}, Valid data: {}, Test data: {}.".format(train.shape, val_data.shape, test.shape))

        ## fill up the missing values
        train_X = train_data["question_text"].fillna("_##_").values
        val_X = val_data["question_text"].fillna("_##_").values
        test_X = test["question_text"].fillna("_##_").values

        ## Tokenize the sentences
        tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)
        tokenizer.fit_on_texts(list(train_X))
        self.word_index = tokenizer.word_index
        train_X = tokenizer.texts_to_sequences(train_X)
        val_X = tokenizer.texts_to_sequences(val_X)
        test_X = tokenizer.texts_to_sequences(test_X)

        ## Pad the sentences
        train_X = pad_sequences(train_X, maxlen=self.data_len)
        val_X = pad_sequences(val_X, maxlen=self.data_len)
        test_X = pad_sequences(test_X, maxlen=self.data_len)

        ## Get the target values
        train_y = train_data['target'].values
        val_y = val_data['target'].values

        # shuffling the data
        np.random.seed(2018)
        trn_idx = np.random.permutation(len(train_X))
        val_idx = np.random.permutation(len(val_X))

        train_X = train_X[trn_idx]
        val_X = val_X[val_idx]
        train_y = train_y[trn_idx]
        val_y = val_y[val_idx]

        train_y = to_categorical(train_y, num_classes=2)
        val_y = to_categorical(val_y, num_classes=2)

        print(config.get_current_time("return data"))
        return train_X, train_y, val_X, val_y, test_X

    def load_glove_em_matrix(self):
        '''

        :return:
        '''
        print(config.get_current_time("load_glove_em_matrix"))
        embeddings_index = dict()

        embedding_max_value = 0
        embedding_min_value = 1

        with open(config.GLOVE_DIR, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(' ')
                if len(line) != 301:
                    continue

                coefs = np.asarray(line[1:], dtype='float32')

                if np.max(coefs) > embedding_max_value:
                    embedding_max_value = np.max(coefs)
                if np.min(coefs) < embedding_min_value:
                    embedding_min_value = np.min(coefs)

                embeddings_index[line[0]] = coefs

        print(config.get_current_time(('Found %s word vectors.' % len(embeddings_index))))

        self.GLOVE_EM = np.zeros((len(self.word_index) + 1, self.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.GLOVE_EM[i] = embedding_vector
            else:
                self.GLOVE_EM[i] = np.random.uniform(low=embedding_min_value, high=embedding_max_value, size=self.EMBEDDING_DIM)

    def load_google_news_em_matrix(self):

        print(config.get_current_time("load_google_news_em_matrix"))
        self.GOOGLE_NEWS_EM = np.zeros((len(self.word_index) + 1, self.EMBEDDING_DIM))
        model = gensim.models.KeyedVectors.load_word2vec_format(config.GOOGLE_NEWS_DIR, binary=True)

        for word, i in self.word_index.items():
            try:
                embedding_vector = model[word]
            except:
                embedding_vector = None

            if embedding_vector is not None:
                self.GOOGLE_NEWS_EM[i] = embedding_vector
            else:
                self.GOOGLE_NEWS_EM[i] = np.random.uniform(low=-0.0018054, high=0.047287, size=self.EMBEDDING_DIM)

    def load_paragram_em_matrix(self):
        '''

        :return:
        '''
        print(config.get_current_time("load_paragram_em_matrix"))
        embeddings_index = dict()

        embedding_max_value = 0
        embedding_min_value = 1

        with open(config.PARAGRAM_DIR, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(' ')
                if len(line) != 301:
                    continue

                coefs = np.asarray(line[1:], dtype='float32')

                if np.max(coefs) > embedding_max_value:
                    embedding_max_value = np.max(coefs)
                if np.min(coefs) < embedding_min_value:
                    embedding_min_value = np.min(coefs)

                embeddings_index[line[0]] = coefs

        print(config.get_current_time(('Found %s word vectors.' % len(embeddings_index))))

        self.PARAGRAM_EM = np.zeros((len(self.word_index) + 1, self.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.PARAGRAM_EM[i] = embedding_vector
            else:
                self.PARAGRAM_EM[i] = np.random.uniform(low=embedding_min_value, high=embedding_max_value, size=self.EMBEDDING_DIM)

    def load_wiki_news_em_matrix(self):
        '''

        :return:
        '''
        print(config.get_current_time("load_wiki_news_em_matrix"))
        embeddings_index = dict()

        embedding_max_value = 0
        embedding_min_value = 1

        with open(config.WIKI_NEWS_DIR, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(' ')
                if len(line) != 301:
                    continue

                coefs = np.asarray(line[1:], dtype='float32')

                if np.max(coefs) > embedding_max_value:
                    embedding_max_value = np.max(coefs)
                if np.min(coefs) < embedding_min_value:
                    embedding_min_value = np.min(coefs)

                embeddings_index[line[0]] = coefs

        print(config.get_current_time(('Found %s word vectors.' % len(embeddings_index))))

        self.WIKI_NEWS_EM = np.zeros((len(self.word_index) + 1, self.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.WIKI_NEWS_EM[i] = embedding_vector
            else:
                self.WIKI_NEWS_EM[i] = np.random.uniform(low=embedding_min_value, high=embedding_max_value, size=self.EMBEDDING_DIM)


if __name__ == '__main__':
    d = DataPreProcessing()
    train_X, train_y, val_X, val_y, test_X = d.load_data()
    pickle.dump(train_X, open(config.TRAIN_DATA_DIR, 'wb'))
    pickle.dump(train_y, open(config.TRAIN_Y_DIR, 'wb'))
    pickle.dump(val_X, open(config.VALID_DATA_DIR, 'wb'))
    pickle.dump(val_y, open(config.VALID_Y_DIR, 'wb'))
    pickle.dump(test_X, open(config.TEST_DATA_DIR, 'wb'))

    # print(train_X[:5])
    print(train_y[:3])

    d.load_glove_em_matrix()
    pickle.dump(d.GLOVE_EM, open(config.GLOVE_EM_DIR, 'wb'))
    del d.GLOVE_EM
    gc.collect()

    d.load_google_news_em_matrix()
    pickle.dump(d.GOOGLE_NEWS_EM, open(config.GOOGLE_NEWS_EM_DIR, 'wb'))
    del d.GOOGLE_NEWS_EM
    gc.collect()

    d.load_paragram_em_matrix()
    pickle.dump(d.PARAGRAM_EM, open(config.PARAGRAM_EM_DIR, 'wb'))
    del d.PARAGRAM_EM
    gc.collect()

    d.load_wiki_news_em_matrix()
    pickle.dump(d.WIKI_NEWS_EM, open(config.WIKI_NEWS_EM_DIR, 'wb'))
    del d.WIKI_NEWS_EM
    gc.collect()







