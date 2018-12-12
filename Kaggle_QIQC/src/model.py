#!/usr/bin/env python
# -*- coding = utf-8 -*-

"""
 @ Create Time: 2018/11/15
 @ Author: songpo.zhang
 @ Target:
"""

import math
import sys
import time
import os
import config
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers import Embedding, merge, Reshape, Activation, RepeatVector, Permute, Lambda, GlobalMaxPool1D, concatenate, multiply
from keras import initializers
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras.layers import Dense, Conv1D, MaxPooling1D, Input, Flatten, Dropout, Concatenate, LSTM, Bidirectional, GRU, GlobalAveragePooling1D
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.utils.conv_utils import convert_kernel
from keras.utils import to_categorical

from keras.utils import np_utils
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score


# 进行配置，使用70%的GPU
configs = tf.ConfigProto()
configs.gpu_options.per_process_gpu_memory_fraction = 0.85
session = tf.Session(config=configs)

# 设置session
set_session(session)

cur_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    f1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
    # print('precision: {}, recall: {}'.format(precision, recall))
    # print('f1: {}'.format(f1))
    return f1


# 写一个LossHistory类，保存loss和acc
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


class MyModel():

    def __init__(self,
                 GLOVE_EM=None, GOOGLE_NEWS_EM=None, PARAGRAM_EM=None, WIKI_NEWS_EM=None,
                 word_index=None,
                 data_len=None,
                 train_X=None,
                 train_y=None,
                 val_X=None,
                 val_y=None):

        # Model Hyperparameters
        self.hidden_dims = 512
        self.EMBEDDING_DIM = 300
        # Training parameters
        self.batch_size = 16
        self.num_epochs = 30

        self.GLOVE_EM = GLOVE_EM
        self.GOOGLE_NEWS_EM = GOOGLE_NEWS_EM
        self.PARAGRAM_EM = PARAGRAM_EM
        self.WIKI_NEWS_EM = WIKI_NEWS_EM

        self.word_index = word_index
        self.data_len = data_len

        self.train_X, self.train_y, self.val_X, self.val_y = train_X, train_y, val_X, val_y

        self.model = None

    def bulid_model(self):
        '''

        :return:
        '''

        print(config.get_current_time("building model ------"))

        # ----------- title local w2v ----------
        with tf.device('/gpu:%d' % (0)):
            tl_embedding_layer = Embedding(len(self.word_index) + 1,
                                           self.EMBEDDING_DIM,
                                           weights=[self.GLOVE_EM],
                                           input_length=self.data_len, trainable=True,
                                           embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2,
                                                                                         seed=None))
        tl_sequence_input = Input(shape=(self.data_len,), name="title_local_w2v_input")
        tl_embedded_sequences = tl_embedding_layer(tl_sequence_input)
        with tf.device('/gpu:%d' % (0)):
            tl_z_pos = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(tl_embedded_sequences)
            tl_z_neg = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(tl_embedded_sequences)
            tl_z_concat = concatenate([tl_z_pos, tl_embedded_sequences, tl_z_neg], axis=-1)

            tl_z = Dense(512, activation='tanh')(tl_z_concat)
            tl_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(tl_z)

        # ---------- title ai w2v ----------
        with tf.device('/gpu:%d' % (0)):
            ta_embedding_layer = Embedding(len(self.word_index) + 1,
                                           self.EMBEDDING_DIM,
                                           weights=[self.GOOGLE_NEWS_EM],
                                           input_length=self.data_len, trainable=True,
                                           embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2,
                                                                                         seed=None))
        ta_sequence_input = Input(shape=(self.data_len,), name="title_ai_w2v_input")
        ta_embedded_sequences = ta_embedding_layer(ta_sequence_input)
        with tf.device('/gpu:%d' % (0)):
            ta_z_pos = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(ta_embedded_sequences)
            ta_z_neg = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(ta_embedded_sequences)
            ta_z_concat = concatenate([ta_z_pos, ta_embedded_sequences, ta_z_neg], axis=-1)

            ta_z = Dense(512, activation='tanh')(ta_z_concat)
            ta_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(ta_z)

        # ----------- des local w2v ----------
        with tf.device('/gpu:%d' % (0)):
            dl_embedding_layer = Embedding(len(self.word_index) + 1,
                                           self.EMBEDDING_DIM,
                                           weights=[self.PARAGRAM_EM],
                                           input_length=self.data_len, trainable=True,
                                           embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2,
                                                                                         seed=None))
        dl_sequence_input = Input(shape=(self.data_len,), name="des_local_w2v_input")
        dl_embedded_sequences = dl_embedding_layer(dl_sequence_input)
        with tf.device('/gpu:%d' % (0)):
            dl_z_pos = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(dl_embedded_sequences)
            dl_z_neg = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(dl_embedded_sequences)
            dl_z_concat = concatenate([dl_z_pos, dl_embedded_sequences, dl_z_neg], axis=-1)

            dl_z = Dense(512, activation='tanh')(dl_z_concat)
            dl_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(dl_z)

        # ---------- des ai w2v ----------
        with tf.device('/gpu:%d' % (0)):
            da_embedding_layer = Embedding(len(self.word_index) + 1,
                                           self.EMBEDDING_DIM,
                                           weights=[self.WIKI_NEWS_EM],
                                           input_length=self.data_len, trainable=True,
                                           embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2,
                                                                                         seed=None))
        da_sequence_input = Input(shape=(self.data_len,), name="des_ai_w2v_input")
        da_embedded_sequences = da_embedding_layer(da_sequence_input)
        with tf.device('/gpu:%d' % (0)):
            da_z_pos = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(da_embedded_sequences)
            da_z_neg = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(da_embedded_sequences)
            da_z_concat = concatenate([da_z_pos, da_embedded_sequences, da_z_neg], axis=-1)

            da_z = Dense(512, activation='tanh')(da_z_concat)
            da_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(da_z)

        # ---------- att ----------
        concat_t_d = concatenate([tl_pool_rnn, ta_pool_rnn, dl_pool_rnn, da_pool_rnn], axis=-1)
        concat_t_d = Reshape((2, 512 * 2))(concat_t_d)

        attention = Dense(1, activation='tanh')(concat_t_d)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(512 * 2)(attention)
        attention = Permute([2, 1])(attention)

        sent_representation = multiply([concat_t_d, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(512 * 2,))(sent_representation)

        # ---------- merge_4models ----------
        model_final_ = Dense(2, activation='relu')(sent_representation)
        model_final_ = Dropout(0.5)(model_final_)
        model_final = Dense(2, activation='softmax')(model_final_)

        self.model = Model(inputs=[tl_sequence_input, ta_sequence_input, dl_sequence_input, da_sequence_input],
                           outputs=model_final)
        adam = optimizers.adam(lr=0.00001)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=[f1])

        print(self.model.summary())

    def save_model(self):

        self.model.save(config.MODEL_SAVE_DIR + "latest_twomodel_wordchar_" + str(cur_time) + '.h5')

    def trainModel(self):

        self.bulid_model()
        history = LossHistory()

        self.model.fit([self.train_X, self.train_X, self.train_X, self.train_X],
                        self.train_y,
                        shuffle=True,
                        validation_data=([self.val_X, self.val_X, self.val_X, self.val_X], self.val_y),
                        epochs=self.num_epochs,
                        batch_size=self.batch_size,
                        verbose=1,
                        callbacks=[history])

        history.loss_plot('epoch')
        self.save_model()

    def predModel(self, data=None):
    #
        test_X = data
    #
        self.model = load_model(config.MODEL_SAVE_DIR + "latest_twomodel_wordchar_" + str(cur_time) + '.h5')
        predlabel = self.model.predict([test_X, test_X, test_X, test_X], batch_size=128, verbose=1)

        return predlabel
