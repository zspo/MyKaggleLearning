#!/usr/bin/env python
# -*- coding = utf-8 -*-

"""
 @ Create Time: 2017/12/22
 @ Author: songpo.zhang
 @ Target:
"""

import numpy as np
import pandas as pd
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils

train = pd.read_csv("../datasets/train.csv").values
test = pd.read_csv("../datasets/test.csv").values

# nb_epoch = 1
#
# batch_size = 128
# img_rows, img_cols = 28, 28

nb_filters_1 = 32 # 64
nb_filters_2 = 64 # 128
nb_filters_3 = 128 # 256
nb_conv = 3

X_train = train[:, 1:].reshape(train.shape[0], 28, 28, 1)
# X_train = train[:,1:].reshape(-1, 28, 28, 1)
X_train = X_train.astype(float)
X_train = X_train / 255.0

Y_train = kutils.to_categorical(train[:, 0])

cnn = models.Sequential()

cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", input_shape=(28, 28, 1), border_mode="same"))
cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", border_mode="same"))
cnn.add(conv.MaxPooling2D(strides=(2, 2)))

cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
cnn.add(conv.MaxPooling2D(strides=(2, 2)))

cnn.add(core.Flatten())
cnn.add(core.Dropout(0.2))
cnn.add(core.Dense(128, activation="relu"))
cnn.add(core.Dense(10, activation="softmax")) # y output=10 ;softmax for classificial

cnn.summary()
cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

cnn.fit(X_train, Y_train, batch_size=128, nb_epoch=1, verbose=1)

X_test = test.reshape(test.shape[0], 28, 28, 1)
X_test = X_test.astype(float)
X_test = X_test / 255.0

prediction = cnn.predict_classes(X_test)

submissions=pd.DataFrame({"ImageId": list(range(1, len(prediction)+1)),
                         "Label": prediction})
submissions.to_csv("keras_cnn.csv", index=False, header=True)
