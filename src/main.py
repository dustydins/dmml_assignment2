#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main program file
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import confusion_matrix

from helpers import show_images
from helpers import plot_confusion_matrix
from data import Data

# ===========================================================
# GLOBAL
# ===========================================================

VERBOSE = 1

# ===========================================================
# DATA PREPARATION
# ===========================================================

data = Data()
data.normalise()
data.randomise()
#  data.cross_val()

# ===========================================================
# MODEL PREPARATION
# ===========================================================


def get_classifier():
    """
    Prepare model
    """
    clf = Sequential()
    clf.add(Dense(128, activation='relu'))
    clf.add(Dense(10, activation='softmax'))
    clf.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return clf


# ===========================================================
# RUN CLASSIFIER
# ===========================================================

acc_per_fold = []
loss_per_fold = []
y_truths = []
y_preds = []
x_test = []

FOLD_NO = 1
for idx, indices in enumerate(data.fold_indices):
    # check if not cross validation
    if not data.fold_indices[0]:
        fold_x_train = data.x_train
        fold_y_train = data.y_train
        fold_x_test = data.x_test
        fold_y_test = data.y_test
    elif idx != 0:
        print("===================================================")
        print(f"FOLD: {FOLD_NO}")
        print("===================================================")
        fold_x_train = data.x_train[indices[0]]
        fold_x_test = data.x_train[indices[1]]
        fold_y_train = data.y_train[indices[0]]
        fold_y_test = data.y_train[indices[1]]
    else:
        continue

    model = get_classifier()

    # fit model
    model.fit(x=fold_x_train, y=fold_y_train, epochs=10, verbose=VERBOSE)

    # evaluate
    test_loss, test_acc = model.evaluate(fold_x_test,
                                         fold_y_test,
                                         verbose=0)
    loss_per_fold.append(test_loss)
    acc_per_fold.append(test_acc * 100)

    # predictions
    fold_probs = model.predict(fold_x_test)
    fold_preds = [np.argmax(instance) for instance in fold_probs]
    for idy, truth in enumerate(fold_y_test):
        y_truths.append(truth)
        y_preds.append(fold_preds[idy])
        x_test.append(fold_x_test[idy])

    FOLD_NO = FOLD_NO + 1

# ===========================================================
# VISUALISE
# ===========================================================

show_images(x_test[:10],
            predictions=y_preds[:10],
            ground_truths=y_truths[:10])

conf_matrix = confusion_matrix(y_truths, y_preds)
plot_confusion_matrix(conf_matrix)
