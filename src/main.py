#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main program file
"""

from datetime import datetime
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import confusion_matrix
from sklearn.base import clone

from helpers import show_images
from helpers import plot_confusion_matrix
from data import Data
from classifiers import Classifiers

# ===========================================================
# GLOBAL
# ===========================================================

VERBOSE = 1
"""
 TEST_TYPE:    0=cross_validation,
               1=train/test 
               2=train(-4000)/test(+4000)
               3=train(-9000)/test(+9000)
"""
TEST_TYPE = 1
CLF = "NN1"

# ===========================================================
# DATA PREPARATION
# ===========================================================

# load data
data = Data()

data.normalise()
data.randomise()

# Test type
if TEST_TYPE == 0:
    data.cross_val()
elif TEST_TYPE == 2:
    data.move_to_test(move_num=4000)
elif TEST_TYPE == 3:
    data.move_to_test(move_num=9000)

# ===========================================================
# MODEL PREPARATION
# ===========================================================

clfs = Classifiers()

def compile_clf():
    """
    Returns a compiled {CLF}
    """
    compile_clf = getattr(clfs, f"compile_{CLF}")
    return compile_clf()

# ===========================================================
# RUN CLASSIFIER
# ===========================================================

acc_per_fold = []
loss_per_fold = []
y_truths = []
y_preds = []
x_test = []

FOLD_NUM = 1
model = compile_clf()
for idx, indices in enumerate(data.fold_indices):
    # check if not cross validation
    if not data.fold_indices[0]:
        fold_x_train = data.x_train
        fold_y_train = data.y_train
        fold_x_test = data.x_test
        fold_y_test = data.y_test
    elif idx != 0:
        print("===================================================")
        print(f"FOLD: {FOLD_NUM}")
        print("===================================================")
        fold_x_train = data.x_train[indices[0]]
        fold_x_test = data.x_train[indices[1]]
        fold_y_train = data.y_train[indices[0]]
        fold_y_test = data.y_train[indices[1]]
    else:
        continue

    model = compile_clf()

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

    FOLD_NUM = FOLD_NUM + 1

# ===========================================================
# SAVE MODEL
# ===========================================================

now = datetime.now()
timestamp = now.strftime("%d-%m-%Y--%H-%M-%s")
model.save(f"../models/{CLF}_{TEST_TYPE}_{timestamp}")

# ===========================================================
# VISUALISE
# ===========================================================

show_images(x_test[:10],
            predictions=y_preds[:10],
            ground_truths=y_truths[:10])

conf_matrix = confusion_matrix(y_truths, y_preds)
plot_confusion_matrix(conf_matrix)
