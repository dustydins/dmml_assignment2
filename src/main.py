#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main program file
"""

import os
import random
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
from termcolor import colored

from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import confusion_matrix
from sklearn.base import clone

import tensorflow as tf

from helpers import show_images
from helpers import plot_confusion_matrix
from helpers import metrics_from_confusion_matrix
from data import Data
from classifiers import Classifiers

# ===========================================================
# CONFIG
# ===========================================================

# suppress TensorFlow logs (Works for Linux, may need changed otherwise)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# format floats in dataframes to .2f
pd.options.display.float_format = '{:,.2f}'.format

# random seeds for reproducability
seed_val = 20156103
os.environ['PYTHONHASHEED']=str(seed_val)
random.seed(seed_val)
np.random.seed(seed_val)
tf.random.set_seed(seed_val)

# ===========================================================
# PARSE CLI ARGS
# ===========================================================
parser = argparse.ArgumentParser()
parser.add_argument('-tt', '--test-type', dest='test_type', 
                    help="0:CROSS VAL, 1:TRAIN/TEST, 2:TRAIN/TEST-4000,\
                    3:TRAIN/TEST-9000",
                    choices=[0, 1, 2, 3], type=int)
parser.add_argument('-c', '--classifier', dest='classifier', 
                    help="Select a model to train",
                    choices=["NN1", "nn1"], type=str, required=True)
parser.add_argument('-nv', '--no-verbose', dest='verbose',
                    help="1(default) for verbosity, 0 otherwise",
                    action="store_false", default=True)
parser.add_argument('-ns', '--no-save', dest='save_model',
                    help="Do not save the model.",
                    action="store_false", default=True)
parser.add_argument('-np', '--no-plots', dest='visualise',
                    help="Do not plot results.",
                    action="store_false", default=True)
args = parser.parse_args()

# ===========================================================
# GLOBAL
# ===========================================================

VERBOSE = args.verbose
TEST_TYPE = args.test_type
CLF = args.classifier.upper()
SAVE_MODEL = args.save_model
VISUALISE = args.visualise

# ===========================================================
# DATA PREPARATION
# ===========================================================

# load data
data = Data()

# pre-process
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
        print(colored("===================================================", "blue"))
        print(colored(f"FOLD: {FOLD_NUM}", "blue"))
        print(colored("===================================================", "blue"))
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

if SAVE_MODEL:
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y--%H-%M-%s")
    model.save(f"../models/{CLF}_{TEST_TYPE}_{timestamp}")

# ===========================================================
# DISPLAY RESULTS
# ===========================================================

conf_matrix = confusion_matrix(y_truths, y_preds)
print(colored("===================================================", "yellow"))
print(colored("CONFUSION MATRIX", "yellow"))
print(colored("===================================================", "yellow"))
print(colored(conf_matrix, "yellow"))
print(colored("===================================================", "magenta"))
print(colored("RESULTS", "magenta"))
print(colored("===================================================", "magenta"))
mean_acc = sum(acc_per_fold)/len(acc_per_fold)
mean_loss = sum(loss_per_fold)/len(loss_per_fold)
print(colored(f"Accuracy: {mean_acc}\nLoss: {mean_loss}", "magenta"))
print(colored("---------------------------------------------------", "magenta"))
metrics = metrics_from_confusion_matrix(conf_matrix)
print(colored(metrics, "cyan"))
print(colored("---------------------------------------------------", "magenta"))
print(colored("Sums:", "cyan"))
print(colored(metrics[["TP", "TN", "FP", "FN"]].sum(axis=0), "cyan"))
print(colored("---------------------------------------------------", "magenta"))
print(colored("Means:", "green"))
print(colored(metrics[["Accuracy", "Precision", "Recall"]].mean(axis=0), "green"))
print(colored("===================================================", "magenta"))

# ===========================================================
# VISUALISE
# ===========================================================

if VISUALISE:
    show_images(x_test[:10],
                predictions=y_preds[:10],
                ground_truths=y_truths[:10])
    
    plot_confusion_matrix(conf_matrix)
