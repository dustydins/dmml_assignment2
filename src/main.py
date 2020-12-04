#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py
Author: Arran Dinsmore
Last updated: 04/12/2020
Description: Takes arguments from CLI, runs an experiment according
             to user specification.
"""

import os
import random
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.python.keras.engine.sequential import Sequential

from metrics import metrics_from_confusion_matrix
from pretty_format import cprint, print_header, print_div, print_footer
from data import Data
from classifiers import Classifiers
from visualise import show_images
from visualise import plot_train_test_acc_loss
from visualise import print_train_test_acc_loss
from visualise import plot_confusion_matrix

# ===========================================================
# CONFIG
# ===========================================================

# suppress TensorFlow logs (Works for Linux, may need changed otherwise)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# format floats in dataframes to .2f
pd.options.display.float_format = '{:,.2f}'.format

# random seeds for reproducability
SEED_VAL = 44
os.environ['PYTHONHASHEED'] = str(SEED_VAL)
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
tf.random.set_seed(SEED_VAL)

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
                    choices=["NN1", "nn1",
                             "DT1", "dt1"],
                    type=str, required=True)
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
CLF = args.classifier.lower()
SAVE_MODEL = args.save_model
VISUALISE = args.visualise

# ===========================================================
# DATA PREPARATION
# ===========================================================

# load data
data = Data()

# pre-process
SECTION_COLOUR = "green"
print_header("PREPROCESSING", SECTION_COLOUR)
cprint("Normalising data from 0-255 to 0-1.", SECTION_COLOUR)
data.normalise()
print_div(SECTION_COLOUR)
cprint("Randomising sets in unison.", SECTION_COLOUR)
data.randomise()
print_div(SECTION_COLOUR)


def print_shape(text, colour):
    """
    Print set shapes pre/post migrating data
    """
    cprint(f"\tx_train {text}-move: {data.x_train.shape}.", colour)
    cprint(f"\ty_train {text}-move: {data.y_train.shape}.", colour)
    cprint(f"\tx_test {text}-move: {data.x_test.shape}.", colour)
    cprint(f"\ty_test {text}-move: {data.y_test.shape}.", colour)


# Test type
if TEST_TYPE == 0:
    cprint("Preparing data for 10 fold cross validation", SECTION_COLOUR)
    data.cross_val()
elif TEST_TYPE == 2:
    cprint("Moving 4000 instances from training to test set", SECTION_COLOUR)
    print_div(SECTION_COLOUR)
    print_shape("pre", SECTION_COLOUR)

    # move 4000 instances from training to test set
    data.move_to_test(move_num=4000)

    print_div(SECTION_COLOUR)
    print_shape("post", SECTION_COLOUR)
elif TEST_TYPE == 3:
    cprint("Moving 9000 instances from training to test set", SECTION_COLOUR)
    print_div(SECTION_COLOUR)
    print_shape("pre", SECTION_COLOUR)

    # move 9000 instances from training to test set
    data.move_to_test(move_num=9000)

    print_div(SECTION_COLOUR)
    print_shape("post", SECTION_COLOUR)

print_footer(SECTION_COLOUR)

# ===========================================================
# MODEL PREPARATION
# ===========================================================

# dataclass storing all classifier configurations used
clfs = Classifiers()


def compile_clf():
    """
    Returns a newly compiled version of specified model
    """
    func = getattr(clfs, f"compile_{CLF}")
    return func()


# ===========================================================
# RUN CLASSIFIER
# ===========================================================

# lists to store results
acc_per_fold_test = []
loss_per_fold_test = []
acc_per_fold_train = []
loss_per_fold_train = []
y_truths = []
y_preds = []
x_test = []

FOLD_NUM = 1
model = compile_clf()
SECTION_COLOUR = "blue"
print_header("RUNNING CLASSIFIER", SECTION_COLOUR)
for idx, indices in enumerate(data.fold_indices):
    # ---------------------------------------------------------------------------
    # SETUP TRAINING AND TEST SET
    # ---------------------------------------------------------------------------
    if not data.fold_indices[0]:
        fold_x_train = data.x_train
        fold_y_train = data.y_train
        fold_x_test = data.x_test
        fold_y_test = data.y_test
    elif idx != 0:
        print_header(f"FOLD: {FOLD_NUM}", SECTION_COLOUR)
        fold_x_train = data.x_train[indices[0]]
        fold_x_test = data.x_train[indices[1]]
        fold_y_train = data.y_train[indices[0]]
        fold_y_test = data.y_train[indices[1]]
    else:
        continue

    model = compile_clf()

    # ---------------------------------------------------------------------------
    # IF NEURAL NETWORK
    # ---------------------------------------------------------------------------
    if isinstance(model, Sequential):

        # fit model
        model.fit(x=fold_x_train, y=fold_y_train, epochs=10, verbose=VERBOSE)

        # evaluate for training and test sets
        test_loss, test_acc = model.evaluate(fold_x_test,
                                             fold_y_test,
                                             verbose=0)
        train_loss, train_acc = model.evaluate(fold_x_train,
                                               fold_y_train,
                                               verbose=0)
        loss_per_fold_test.append(test_loss)
        acc_per_fold_test.append(test_acc * 100)
        loss_per_fold_train.append(train_loss)
        acc_per_fold_train.append(train_acc * 100)

        # predictions
        fold_probs = model.predict(fold_x_test)
        fold_preds = [np.argmax(instance) for instance in fold_probs]

    # ---------------------------------------------------------------------------
    # IF DECISION TREE
    # ---------------------------------------------------------------------------
    elif isinstance(model, DecisionTreeClassifier):

        # fit model
        model.fit(fold_x_train, fold_y_train)

        # evaluate on train and test sets
        test_acc = model.score(fold_x_test, fold_y_test)
        train_acc = model.score(fold_x_train, fold_y_train)
        acc_per_fold_test.append(test_acc * 100)
        acc_per_fold_train.append(train_acc * 100)
        loss_per_fold_test.append(0)
        loss_per_fold_train.append(0)

        # predictions
        fold_preds = model.predict(fold_x_test)

    # ---------------------------------------------------------------------------
    # STORE PREDICTIONS
    # ---------------------------------------------------------------------------
    for idy, truth in enumerate(fold_y_test):
        y_truths.append(truth)
        y_preds.append(fold_preds[idy])
        x_test.append(fold_x_test[idy])

    FOLD_NUM = FOLD_NUM + 1

print_footer(SECTION_COLOUR)

# ===========================================================
# SAVE MODEL
# ===========================================================

if SAVE_MODEL and isinstance(model, Sequential):
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y--%H-%M-%s")
    model.save(f"../models/{CLF}_{TEST_TYPE}_{timestamp}")

# ===========================================================
# DISPLAY RESULTS
# ===========================================================

conf_matrix = confusion_matrix(y_truths, y_preds)

# print confusion matrix
SECTION_COLOUR = "yellow"
print_header("CONFUSION MATRIX", SECTION_COLOUR)
cprint(conf_matrix, SECTION_COLOUR)
print_footer(SECTION_COLOUR)

# print overall accuracy/loss per fold
SECTION_COLOUR = "magenta"
print_header("RESULTS PER FOLD", SECTION_COLOUR)
print_train_test_acc_loss(acc_per_fold_train, loss_per_fold_train,
                          acc_per_fold_test, loss_per_fold_test)
if TEST_TYPE == 0:
    mean_acc_train = sum(acc_per_fold_train)/len(acc_per_fold_train)
    mean_loss_train = sum(loss_per_fold_train)/len(loss_per_fold_train)
    cprint("Training Set Mean:", SECTION_COLOUR)
    cprint(f"\tAccuracy: {mean_acc_train:.2f}", SECTION_COLOUR)
    cprint(f"\tLoss: {mean_loss_train:.2f}", SECTION_COLOUR)
    mean_acc_test = sum(acc_per_fold_test)/len(acc_per_fold_test)
    mean_loss_test = sum(loss_per_fold_test)/len(loss_per_fold_test)
    cprint("Test Set Mean:", SECTION_COLOUR)
    cprint(f"\tAccuracy: {mean_acc_test:.2f}", SECTION_COLOUR)
    cprint(f"\tLoss: {mean_loss_test:.2f}", SECTION_COLOUR)
print_footer(SECTION_COLOUR)

# print metrics per class
SECTION_COLOUR = "cyan"
print_header("PER CLASS RESULTS", SECTION_COLOUR)
metrics = metrics_from_confusion_matrix(conf_matrix)
cprint(tabulate(metrics, tablefmt='psql',
                headers='keys'), SECTION_COLOUR)
print_div(SECTION_COLOUR)
cprint("Sums:", SECTION_COLOUR)
cprint(metrics[["TP", "TN", "FP", "FN"]].sum(axis=0), SECTION_COLOUR)
print_div(SECTION_COLOUR)
cprint("Means:", SECTION_COLOUR)
cprint(metrics[["Accuracy", "Precision", "Recall"]].mean(axis=0),
       SECTION_COLOUR)
print_footer(SECTION_COLOUR)

# ===========================================================
# VISUALISE
# ===========================================================

if VISUALISE:

    if TEST_TYPE == 0:
        # if cross validation, plot each fold's accuracy/loss
        plot_train_test_acc_loss(acc_per_fold_train,
                                 loss_per_fold_train,
                                 acc_per_fold_test,
                                 loss_per_fold_test)

    # display first 10 images with prediction labels
    show_images(x_test[:10],
                predictions=y_preds[:10],
                ground_truths=y_truths[:10])

    # display confusion matrix
    plot_confusion_matrix(conf_matrix)

    # if decision tree, plot tree
    if isinstance(model, DecisionTreeClassifier):
        plot_tree(model)
        plt.show()
