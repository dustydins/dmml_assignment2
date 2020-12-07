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
import math
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
from metrics import get_epoch_hist_df
from pretty_format import cprint, print_header, print_div, print_footer
from data import Data
from classifiers import Classifiers
from preprocess import data_augmentation
from visualise import show_images
from visualise import get_train_test_acc_loss_df
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


# ===========================================================
# PARSE CLI ARGS
# ===========================================================
parser = argparse.ArgumentParser()
parser.add_argument('-tt', '--test-type', dest='test_type',
                    help="0:CROSS VAL, 1:TRAIN/TEST, 2:TRAIN/TEST-4000,\
                    3:TRAIN/TEST-9000",
                    choices=[0, 1, 2, 3], type=int,
                    default=0)
parser.add_argument('-c', '--classifier', dest='classifier',
                    help="Select a model to train",
                    type=str, default="nn")
parser.add_argument('-nv', '--no-verbose', dest='verbose',
                    help="1(default) for verbosity, 0 otherwise",
                    action="store_false", default=True)
parser.add_argument('-ns', '--no-save', dest='save_model',
                    help="Do not save the model.",
                    action="store_false", default=True)
parser.add_argument('-np', '--no-plots', dest='visualise',
                    help="Do not plot results.",
                    action="store_false", default=True)
parser.add_argument('-sr', '--save-results', dest='save_results',
                    help="Select a destination to save results.",
                    type=str, default="temp_results")
parser.add_argument('-rs', '--random-seed', dest='random_seed',
                    help="Use a random seed",
                    type=int, default=-1)
parser.add_argument('-V', '--validation_split', dest='val_split',
                    help="Set a different validation split",
                    type=float, default=0.33)
parser.add_argument('-LR', '--learning_rate', dest='learning_rate',
                    help="Set a different learning rate",
                    type=float, default=0.001)
parser.add_argument('-NH', '--num_hidden', dest='num_hidden',
                    help="Set a different number of hidden layers",
                    type=int, default=2)
parser.add_argument('-NN', '--num_nodes', dest='num_nodes',
                    help="Set a different number of nodes per hidden layer",
                    type=int, default=128)
parser.add_argument('-E', '--epochs', dest='epochs',
                    help="Set a different number of epochs",
                    type=int, default=10)
parser.add_argument('-en', '--experiment-name', dest='experiment',
                    help="Define a custom experiment name",
                    type=str, default="n/a")
parser.add_argument('-da', '--data-aug', dest='data_aug',
                    help="Proportion of training set to augment",
                    type=float, default=-1.0)
args = parser.parse_args()

# ===========================================================
# GLOBAL
# ===========================================================

VERBOSE = args.verbose
TEST_TYPE = args.test_type
CLF = args.classifier.lower()
SAVE_MODEL = args.save_model
SAVE_RESULTS_TO = args.save_results
VISUALISE = args.visualise
VAL_SPLIT = args.val_split
LEARNING_RATE = args.learning_rate
NUM_HIDDEN = args.num_hidden
NUM_NODES = args.num_nodes
EPOCHS = args.epochs
DATA_AUG = args.data_aug
if args.experiment:
    EXPERIMENT_STR = args.experiment
else:
    EXPERIMENT_STR = f"Model: {CLF.upper()} | Test Type: {TEST_TYPE}"
# random seeds for reproducability
if args.random_seed != -1:
    os.environ['PYTHONHASHEED'] = str(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

# ===========================================================
# DATA PREPARATION
# ===========================================================

# load data
data = Data()
data.df_to_np()

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
    return func(num_hidden=NUM_HIDDEN, num_nodes=NUM_NODES,
                learning_rate=LEARNING_RATE)


# ===========================================================
# RUN CLASSIFIER
# ===========================================================

# lists to store results
epoch_hist = []
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
# for each fold in data fold indicies
for idx, indices in enumerate(data.fold_indices):
    # ---------------------------------------------------------------------------
    # SETUP TRAINING AND TEST SET
    # ---------------------------------------------------------------------------
    if not data.fold_indices[0]:
        # if not cross validation, just use train/test sets
        fold_x_train = data.x_train
        fold_y_train = data.y_train
        fold_x_test = data.x_test
        fold_y_test = data.y_test
    elif idx != 0:
        # if cross validation, use indices to create new train/test sets
        print_header(f"FOLD: {FOLD_NUM}", SECTION_COLOUR)
        fold_x_train = data.x_train[indices[0]]
        fold_x_test = data.x_train[indices[1]]
        fold_y_train = data.y_train[indices[0]]
        fold_y_test = data.y_train[indices[1]]
    else:
        continue

    # reshape if cnn
    if "cnn" in CLF:
        fold_x_train = fold_x_train.reshape(fold_x_train.shape[0],
                                            48, 48, 1)
        fold_x_test = fold_x_test.reshape(fold_x_test.shape[0],
                                          48, 48, 1)

    # ---------------------------------------------------------------------------
    # PREP & FIT
    # ---------------------------------------------------------------------------

    model = compile_clf()

    # apply data augmentation
    if DATA_AUG > 0:
        original_shape = fold_x_train.shape
        fold_x_train = fold_x_train.reshape(fold_x_train.shape[0],
                                            48, 48, 1)
        data_aug = data_augmentation(fold_x_train, validation_split=VAL_SPLIT)
        batch_size = math.floor(original_shape[0]*DATA_AUG)
        for img in range(batch_size):
            data_aug.random_transform(fold_x_train[idx])
        fold_x_train = fold_x_train.reshape(original_shape)

    history = model.fit(x=fold_x_train,
                        y=fold_y_train,
                        validation_split=VAL_SPLIT,
                        epochs=EPOCHS,
                        verbose=VERBOSE)
    epoch_hist.append(history)

    # ---------------------------------------------------------------------------
    # EVALUATE FOR TRAINING AND TEST SETS
    # ---------------------------------------------------------------------------
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
# SAVE RESULTS
# ===========================================================

# get epoch history
epoch_hist_dfs = get_epoch_hist_df(epoch_hist, EXPERIMENT_STR)

# get the per fold results as pandas dataframes
results_df = get_train_test_acc_loss_df(acc_per_fold_train,
                                        loss_per_fold_train,
                                        acc_per_fold_test,
                                        loss_per_fold_test,
                                        experiment=EXPERIMENT_STR)

# get per class results
conf_matrix = confusion_matrix(y_truths, y_preds)
per_class_results_df = metrics_from_confusion_matrix(conf_matrix)
per_class_results_df["experiment"] = EXPERIMENT_STR

# create directory if not already created
outdir = f"../results/{SAVE_RESULTS_TO}"
if not os.path.exists(outdir):
    os.mkdir(outdir)

# save df as printed to terminal
results_df[0].to_csv(f"{outdir}/per_fold.csv",
                     mode='a', header=False)
# save melted df used for some plots
results_df[1].to_csv(f"{outdir}/per_fold_melted.csv",
                     mode='a', header=False)
# save epoch df used for some plots
epoch_hist_dfs[0].to_csv(f"{outdir}/epoch_hist.csv",
                         mode='a', header=False)
# save melted epoch df used for some plots
epoch_hist_dfs[1].to_csv(f"{outdir}/epoch_hist_melted.csv",
                         mode='a', header=False)
# save per class metrics df used for some plots
per_class_results_df.to_csv(f"{outdir}/per_class.csv",
                            mode='a', header=False)

# ===========================================================
# DISPLAY RESULTS
# ===========================================================

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
cprint(tabulate(per_class_results_df, tablefmt='psql',
                headers='keys'), SECTION_COLOUR)
print_div(SECTION_COLOUR)
cprint("Sums:", SECTION_COLOUR)
cprint(per_class_results_df[["TP", "TN",
                             "FP", "FN"]].sum(axis=0), SECTION_COLOUR)
print_div(SECTION_COLOUR)
cprint("Means:", SECTION_COLOUR)
cprint(per_class_results_df[["Accuracy", "Precision",
                             "Recall"]].mean(axis=0),
       SECTION_COLOUR)
print_footer(SECTION_COLOUR)

# print epoch history
SECTION_COLOUR = "yellow"
print_header("EPOCH HISTORY", SECTION_COLOUR)
cprint(tabulate(epoch_hist_dfs[0],
                tablefmt='psql',
                headers='keys'), SECTION_COLOUR)
print_footer(SECTION_COLOUR)

# ===========================================================
# VISUALISE
# ===========================================================

if VISUALISE:

    # plot each fold's accuracy/loss on train and test sets
    plot_train_test_acc_loss(acc_per_fold_train,
                             loss_per_fold_train,
                             acc_per_fold_test,
                             loss_per_fold_test,
                             EXPERIMENT_STR)

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
