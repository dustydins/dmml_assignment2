#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
visualise.py
Author: Arran Dinsmore
Last updated: 04/12/2020
Description: Collection of visualisation methods
"""

import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored

from data import Labels

NUM_FOLDS = 10


def is_unique(column):
    """
    Checks if a pd.DataFrame column's values are all the same
    """
    arr = column.to_numpy()
    return (arr[0] == arr).all()


def show_images(images, predictions=None, size=48,
                ground_truths=None):
    """
    Display a list of images in a single figure with matplotlib.
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).
    predictions: List of predictions corresponding to each image.
    ground_truths: List of ground truth labels corresponding to each image.
    ranked_features: Highlights pixels given from an array
    """
    cols = 5
    assert((predictions is None or ground_truths is None)
           or (len(images) == len(predictions)))
    n_images = len(images)
    images = [image.reshape(size, size) for image in images]
    if predictions is None:
        predictions = ['Image (%d)' % i for i in range(1, n_images + 1)]
        ground_truths = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for idx, (image, predicted, actual) in enumerate(
            zip(images, predictions, ground_truths)):
        _ax = fig.add_subplot(cols, np.ceil(n_images/float(cols)), idx + 1)
        _ax.set_xticks([])
        _ax.set_yticks([])
        plt.imshow(image)
        if predicted == actual:
            _ax.set_title(str(Labels(predicted)), color='green', fontsize=6)
            #  _ax.set_title(predicted, color='green', fontsize=6)
        else:
            _ax.set_title(str(Labels(predicted)), color='red', fontsize=6)
            #  _ax.set_title(predicted, color='red', fontsize=6)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def get_train_test_acc_loss_df(train_acc, train_loss,
                               test_acc, test_loss,
                               experiment="N/A"):
    """
    Creates a DataFrame for train/test acc/loss per fold
    """
    _df = pd.DataFrame()
    _df["fold_num"] = list(range(1, NUM_FOLDS+1))
    if len(train_acc) == NUM_FOLDS:
        _df["train_acc"] = train_acc
        _df["train_loss"] = train_loss
        _df["test_acc"] = test_acc
        _df["test_loss"] = test_loss
    else:
        _df["train_acc"] = [train_acc[0] for idx in range(NUM_FOLDS)]
        _df["train_loss"] = [train_loss[0] for idx in range(NUM_FOLDS)]
        _df["test_acc"] = [test_acc[0] for idx in range(NUM_FOLDS)]
        _df["test_loss"] = [test_loss[0] for idx in range(NUM_FOLDS)]
    _df["experiment"] = [experiment for idx in range(NUM_FOLDS)]

    # melt and add column for metric used
    acc_df = _df[["fold_num",
                  "train_acc",
                  "test_acc",
                  "experiment"]].melt(id_vars=["fold_num",
                                               "experiment"]).copy()
    acc_df["metric"] = "accuracy"
    acc_df["value"] = acc_df["value"] / 100
    loss_df = _df[["fold_num",
                   "train_loss",
                   "test_loss",
                   "experiment"]].melt(id_vars=["fold_num",
                                                "experiment"]).copy()
    loss_df["metric"] = "loss"
    melt_df = pd.concat([acc_df, loss_df])

    # add column for data set used
    train_df = melt_df[melt_df["variable"].str.contains("train")].copy()
    train_df["set"] = "train"
    test_df = melt_df[melt_df["variable"].str.contains("test")].copy()
    test_df["set"] = "test"
    melt_df = pd.concat([train_df, test_df])
    return _df, melt_df, loss_df


def print_train_test_acc_loss(train_acc, train_loss,
                              test_acc, test_loss, colour="magenta"):
    """
    Prints train/test accuracy and loss for each fold
    """
    _df = get_train_test_acc_loss_df(train_acc, train_loss,
                                     test_acc, test_loss)

    if is_unique(_df[0]["train_acc"]):
        to_print_df = _df[0].drop(["experiment"], axis=1).head(1)
    else:
        to_print_df = _df[0].drop(["experiment"], axis=1)

    print(colored(tabulate(to_print_df, tablefmt='psql',
                           headers='keys'), colour))


def plot_train_test_acc_loss(train_acc, train_loss, test_acc, test_loss):
    """
    Prints train/test accuracy and loss for each fold
    """
    _df, melt_df, loss_df = get_train_test_acc_loss_df(train_acc, train_loss,
                                                       test_acc, test_loss)

    # plot both loss and acc if NN, otherwise just acc
    if not (loss_df["value"] == 0).all():
        _ax = sns.lineplot(data=melt_df, x="fold_num",
                           y="value", hue="set", style="metric")
        _ax.set(xlabel="Fold Number", ylabel="Metric Value",
                title="Train/Test Accuracy/Loss Per Fold")
    else:
        melt_df = melt_df[melt_df["metric"] == "accuracy"]
        _ax = sns.lineplot(data=melt_df, x="fold_num",
                           y="value", hue="set")
        _ax.set(xlabel="Fold Number", ylabel="Accuracy",
                title="Train/Test Accuracy Per Fold")
    plt.show()


def plot_confusion_matrix(conf_matrix, title="Confusion Matrix"):
    """
    Plot confusion matrix
    """
    cm_df = pd.DataFrame(conf_matrix, index=[str(label) for label in Labels],
                         columns=[str(label) for label in Labels])
    plt.figure(figsize=(16, 16))
    axes = sns.heatmap(cm_df, annot=True, cmap="viridis", fmt='g')
    axes.set(ylabel='Ground Truth', xlabel='Predictions')
    axes.set_title(title)
    plt.show()
