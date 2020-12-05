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
from metrics import get_train_test_acc_loss_df


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


def plot_train_test_acc_loss(train_acc, train_loss,
                             test_acc, test_loss,
                             experiment):
    """
    Prints train/test accuracy and loss for each fold
    """
    _df, melt_df, loss_df = get_train_test_acc_loss_df(train_acc, train_loss,
                                                       test_acc, test_loss,
                                                       experiment)

    # plot both loss and acc if NN, otherwise just acc
    if not (loss_df["value"] == 0).all():
        _ax = sns.lineplot(data=melt_df, x="fold_num",
                           y="value", hue="metric", style="set")
        _ax.set(xlabel="Fold Number", ylabel="Metric Value",
                title=experiment)
    else:
        melt_df = melt_df[melt_df["metric"] == "accuracy"]
        _ax = sns.lineplot(data=melt_df, x="fold_num",
                           y="value", style="set")
        _ax.set(xlabel="Fold Number", ylabel="Accuracy",
                title=experiment)
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
