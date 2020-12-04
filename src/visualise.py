#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collection of visualisation methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored

from data import Labels


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


def _train_test_acc_loss_df(train_acc, train_loss, test_acc, test_loss):
    """
    Creates a DataFrame for train/test acc/loss per fold
    """
    _df = pd.DataFrame()
    _df["train_acc"] = train_acc
    _df["train_loss"] = train_loss
    _df["test_acc"] = test_acc
    _df["test_loss"] = test_loss
    return _df


def print_train_test_acc_loss(train_acc, train_loss,
                              test_acc, test_loss, colour="magenta"):
    """ 
    Prints train/test accuracy and loss for each fold
    """
    _df = _train_test_acc_loss_df(train_acc, train_loss,
                                  test_acc, test_loss)
    print(colored(_df, colour))


def plot_train_test_acc_loss(train_acc, train_loss, test_acc, test_loss):
    """ 
    Prints train/test accuracy and loss for each fold
    """
    _df = _train_test_acc_loss_df(train_acc, train_loss,
                                  test_acc, test_loss)


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
