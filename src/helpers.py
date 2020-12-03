#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Helpers for assignment 1 - DMML"""

import enum
import cv2
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt

METRIC_LABLES = ['TP', 'TN', 'FP', 'FN', 'Accuracy',
                 'Precision', 'Recall', 'Label']


class Labels(enum.Enum):
    """ enum for class labels """
    SpeedLimit20 = 0
    SpeedLimit30 = 1
    SpeedLimit50 = 2
    SpeedLimit60 = 3
    SpeedLimit70 = 4
    LeftTurn = 5
    RightTurn = 6
    BewarePedastrianCrossing = 7
    BewareChildren = 8
    BewareCycleRouteAhead = 9

    def __str__(self):
        return str(self.name)


def show_images(images, predictions=None, size=48,
                ground_truths=None):
    """Display a list of images in a single figure with matplotlib.
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


def randomise_sets(set1, *others):
    """ Randomises the order of two sets of same length in unison """
    print("Shuffling {} sets in unison...".format(1+len(others)))
    original_state = np.random.get_state()
    np.random.shuffle(set1)
    print("\tSet 1 shuffle: complete.")
    count = 1
    for setx in others:
        assert len(set1) == len(setx)
        count += 1
        np.random.set_state(original_state)
        np.random.shuffle(setx)
        print("\tSet {} shuffle: complete.".format(count))


def downsample(images, from_size=48, to_size=28):
    """ feature reduction by downsampling """
    print("Downsampling images from size {} to {}...".format(
        from_size, to_size))
    images = np.apply_along_axis(
        func1d=lambda img: cv2.resize(
            img.reshape(from_size, from_size),
            dsize=(to_size, to_size)),
        axis=1, arr=images).reshape(-1, to_size*to_size)
    print("\tDownsampling: complete.")


def plot_confusion_matrix(conf_matrix, title="Confusion Matrix"):
    """ plot confusion matrix """
    cm_df = pd.DataFrame(conf_matrix, index=[str(label) for label in Labels],
                         columns=[str(label) for label in Labels])
    plt.figure(figsize=(16, 16))
    axes = sn.heatmap(cm_df, annot=True, cmap="viridis", fmt='g')
    axes.set(ylabel='ground truth', xlabel='predictions')
    axes.set_title(title)
    plt.show()


def per_class_metrics(conf_matrix, label):
    """ returns  metrics wrt class label calculated from
    a given confusion matrix:
    returns [tp, tn, fp, fn, accuracy, precision, recall]
    """
    assert label < len(conf_matrix)
    metrics = np.zeros(8)
    metrics[0] = conf_matrix[label][label]
    metrics[2] = -abs(metrics[0])
    metrics[3] = -abs(metrics[0])
    #  metrics[2:3] = -abs(metrics[0])
    metrics[3] += sum(conf_matrix[label])
    for row in conf_matrix:
        metrics[2] += row[label]
        metrics[1] += sum(row)
    metrics[1] -= metrics[0]+metrics[2]+metrics[3]
    metrics[4] = (metrics[0]+metrics[1])/sum(sum(conf_matrix))
    metrics[5] = metrics[0]/(metrics[0]+metrics[2])
    metrics[6] = metrics[0]/(metrics[0]+metrics[3])
    metrics[7] = label
    return metrics


def metrics_from_confusion_matrix(conf_matrix):
    """ returns DataFrame of metrics for each class in
    confusion matrix"""
    all_metrics = np.empty((len(Labels), len(METRIC_LABLES)))
    for label in Labels:
        all_metrics[label.value] = per_class_metrics(conf_matrix, label.value)
    all_metrics_df = pd.DataFrame(all_metrics, columns=METRIC_LABLES)
    return all_metrics_df
