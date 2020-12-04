#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
metrics.py
Author: Arran Dinsmore
Last updated: 04/12/2020
Description: Module for handling metric calculations
             & presentation.
"""

import warnings
import pandas as pd
import numpy as np

from data import Labels

METRIC_LABLES = ['TP', 'TN', 'FP', 'FN', 'Accuracy',
                 'Precision', 'Recall', 'Label']


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics[4] = (metrics[0]+metrics[1])/sum(sum(conf_matrix))
        metrics[5] = metrics[0]/(metrics[0]+metrics[2])
        metrics[6] = metrics[0]/(metrics[0]+metrics[3])
    metrics[7] = label
    metrics = np.nan_to_num(metrics)
    return metrics


def metrics_from_confusion_matrix(conf_matrix):
    """ returns DataFrame of metrics for each class in
    confusion matrix"""
    all_metrics = np.empty((len(Labels), len(METRIC_LABLES)))
    for label in Labels:
        all_metrics[label.value] = per_class_metrics(conf_matrix, label.value)
    all_metrics_df = pd.DataFrame(all_metrics, columns=METRIC_LABLES)
    return all_metrics_df
