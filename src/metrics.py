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


def get_epoch_hist_df(epoch_hist, experiment="N/A"):
    """
    Convert epoch history to dataframe
    """
    epoch_hist_df = pd.DataFrame(columns=[*epoch_hist[0].history.keys(),
                                          "fold_num", "epoch", "experiment"])
    for idx, hist in enumerate(epoch_hist):
        hist_dict = hist.history
        fold = idx + 1
        hist_dict_df = pd.DataFrame.from_dict(hist_dict)
        hist_dict_df["fold_num"] = fold
        hist_dict_df["epoch"] = hist_dict_df.index + 1
        hist_dict_df["experiment"] = experiment
        epoch_hist_df = epoch_hist_df.append(hist_dict_df, ignore_index=True)

    # add columns for metric used
    acc_df = epoch_hist_df[["accuracy",
                            "val_accuracy",
                            "fold_num",
                            "epoch",
                            "experiment"]].melt(
                                id_vars=["epoch", "experiment", "fold_num"])
    loss_df = epoch_hist_df[["loss",
                             "val_loss",
                             "fold_num",
                             "epoch",
                             "experiment"]].melt(
                                 id_vars=["epoch", "experiment", "fold_num"])
    acc_df["metric"] = "accuracy"
    loss_df["metric"] = "loss"
    melt_df = pd.concat([acc_df, loss_df])

    # add column for data set used
    train_df = melt_df[~melt_df["variable"].str.contains("val")].copy()
    train_df["set"] = "train"
    test_df = melt_df[melt_df["variable"].str.contains("val")].copy()
    test_df["set"] = "test"
    melt_df = pd.concat([train_df, test_df])

    return epoch_hist_df, melt_df


def get_train_test_acc_loss_df(train_acc, train_loss,
                               test_acc, test_loss,
                               experiment="N/A"):
    """
    Creates a DataFrame for train/test acc/loss per fold
    """

    num_folds = 10

    _df = pd.DataFrame()
    _df["fold_num"] = list(range(1, num_folds+1))
    if len(train_acc) == num_folds:
        _df["train_acc"] = train_acc
        _df["train_loss"] = train_loss
        _df["test_acc"] = test_acc
        _df["test_loss"] = test_loss
    else:
        _df["train_acc"] = [train_acc[0] for idx in range(num_folds)]
        _df["train_loss"] = [train_loss[0] for idx in range(num_folds)]
        _df["test_acc"] = [test_acc[0] for idx in range(num_folds)]
        _df["test_loss"] = [test_loss[0] for idx in range(num_folds)]
    _df["experiment"] = [experiment for idx in range(num_folds)]

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
