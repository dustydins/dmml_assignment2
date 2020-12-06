#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
experiments.py
Author: Arran Dinsmore
Last updated: 05/12/2020
Description: Module for loading and visualising results from
             multiple experiments
"""

import argparse

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
#  from tabulate import tabulate

from pretty_format import cprint

# ===========================================================
# CONFIG
# ===========================================================

# format floats in dataframes to .2f
pd.options.display.float_format = '{:,.2f}'.format

# sns palette
sns.set(style="ticks", palette="Set2")

# ===========================================================
# PARSE CLI ARGS
# ===========================================================
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--load-csv', dest='results_dir',
                    help="Filename of csv holding experiment results",
                    type=str, required=True)
parser.add_argument('-s', '--sort', dest='sort',
                    help="Sort dataframes by experiment column",
                    action="store_true", default=False)
args = parser.parse_args()

RESULTS_DIR = f"../results/{args.results_dir}"
PER_FOLD_CSV = f"{RESULTS_DIR}/per_fold.csv"
PER_FOLD_MELTED_CSV = f"{RESULTS_DIR}/per_fold_melted.csv"
PER_CLASS_CSV = f"{RESULTS_DIR}/per_class.csv"
EPOCH_HIST_MELTED_CSV = f"{RESULTS_DIR}/epoch_hist_melted.csv"
SORT = args.sort

# ===========================================================
# LOAD CSV TO DATAFRAME
# ===========================================================

try:
    PER_FOLD_HEADERS = ["fold_num", "train_acc", "train_loss",
                        "test_acc", "test_loss", "experiment"]

    per_fold_df = pd.read_csv(PER_FOLD_CSV,
                              names=PER_FOLD_HEADERS)

    PER_FOLD_MELTED_HEADERS = ["fold_num", "experiment", "variable",
                               "value", "metric", "set"]

    per_fold_melted_df = pd.read_csv(PER_FOLD_MELTED_CSV,
                                     names=PER_FOLD_MELTED_HEADERS)

    EPOCH_HEADERS = ["epoch", "experiment", "fold_num",
                     "variable", "value", "metric", "set"]
    epoch_melted_df = pd.read_csv(EPOCH_HIST_MELTED_CSV,
                                  names=EPOCH_HEADERS)

    PER_CLASS_HEADERS = ["TP", "TN", "FP", "FN", "Accuracy",
                         "Precision", "Recall", "Label", "experiment"]
    per_class_df = pd.read_csv(PER_CLASS_CSV, names=PER_CLASS_HEADERS)

except IOError as exc:
    print(exc)

if SORT:
    epoch_melted_df = epoch_melted_df.sort_values(by=['experiment'])
    per_fold_melted_df = per_fold_melted_df.sort_values(by=['experiment'])
    per_class_df = per_class_df.sort_values(by=['experiment'])
# ===========================================================
# VISUALISE - PER FOLD
# ===========================================================

#  plot for accuracy
acc_df = per_fold_melted_df[per_fold_melted_df["metric"] == "accuracy"]
_ax = sns.lineplot(data=acc_df, x="fold_num",
                   y="value", hue="experiment", style="set")
_ax.set(xlabel="Fold Number", ylabel="Accuracy",
        title="Train/Test Accuracy Per Fold")
plt.show()

# print differences
cprint(per_fold_df, "cyan")

# ===========================================================
# VISUALISE - PER CLASS
# ===========================================================

# precision boxplot
_ax = sns.boxplot(data=per_class_df, x="Label", y="Precision",
                  hue="experiment")
_ax.set(xlabel="Class", title="Precision Results Per Class")
plt.show()

# recall boxplot
_ax = sns.boxplot(data=per_class_df, x="Label", y="Recall",
                  hue="experiment")
_ax.set(xlabel="Class", title="Recall Results Per Class")
plt.show()

# ===========================================================
# VISUALISE - PER EPOCH
# ===========================================================

# per epoch lineplots
_ax = sns.relplot(
    data=epoch_melted_df,
    x="epoch", y="value",
    hue="set", style="metric", col="experiment", col_wrap=2,
    kind="line", style_order=["accuracy", "loss"],
    height=4
)
plt.show()
