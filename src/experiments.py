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
from tabulate import tabulate

from pretty_format import cprint, print_header, print_footer

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
parser.add_argument('-sb', '--show-baseline', dest='baseline',
                    help="Show baseline alongside experiment",
                    action="store_true", default=False)
args = parser.parse_args()

BASELINE = args.baseline
RESULTS_DIR = f"../results/{args.results_dir}"
PER_FOLD_CSV = f"{RESULTS_DIR}/per_fold.csv"
PER_FOLD_MELTED_CSV = f"{RESULTS_DIR}/per_fold_melted.csv"
PER_CLASS_CSV = f"{RESULTS_DIR}/per_class.csv"
BASELINE_CLASS_CSV = "../results/baseline/per_class.csv"
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
    per_fold_df["experiment"] = per_fold_df["experiment"].replace("NO_GENERATIONK", "CONTROL")

    PER_FOLD_MELTED_HEADERS = ["fold_num", "experiment", "variable",
                               "value", "metric", "set"]

    per_fold_melted_df = pd.read_csv(PER_FOLD_MELTED_CSV,
                                     names=PER_FOLD_MELTED_HEADERS)
    per_fold_melted_df["experiment"] = per_fold_melted_df["experiment"].replace("NO_GENERATIONK", "CONTROL")

    EPOCH_HEADERS = ["epoch", "experiment", "fold_num",
                     "variable", "value", "metric", "set"]
    epoch_melted_df = pd.read_csv(EPOCH_HIST_MELTED_CSV,
                                  names=EPOCH_HEADERS)
    epoch_melted_df["experiment"] = epoch_melted_df["experiment"].replace("NO_GENERATIONK", "CONTROL")

    PER_CLASS_HEADERS = ["TP", "TN", "FP", "FN", "Accuracy",
                         "Precision", "Recall", "Label", "experiment"]
    per_class_df = pd.read_csv(PER_CLASS_CSV, names=PER_CLASS_HEADERS)
    if BASELINE:
        baseline_class_df = pd.read_csv(BASELINE_CLASS_CSV,
                                        names=PER_CLASS_HEADERS)
        per_class_df = pd.concat([baseline_class_df, per_class_df])

except IOError as exc:
    print(exc)

if SORT:
    epoch_melted_df = epoch_melted_df.sort_values(by=['experiment'])
    per_fold_melted_df = per_fold_melted_df.sort_values(by=['experiment'])
    per_class_df = per_class_df.sort_values(by=['experiment'])
# ===========================================================
# VISUALISE - PER FOLD
# ===========================================================

#  plot for loss
loss_df = per_fold_melted_df[per_fold_melted_df["metric"] == "loss"]
_ax = sns.lineplot(data=loss_df, x="fold_num",
                   y="value", hue="experiment", style="set")
_ax.set(xlabel="Fold Number", ylabel="Loss",
        title="Train/Test Loss Per Fold")
plt.show()

# calculate differences differences
diff_df = per_fold_df.groupby("experiment", as_index=False, sort=False).agg(
    {"train_acc": 'mean',
     "train_loss": 'mean',
     "test_acc": 'mean',
     "test_loss": 'mean'})
diff_df["accuracy_delta"] = diff_df["test_acc"] - diff_df["train_acc"]
diff_df["loss_delta"] = diff_df["test_loss"] - diff_df["train_loss"]
# put metrics on same scale
diff_df["train_acc"] = diff_df["train_acc"] / 100
diff_df["test_acc"] = diff_df["test_acc"] / 100
diff_df["accuracy_delta"] = diff_df["accuracy_delta"] / 100

# print dataframe
print_header("Mean Accuracy/Loss Per Fold - \
Training/Test Set for each Experiment, with Deltas", "yellow")
cprint(tabulate(diff_df,
                tablefmt="psql",
                headers="keys"), "yellow")
print_footer("yellow")

# plot horizontal bar plot of diffs
melted_diff_df = diff_df[["experiment", "accuracy_delta",
                          "loss_delta"]].melt(id_vars=["experiment"])
_ax = sns.catplot(data=melted_diff_df,
                  kind="bar",
                  x="value",
                  y="experiment",
                  hue="variable")
_ax.set_axis_labels("Difference (Test - Training)", "Experiment")
_ax.despine(bottom=True)
plt.show()

# for single config plots
_ax = sns.lineplot(data=per_fold_melted_df, x="fold_num",
                   y="value", hue="metric", style="set")
_ax.set(xlabel="Fold Number", ylabel="Metric Value",
        title="experiment")
plt.show()

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
