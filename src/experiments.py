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

#  from pretty_format import cprint, print_header, print_div, print_footer

# ===========================================================
# CONFIG
# ===========================================================

# format floats in dataframes to .2f
pd.options.display.float_format = '{:,.2f}'.format

# ===========================================================
# PARSE CLI ARGS
# ===========================================================
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--load-csv', dest='results_dir',
                    help="Filename of csv holding experiment results",
                    type=str, required=True)
args = parser.parse_args()

RESULTS_DIR = f"../results/{args.results_dir}"
PER_FOLD_CSV = f"{RESULTS_DIR}/per_fold.csv"
PER_FOLD_MELTED_CSV = f"{RESULTS_DIR}/per_fold_melted.csv"
EPOCH_HIST_MELTED_CSV = f"{RESULTS_DIR}/epoch_hist_melted.csv"

# ===========================================================
# LOAD CSV TO DATAFRAME
# ===========================================================

PER_FOLD_HEADERS = ["fold_num", "experiment", "variable",
                    "value", "metric", "set"]

per_fold_melted_df = pd.read_csv(PER_FOLD_MELTED_CSV,
                                 names=PER_FOLD_HEADERS)

try:
    EPOCH_HEADERS = ["epoch", "experiment", "fold_num",
                     "variable", "value", "metric", "set"]
    epoch_melted_df = pd.read_csv(EPOCH_HIST_MELTED_CSV,
                              names=EPOCH_HEADERS)
except Exception:
    pass
    

# ===========================================================
# VISUALISE
# ===========================================================

#  plot for accuracy
acc_df = per_fold_melted_df[per_fold_melted_df["metric"] == "accuracy"]
_ax = sns.lineplot(data=acc_df, x="fold_num",
                   y="value", hue="experiment", style="set")
_ax.set(xlabel="Fold Number", ylabel="Accuracy",
        title="Train/Test Accuracy Per Fold")
plt.show()

#  plot accuracy multi-facet
#  facet = sns.FacetGrid(per_fold_melted_df, col="experiment", row="metric",
                      #  height=2, aspect=0.75, hue="set",
                      #  margin_titles=True, sharey=False)
#  facet.map(sns.lineplot, "fold_num", "value")
#  facet.add_legend()
#  plt.show()


# Plot the lines on two facets
#  sns.relplot(
    #  data=per_fold_melted_df,
    #  x="fold_num", y="value",
    #  hue="set", style="metric", col="experiment", col_wrap=2,
    #  kind="line", style_order=["accuracy", "loss"],
    #  height=4
#  )
#  plt.show()

#  print(epoch_melted_df.head(10))
sns.relplot(
    data=epoch_melted_df,
    x="epoch", y="value",
    hue="set", style="metric", col="experiment", col_wrap=2,
    kind="line", style_order=["accuracy", "loss"],
    height=4
)
plt.show()

#  fig = sns.relplot(
    #  data=per_fold_melted_df,
    #  x="fold_num", y="value",
    #  hue="set", col="experiment", row="metric",
    #  kind="line",
    #  height=2
#  )
#  plt.show()
