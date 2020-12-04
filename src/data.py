#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Handles the data concerned with the coursework
"""

import enum
from dataclasses import dataclass
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import KFold
from scipy.ndimage.interpolation import shift

from preprocess import randomise_sets
from pretty_format import *

np.random.seed(42)


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


# filepaths
print_header("GATHERING DATA")

X_TRAIN_GR_SMPL = "../data/x_train_gr_smpl.csv"
X_TEST_GR_SMPL = "../data/x_test_gr_smpl.csv"
Y_TRAIN_SMPL = "../data/y_train_smpl.csv"
Y_TEST_SMPL = "../data/y_test_smpl.csv"

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

@dataclass
class Data:
    """
    Data - stores data sets
    """
    x_train = genfromtxt(X_TRAIN_GR_SMPL, delimiter=',', skip_header=1)
    y_train = genfromtxt(Y_TRAIN_SMPL, delimiter=',', skip_header=1)
    x_test = genfromtxt(X_TEST_GR_SMPL, delimiter=',', skip_header=1)
    y_test = genfromtxt(Y_TEST_SMPL, delimiter=',', skip_header=1)
    fold_indices = [False]

    def normalise(self):
        """
        Normalises data from 0-255 to 0-1
        """
        self.x_train /= 255
        self.x_test /= 255

    def randomise(self):
        """ randomises sets"""
        randomise_sets(self.x_train, self.y_train)
        randomise_sets(self.x_test, self.y_test)

    def get_class_sets(self, test=False):
        """ separates and returns 10 sets for each class"""
        classes = [[], [], [], [], [], [], [], [], [], []]
        data_set = (None, None)
        if test:
            data_set = (self.x_test, self.y_test)
        else:
            data_set = (self.x_train, self.y_train)
        for pixels, label in zip(data_set[0], data_set[1]):
            for idx in enumerate(classes):
                if label == idx:
                    classes[idx].append(pixels)
        return np.array(classes)

    def cross_val(self):
        """
        Sets fold indices for 10 fold cross validation
        """
        kf_cv = KFold(n_splits=10, shuffle=True, random_state=42)
        for train_idx, test_idx in kf_cv.split(self.x_train, self.y_train):
            self.fold_indices.append((train_idx, test_idx))
        self.fold_indices[0] = True

    def move_to_test(self, move_num=0):
        """
        Move n instances from training set to test set
        """
        assert len(self.x_train) >= move_num
        self.x_test = np.append(self.x_test, self.x_train[:move_num], axis=0)
        self.y_test = np.append(self.y_test, self.y_train[:move_num], axis=0)
        self.x_train = np.delete(self.x_train, [range(move_num)], axis=0)
        self.y_train = np.delete(self.y_train, [range(move_num)], axis=0)
