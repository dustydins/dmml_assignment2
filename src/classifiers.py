#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
classifiers.py
Author: Arran Dinsmore
Last updated: 04/12/2020
Description: Collection of classifiers and methods to compile them
"""

from dataclasses import dataclass

from keras.models import Sequential
from keras.layers import Dense

from sklearn.tree import DecisionTreeClassifier


@dataclass
class Classifiers:
    """
    Data - stores data sets
    """

    _nn1 = None
    _dt1 = None

    # =================================================================
    # Neural Networks - NN
    # =================================================================

    def compile_nn1(self):
        """
        NN1 - baseline neural network
        """
        self._nn1 = Sequential()
        self._nn1.add(Dense(128, activation='relu'))
        self._nn1.add(Dense(10, activation='softmax'))
        self._nn1.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        return self._nn1

    # =================================================================
    # Decision Trees - DT
    # =================================================================

    def compile_dt1(self):
        """
        DT1 - baseline decision tree
        """
        self._dt1 = DecisionTreeClassifier(random_state=0, max_depth=2)
        return self._dt1
