#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A collection of classifiers used for experiments
"""

import numpy as np

from dataclasses import dataclass

from keras.models import Sequential
from keras.layers import Dense


@dataclass
class Classifiers:
    """
    Data - stores data sets
    """

    NN_1 = None

    def compile_NN_1(self):
        self.NN_1 = Sequential()
        self.NN_1.add(Dense(128, activation='relu'))
        self.NN_1.add(Dense(10, activation='softmax'))
        self.NN_1.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return self.NN_1
