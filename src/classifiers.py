#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A collection of classifiers used for experiments
"""

from dataclasses import dataclass

from keras.models import Sequential
from keras.layers import Dense


@dataclass
class Classifiers:
    """
    Data - stores data sets
    """

    NN1 = None

    def compile_NN1(self):
        self.NN1 = Sequential()
        self.NN1.add(Dense(128, activation='relu'))
        self.NN1.add(Dense(10, activation='softmax'))
        self.NN1.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return self.NN1
