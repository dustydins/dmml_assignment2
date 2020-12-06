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
from keras.optimizers import Adam


@dataclass
class Classifiers:
    """
    Data - stores data sets
    """

    _nn = None
    _cnn = None

    # =================================================================
    # Neural Network - NN
    # =================================================================

    def compile_nn(self, num_hidden=2, num_nodes=128,
                   learning_rate=0.001):
        """
        NN - construct a neural network
        """
        optimiser = Adam(lr=learning_rate)

        self._nn = Sequential()
        for _ in range(num_hidden):
            self._nn.add(Dense(num_nodes, activation='relu'))
        self._nn.add(Dense(10, activation='softmax'))
        self._nn.compile(optimizer=optimiser,
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        return self._nn

    # =================================================================
    # Conv Network - CNN
    # =================================================================

    def compile_cnn(self, num_hidden=2, num_nodes=128,
                    learning_rate=0.001):
        """
        CNN - construct a convolutional neural network
        """
        optimiser = Adam(lr=learning_rate)
        self._cnn = Sequential()
        for _ in range(num_hidden):
            self._cnn.add(Dense(num_nodes, activation='relu'))
        self._cnn.add(Dense(10, activation='softmax'))
        self._cnn.compile(optimizer=optimiser,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        return self._cnn
