#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
preprocess.py
Author: Arran Dinsmore
Last updated: 04/12/2020
Description: A collection of helper functions for
             preprocessing data
"""

import numpy as np
import cv2

from keras.preprocessing.image import ImageDataGenerator


def randomise_sets(set1, *others):
    """
    Randomises the order of two sets of same length in unison
    """
    original_state = np.random.get_state()
    np.random.shuffle(set1)
    count = 1
    for setx in others:
        assert len(set1) == len(setx)
        count += 1
        np.random.set_state(original_state)
        np.random.shuffle(setx)


def downsample(images, from_size=48, to_size=28):
    """
    feature reduction by downsampling
    """
    images = np.apply_along_axis(
        func1d=lambda img: cv2.resize(
            img.reshape(from_size, from_size),
            dsize=(to_size, to_size)),
        axis=1, arr=images).reshape(-1, to_size*to_size)


def data_augmentation(img, rot=10):
    """
    Apply data augmentation to provided sets
    """
    augmentor = ImageDataGenerator(rotation_range=rot,
                                   #  zoom_range=0.1,
                                   #  width_shift_range=0.1,
                                   #  height_shift_range=0.1,
                                   #  brightness_range=(0.3, 0.5),
                                   fill_mode="constant",
                                   data_format="channels_last")
    return augmentor.random_transform(img)
