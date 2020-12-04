#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for preprocessing image data
"""

import numpy as np


def randomise_sets(set1, *others):
    """ Randomises the order of two sets of same length in unison """
    original_state = np.random.get_state()
    np.random.shuffle(set1)
    count = 1
    for setx in others:
        assert len(set1) == len(setx)
        count += 1
        np.random.set_state(original_state)
        np.random.shuffle(setx)


def downsample(images, from_size=48, to_size=28):
    """ feature reduction by downsampling """
    images = np.apply_along_axis(
        func1d=lambda img: cv2.resize(
            img.reshape(from_size, from_size),
            dsize=(to_size, to_size)),
        axis=1, arr=images).reshape(-1, to_size*to_size)
