#!/usr/bin/env python

"""preprocessor.py: Houses the preprocess_images() method, which preprocesses any unprocessed input data."""

__author__ = "Hudson Liu"

import cv2
import numpy as np

NUM_IMAGES: int = 100
RESOLUTION: list[int] = [224, 224]

def preprocess_images(x: np.ndarray) -> np.ndarray:
    """
    Convert images of shape (a, x, y, c) or (a, x, y) to (100, 224, 224, 1), 
    with a being the number of images, x and y being the dimensions, and c being the color channels.
    Is automatically called by the getLR() function, but can still be called upon manually.
    
    :param x: The input images, accepts a Numpy ndarray
    :type x: np.ndarray
    
    :return: The modified images of shape (100, 224, 224, 1)
    :rtype: np.ndarray
    
    :raises ValueError: if the input image's color channels are not either None, 1, or 3
    """
    # Splice images
    x = x[0:NUM_IMAGES]

    # Test if images are RGB
    if len(x.shape) == 3: #If there is no color channel
        rgb = False
    elif x.shape[3] == 1: #If there is one color channel
        rgb = False
    elif x.shape[3] == 3: #If there are 3 color channels
        rgb = True
    else:
        raise ValueError(f"The input images's color channels were expected to be 1, 3, or None, instead received {x.shape[3]}")

    # Scales images to 224x224 resolution and makes image greyscale
    x_resized = []
    for img in x:
        # if it's enlargening
        if img.shape[0] <= RESOLUTION[0] and img.shape[1] <= RESOLUTION[1]:
            interpolation = cv2.INTER_CUBIC
        # if it's downscaling
        else:
            interpolation = cv2.INTER_AREA
        resized = cv2.resize(img, dsize=(RESOLUTION[0], RESOLUTION[1]), interpolation=interpolation)
        # Converts RGB to grayscale
        if rgb:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            
        x_resized.append(resized)

    # Normalize
    x_resized = np.array(x_resized)
    x_resized = x_resized.astype("float32") / 255.0

    # Add new axis to images to make their shape (224, 224, 1)
    x_resized = np.expand_dims(x_resized, axis=-1)

    # Return
    return x_resized
