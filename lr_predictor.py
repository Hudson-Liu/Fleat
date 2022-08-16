#!/usr/bin/env python

"""lr_predictor.py: Houses getLR()"""

__author__ = "Hudson Liu"

import tensorflow as tf
import keras
import numpy as np
from preprocessor import preprocess_images
import pickle

def getLR(sample_data: np.ndarray, model: keras.Model, optimizer: str) -> float:
    """
    Runs the pretrained learning rate predictor model for any given dataset, model, and optimizer. 
    Returns a predicted learning rate.
    
    :param sample_data: The sample images, accepts a Numpy ndarray. Needs to be of shape (a, x, y, c) or (a, x, y),
        with a being the number of images, x and y being the dimensions, and c being the color channels.
    :type sample_data: np.ndarray
    
    :param model: A fully built Keras model, does not need to be compiled.
    :type model: keras.Model
    
    :param optimizer: A string representing a valid Keras optimizer. The following are the only valid Keras optimizers:
        - SGD
        - RMSprop
        - Adam
        - Adadelta
        - Adagrad
        - Adamax
        - Nadam
        - Ftrl
    
    Uppercase or lowercase does not matter.
    Custom optimizers are not currently supported by Fleat.
    :type optimizer: str
    
    :return: A learning rate
    :rtype: float
    
    :raises ValueError: if the optimizer is not a valid Keras optimizer
    """
    #Preprocesses the images
    processed = preprocess_images(sample_data)
    
    #Normalize parameter number
    param_num = model.count_params()
    with open("min_max", "rb") as fp:
        min_max = pickle.load(fp)
    normalized_params = (param_num - min_max[0])/(min_max[1] - min_max[0])
    
    #One hot encode the optimizer, not case sensitives
    KNOWN_OPTIMIZERS = ["sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"]
    onehot = []
    optimizer = optimizer.lower()
    for index, opt in enumerate(KNOWN_OPTIMIZERS):
        if optimizer == opt:
            empty = np.zeros(len(KNOWN_OPTIMIZERS))
            empty[index] = 1
            onehot = empty
            break
    if len(onehot) == 0:
        raise ValueError(f"The optimizer string is expected to be a valid Keras optimizer, instead received {optimizer}.")
    
    #Load model and run prediction
    predictor = tf.saved_model.load("./Fleat_model/")
    learning_rate = predictor.predict([processed, normalized_params, onehot])
    
    #Return prediction
    return learning_rate
    