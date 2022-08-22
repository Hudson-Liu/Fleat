#!/usr/bin/env python

"""lr_predictor.py: Houses the getLR() method"""

__author__ = "Hudson Liu"

import tensorflow as tf
import keras
import numpy as np
from fleat.preprocessor import preprocess_images
import pickle
import os

def getLR(sample_data: np.ndarray, model: keras.Model, optimizer: str) -> float:
    """
    Runs the pretrained learning rate predictor model for any given dataset, model, and optimizer. 
    Returns a predicted learning rate.
    
    :param sample_data: The sample images, accepts a Numpy ndarray. Needs to be of shape (a, x, y, c) or (a, x, y),
        with a being the number of images, x and y being the dimensions, and c being the color channels.
    :type sample_data: np.ndarray
    
    :param model: A fully built Keras model, does not need to be compiled.
    :type model: keras.Model
    
    :param optimizer: A string representing a valid Keras optimizer. The following are the only valid Keras optimizers:\n
        - SGD\n
        - RMSprop\n
        - Adam\n
        - Adadelta\n
        - Adagrad\n
        - Adamax\n
        - Nadam\n
        - Ftrl\n
        Uppercase or lowercase does not matter.\n
        Custom optimizers are not currently supported by Fleat.
    :type optimizer: str
    
    :return: A learning rate
    :rtype: float
    
    :raises ValueError: if the input image's color channels are not either None, 1, or 3
    :raises ValueError: if the optimizer is not a valid Keras optimizer
    """
    #Get modulepath (where this package was installed) for importing files later
    module_path = os.path.dirname(os.path.realpath(__file__))
    
    #Preprocesses the images
    processed = preprocess_images(sample_data)
    
    #Normalize parameter number
    param_num = model.count_params()
    with open(f"{module_path}\\min_max", "rb") as fp:
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
    
    #Concatenate all the data about the model
    model_data = [*onehot, normalized_params]
    
    #Convert lists to numpy arrays and add a dimension to each array
    processed = np.array([processed])
    model_data = np.array([model_data])
    
    #Load model and run prediction
    predictor = tf.keras.models.load_model(f"{module_path}\\Fleat_model")
    learning_rate = predictor.predict([processed, model_data])
    
    #Reverse the scaling
    learning_rate = learning_rate[0][0] / 1000
    
    #Return prediction
    return learning_rate
    
