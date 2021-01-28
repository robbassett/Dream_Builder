import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
import glob
import time
import IPython.display as display
import PIL.Image

from dream_builder import *

class CNN_classifier():

    def __init__(self):
        C2D_dic = {
            'padding':'same',
            'activation':'relu'
        }
        
        model = keras.Sequential()
        model.add(layers.Conv2D(input_shape=(64,64,3),filters=64,kernel_size=(3,3),**C2D_dic))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3,3),strides=(2,2)))
        model.add(layers.Conv2D(128,(4,4),**C2D_dic))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3,3),strides=(2,2)))
        model.add(layers.Conv2D(256,(5,5),**C2D_dic))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3,3),strides=(2,2)))
        model.add(layers.Conv2D(512,(3,3),**C2D_dic))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3,3),strides=(2,2)))
        model.add(layers.Conv2D(512,(3,3),**C2D_dic))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3,3),strides=(2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1,activation='sigmoid'))

        optimizer = keras.optimizers.RMSprop(lr=0.001)

        self.model = model
        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.binary_crossentropy,
            metrics=['accuracy']
        )

if __name__ == '__main__':

    CCLS = CNN_classifier()
    DreamMyImage('images/garcia.jpg',CCLS.model,'test_weights/cmdd.h5','test_dream.png',reweight=False)
