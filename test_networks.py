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
        
        model.add(layers.Conv2D(input_shape=(73,73,3),filters=25,kernel_size=(3,3),**C2D_dic))
        model.add(layers.Conv2D(25,(3,3),**C2D_dic))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        
        model.add(layers.Conv2D(50,(3,3),**C2D_dic))
        model.add(layers.Conv2D(50,(3,3),**C2D_dic))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        
        model.add(layers.Conv2D(70,(3,3),**C2D_dic))
        model.add(layers.Conv2D(70,(3,3),**C2D_dic))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(100,activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(100,activation='relu'))
        model.add(layers.Dropout(0.2))

        model.add(layers.Dense(4,activation='softmax'))

        optimizer = keras.optimizers.Adam(lr=0.001)

        self.model = model
        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.categorical_crossentropy,
            metrics=['accuracy']
        )

if __name__ == '__main__':

    # Here's a usage example using the above CNN that was trained to determine the 3D shape of
    # simulated galaxies from mock 2D projections. Here the training data were not actually RGB
    # images, but a stack of 2D maps: velocity map in channel 1, velocity dispersion map in channel 2,
    # and mass map in channel 3.

    # It should be possible to use your own sequentially built CNN along with your pretrained weights
    # below. In the call of the DreamMyImage function you'll need to replace "CCLS.model" with your
    # CNN, 'test_weights/tst_CNN.h5' with your saved weights, and 'images/garcia.jpg' with the image
    # you want to deep dream on. The final required argument is the output file name. If you want to
    # do this multiple times in the same script, just remember to rebuild your model before calling
    # DreamMyImage as this function removes the last few layers of any CNN that includes Flattening
    # at the end. Play around with the kwargs too for the best results!
    
    kwdic = {
        'nstep':25,
        'octave_scale':1.4,
        'step_size':0.015
    }

    # Maximise 3 random layers:
    CCLS = CNN_classifier()
    DreamMyImage('images/garcia.jpg',CCLS.model,'test_weights/tst_CNN_scaled.h5','test_dream1.png',nlayer=3,**kwdic)
    
    # Maximise a specific layers, here you can print the summary of the model to get all the layer names inside.
    # Be aware that if you create multiple models in one script without specifying the names of each layer, the
    # names will change slightly for each instance. Activating different layers will produce wildly different
    # results, so play around here as well.
    CCLS = CNN_classifier()
    CCLS.model.summary()
    lnames = ['conv2d_8']
    DreamMyImage('images/garcia.jpg',CCLS.model,'test_weights/tst_CNN_scaled.h5',f'test_dream2.png',layer_names=lnames,**kwdic)
