# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 13:02:20 2022

@author: kawta
"""



#Import the libraries

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import vgg16 as vgg



###############################################################################
#######  I M P O R T   P R E - T R A I N E D   M O D E L   ####################
###############################################################################
###############################################################################
#######  A N D   A D D I N G   R E G U L A R I Z A T I O N ####################
###############################################################################

from tensorflow.keras.applications import vgg16 as vgg


# Generate a model with all layers (with top)
vgg16 = vgg.VGG16(weights=None, include_top=False, input_shape=(48,48,3))
regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)
print(f'vgg16.losses (original): {vgg16.losses}')

for i in range(len(vgg16.layers)):
  if isinstance(vgg16.layers[i], tf.keras.layers.Conv2D):
    print('Adding regularizer to layer {}'.format(vgg16.layers[i].name))
    vgg16.layers[i].kernel_regularizer = regularizer
    
print(f'vgg16.losses (after setting reg): {vgg16.losses}')

# Add Dense layer
classes = 10
x = vgg16.output
x = tf.keras.layers.Dense(
    classes, kernel_regularizer=regularizer, name='labels')(
        x)

my_model = tf.keras.Model(vgg16.input, x)
print(f'vgg16.losses (after adding dense): {vgg16.losses}')

my_model.summary()


import os
import tempfile

tmp_weights_dir = tempfile.gettempdir()
tmp_weights_path = os.path.join(tmp_weights_dir, 'tmp_weights.h5')

# Save model config and weights and reload the model from these values.
model_json = vgg16.to_json()
vgg16.save_weights(tmp_weights_path)
model = tf.keras.models.model_from_json(model_json)
model.load_weights(tmp_weights_path, by_name=True)
print(f'model.losses (after reloading):')
loss_str = '\n'.join(['\t' + str(loss) for loss in model.losses])
print(loss_str)







