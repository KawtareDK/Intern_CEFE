# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 15:20:02 2021

@author: kawta
"""


## Importing the neccesary libraries !

import numpy as np ##NumPy is a general-purpose array-processing package. It provides a high-performance multidimensional array object, and tools for working with these arrays.
import pandas as pd ## It offers data structures and operations for manipulating numerical tables and time series
from glob import glob ## (short for global) is used to return all file paths that match a specific pattern


## Libraries for neural networks

# Keras layers are the building blocks of the keras library that can be stacked together just like legos for creating neural network models
from keras.layers import Input, Lambda, Dense, Flatten 
## Dense layer is for creating a deeply connected layer in the neural network where each of the neurons of the dense layers receives input from all neurons of the previous layer
## Lambda layer is used for transforming the input data with the help of an expression or function
## Flatten layer is used for flattening of the input. For example, if we have an input shape as (batch_size, 3,3), after applying the flatten layer, the output shape is changed to (batch_size,9)


from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from skimage.io import imread, imshow

image = imread("./Data/train/CFD-AF-200-N.jpg")
plt.imshow(image)
image.shape
IMSIZE = [224, 224]


##Split folders 
import splitfolders
splitfolders.ratio("CFD", output="output", seed=1337, ratio=(.8, .2), group_prefix=None) # default values

### Image data gen and importing the dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen =ImageDataGenerator(rescale= 1./255,
                                  shear_range= 0.2,
                                  zoom_range= 0.2,
                                  horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale=1./255)


## corresponding class indices
train_datagen.class_indices


 
# create generators
train_generator = train_datagen.flow_from_directory(
  "Data/train",
  target_size=IMSIZE,
  shuffle=True,
  batch_size=32,
  class_mode='categorical'
)
 
test_generator = test_datagen.flow_from_directory(
  "Data/val",
  target_size=IMSIZE,
  shuffle=True,
  batch_size=32,
)


#### Model creation from VGGGGGG16


def create_model():
    vgg = VGG16(input_shape=IMSIZE + [3], weights='imagenet', include_top=False)

# Freeze existing VGG already trained weights
    for layer in vgg.layers:
        layer.trainable = False
    
# get the VGG output
    out = vgg.output
    
# Add new dense layer at the end
    x = Flatten()(out)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=vgg.input, outputs=x)
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    model.summary()
    return model

mymodel = create_model()

#### FIT THE MODEl

early_stop = EarlyStopping(monitor='val_loss',patience=1)
r = mymodel.fit_generator(
  train_generator,
  validation_data=test_generator,
  epochs=10,
  steps_per_epoch=len(train_generator) // 6,
  validation_steps=len(test_generator) // 6,
  callbacks=[early_stop]
)



#### evallll
score = mymodel.evaluate_generator(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



























































