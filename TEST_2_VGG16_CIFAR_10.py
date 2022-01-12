# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:37:21 2022

@author: kawta
"""

###############################################################################
#######  T R A N S F E R   L E A R N I N G   W I T H   V G G   1 6  ###########
###############################################################################
###############################################################################
#######   3 eme  F A C O N    D E     P R O C E D E R      ####################
###############################################################################


#Import the libraries

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import Model
from tensorflow.keras.applications import vgg16 as vgg
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



### we import scipy here to resize our images quickly

import scipy.misc

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

X_train = np.array([cv2.resize(x, (48,48), interpolation = cv2.INTER_AREA) for x in x_train])
X_test = np.array([cv2.resize(x, (48,48), interpolation = cv2.INTER_AREA) for x in x_test])



# perform our one hot encoding

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)



###############################################################################
#######  I M P O R T   P R E - T R A I N E D   M O D E L   ####################
###############################################################################


from tensorflow.keras.applications import vgg16 as vgg


# Generate a model with all layers (with top)
vgg16 = vgg.VGG16(weights=None, include_top=True, input_shape=(48,48,3))

#Add a layer where input is the output of the  second last layer 
x = Dense(10, activation='softmax', name='predictions')(vgg16.layers[-2].output)

#Then create the corresponding model 
my_model = Model(vgg16.input,x)
my_model.summary()




## Now lets compile our merged model
my_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

 

## We are now ready to prepare data augmentation configuration



train_datagen = ImageDataGenerator(rescale=1. /255,
                                   horizontal_flip=False)

## Now we use our Data Gen to get our data

train_datagen.fit(X_train)
train_gen = train_datagen.flow(X_train,
                               y_train,
                               batch_size=100)

val_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=False)
val_gen = val_datagen.flow(X_test,
                           y_test,
                           batch_size=100)


## We are noooow reaaaady to start the training 
train_steps_per_epoch = X_train.shape[0] // 100
val_steps_per_epoch = X_test.shape[0] // 100

history = my_model.fit_generator(train_gen,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_gen,
                              validation_steps=val_steps_per_epoch,
                              epochs=10,
                              verbose=1)





# loss
plt.plot(history['loss'], label='train loss')
plt.plot(history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(history['accuracy'], label='train acc')
plt.plot(history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')





















