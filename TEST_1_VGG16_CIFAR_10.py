# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:37:21 2022

@author: kawta
"""

###############################################################################
#######  T R A N S F E R   L E A R N I N G   W I T H   V G G   1 6  ###########
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



## import keras functions 

batch_size = 32
epochs = 10
num_class = 10
learning_rate = 1e-4
momentum = 0.9



### we import scipy here to resize our images quickly

import scipy.misc

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

X_train = np.array([cv2.resize(x, (48,48), interpolation = cv2.INTER_AREA) for x in x_train])
X_test = np.array([cv2.resize(x, (48,48), interpolation = cv2.INTER_AREA) for x in x_test])



# perform our one hot encoding

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_class)
y_test = to_categorical(y_test, num_class)



###############################################################################
#######  I M P O R T   P R E - T R A I N E D   M O D E L   ####################
###############################################################################


from tensorflow.keras.applications import vgg16 as vgg




###############################################################################
#######   2 eme  F A C O N    D E     P R O C E D E R      ####################
###############################################################################


# Pretrained convolutional layers are loaded using the Imagenet weights.
# Include_top is set to False, in order to exclude the model's fully-connected layers.
base_model = vgg.VGG16(weights = 'imagenet',
                       include_top=False,
                       input_shape=(48,48,3))
                       

## Extract the last layer from third block of vgg16 model
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
last = base_model.get_layer('block5_pool').output



## Add classification layers on top of it (i.e. fully connected layers)
#  x is our 'model' that we're putting ontop part of our pre-trained model (above)
# This is 'bootstrapping' a new top_model onto the pretrained layers.

x = GlobalAveragePooling2D()(last)
x = BatchNormalization()(x)
x = Dense(256, activation = 'relu')(x)
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.5)(x)

## We make our top model 
num_class= 10
top = Dense(num_class, activation = 'softmax')(x)


# Construct our full model now
model = Model(base_model.input, top)



###############################################################################
#######   2 eme  F A C O N    D E     P R O C E D E R      ####################
###############################################################################



# Generate a model with all layers (with top)
vgg16 = VGG16(weights=None, include_top=True)

#Add a layer where input is the output of the  second last layer 
x = Dense(8, activation='softmax', name='predictions')(vgg16.layers[-2].output)

#Then create the corresponding model 
my_model = Model(input=vgg16.input, output=x)
my_model.summary()


















###############################################################################
#######   2 eme  F A C O N    D E     P R O C E D E R      ####################
###############################################################################



#top_model = conv_base.output
#top_model = Flatten(name="flatten")(top_model)
#top_model = Dense(4096, activation='relu')(top_model)
#top_model = Dense(1072, activation='relu')(top_model)
#top_model = Dropout(0.2)(top_model)
#output_layer = Dense(n_classes, activation='softmax')(top_model)
    
# Group the convolutional base and new fully-connected layers into a Model object.
#model = Model(inputs=conv_base.input, outputs=output_layer)






# we now just iterate through our base model to 'freeze' the layers so that we don't train them 
for layer in base_model.layers:
    layer.trainable = False
    

## Now lets compile our merged model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

model.summary()
 

## We are now ready to prepare data augmentation configuration

train_datagen = ImageDataGenerator(rescale=1. /255,
                                   horizontal_flip=False)

## Now we use our Data Gen to get our data

train_datagen.fit(X_train)
train_gen = train_datagen.flow(X_train,
                               y_train,
                               batch_size=batch_size)

val_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=False)
val_gen = val_datagen.flow(X_test,
                           y_test,
                           batch_size=batch_size)


## We are noooow reaaaady to start the training 
train_steps_per_epoch = X_train.shape[0] // batch_size
val_steps_per_epoch = X_test.shape[0] // batch_size

history = model.fit_generator(train_gen,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_gen,
                              validation_steps=val_steps_per_epoch,
                              epochs=epochs,
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



















