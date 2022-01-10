# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:36:12 2022

@author: kawta
"""



###############################################################################
###############################################################################
############# E X E M P L E      N°2   A V E C   C I F A R - 10  ##############
###############################################################################
###############################################################################


import tensorflow.keras
import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import regularizers


# Load CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# Model configuration for CIFAR-10 data
img_width, img_height, num_channels = 32, 32, 3
input_shape = (img_height, img_width, num_channels)
print(x_train.shape)


# Normalize data
x_train = x_train / 255
x_test = x_test / 255


# Convert target vectors to categorical targets
y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
y_test = tensorflow.keras.utils.to_categorical(y_test, 10)


# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.025))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.025))
model.add(Flatten())
model.add(Dense(256, activation='relu', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
model.add(Dense(10, activation='softmax', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))

### Compile the model 

model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit data to model
history = model.fit(x_train, y_train,
            batch_size=50,
            epochs=50,
            verbose=1,
            validation_split=0.2)




# Generate generalization metrics
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


# Plot history: Loss
plt.plot(history.history['loss'], label='Training data')
plt.plot(history.history['val_loss'], label='Validation data')
plt.title('L1/L2 Activity Loss')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

# Plot history: Accuracy
plt.plot(history.history['acc'], label='Training data')
plt.plot(history.history['val_acc'], label='Validation data')
plt.title('L1/L2 Activity Accuracy')
plt.ylabel('%')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()



