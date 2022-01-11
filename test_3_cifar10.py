# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:36:13 2022

@author: kawta
"""



###############################################################################
###############################################################################
############# E X E M P L E      N°3   A V E C   C I F A R - 10  ##############
###############################################################################
###############################################################################


import tensorflow.keras
import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
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


# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, activation='relu', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
model.add(Dense(10, activation='softmax', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))

### Compile the model 

model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit data to model
history = model.fit(x_train, y_train,
            batch_size=1024,
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


## Prediction test
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
somme=0
for i in range(0,2) :
    somme+=cm[i,i]
precision=somme/2000


###############################################################################
###############################################################################
#############         O  P  T  I  M  I  S  A  T  I  O  N         ##############
###############################################################################
###############################################################################



from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout




model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer = 'uniform', activation='relu', input_shape=input_shape, activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3),kernel_initializer = 'uniform', activation='relu', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3),kernel_initializer = 'uniform', activation='relu', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, activation='relu', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
model.add(Dense(10, activation='softmax', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))








def create_network():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer = 'uniform', activation='relu', input_shape=input_shape, activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3),kernel_initializer = 'uniform', activation='relu', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3),kernel_initializer = 'uniform', activation='relu', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
    model.add(Dense(10, activation='softmax', activity_regularizer=regularizers.l1_l2(l1=0.04, l2=0.01)))
    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,optimizer=tensorflow.keras.optimizers.Adam(),metrics=['accuracy'])
    return network

network=KerasClassifier(build_fn=create_network,batch_size=1024,epochs=10)
precision2=cross_val_score(estimator=model,x_train=x_train,y_train=y_train,cv=10)

moyenne2=precision2.mean()
ecart_type2=precision2.std()

from sklearn.model_selection import GridSearchCV

def create_network_opt(algo_gradient,nb1,nb2):
    network = Sequential()
    network.add(Dense(units =nb1, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
    network.add(Dropout(rate=0.2))
    network.add(Dense(units =nb2, kernel_initializer = 'uniform', activation = 'relu'))
    network.add(Dropout(rate=0.2))
    network.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    network.compile(optimizer = algo_gradient, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return network

network3=KerasClassifier(build_fn=create_network_opt)
hyperparam={'batch_size':[10,25],'epochs':[10,50],'algo_gradient':['adam','rmsprop'],
            'nb1':[6,10],'nb2':[6,10]}

best_precision=grid_search.best_score_
best_params=grid_search.best_params_



precision3=cross_val_score(estimator=network,X=X,y=y,cv=10)

moyenne3=precision3.mean()
ecart_type3=precision3.std()
