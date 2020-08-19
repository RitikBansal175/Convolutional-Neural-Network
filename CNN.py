# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:18:04 2020

@author: devil may cry
"""

#Importing the Libraries
from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten 
from keras.layers import Dense

#Initializing the Neural network
classifier = Sequential()

#adding Convolution layer
classifier.add(Convolution2D(64, 3, 3, input_shape =(64,64,3), activation ='relu' ))

#Applying MaxPooling
classifier.add(MaxPooling2D(pool_size =(2,2), strides = 2))

#Flattening the Feature Maps
classifier.add(Flatten())

#Building the ann Fully Connected Layer
classifier.add(Dense(output_dim=64, activation ='relu'))
classifier.add(Dense(output_dim=1, activation ='sigmoid'))

#Compiling the neural network
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics= ['accuracy'])

#Fitting the CNN to the image dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit(
        training_set,
        steps_per_epoch=8000,
        epochs=10,
        validation_data=test_set,
        validation_steps=2000)
