import cv2 as cv
import os
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow import keras

from keras.initializers import HeUniform
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, Dropout


from sklearn.model_selection import train_test_split

no_tumor = 'data/no'
tumor = 'data/yes'

dataset = []
label = []

for i, image in enumerate(os.listdir(no_tumor)):
    if image.split('.')[-1] != 'jpg':
        continue
    else:
        img = cv.imread(os.path.join(no_tumor, image))
        img = Image.fromarray(img, 'RGB')
        img = img.resize((64,64))
        dataset.append(np.array(img))
        label.append(0)

for i, image in enumerate(os.listdir(tumor)):
    if image.split('.')[-1] != 'jpg':
        continue
    else:
        img = cv.imread(os.path.join(tumor, image))
        img = Image.fromarray(img, 'RGB')
        img = img.resize((64,64))
        dataset.append(np.array(img))
        label.append(1)
        
dataset = np.array(dataset)
label = np.array(label)

# data splitting

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)
#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

# normalizing the data

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
# Model building

initializer = HeUniform()

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=x_train.shape[1:]))  # 64x64x3
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer=initializer))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer=initializer)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# Binary CrossEntropy: Dense = 1, Sigmoid Activation
# Categorical CrossEntropy: Dense = 2, Softmax Activation

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), shuffle=False)

model.save('BT10EpochsCategorical.h5')
