# -*- coding: utf-8 -*-
"""LAB2_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GrMELlc9hHUjVlLBSek1TYZJRIxPcl9q
"""

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from keras.optimizers import SGD, Adam, Adamax
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder
from keras import metrics
import matplotlib.pyplot as plt

dataset=pd.read_csv('drive/My Drive/Python_ICP/heart.csv')
dataset.head()
X = dataset.iloc[:,0:13]
Y = dataset.iloc[:,13]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.25, random_state=100)

# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# HyperParameters1
activation_function="tanh"
learning_rate=0.1
epochs=100
b_size=64
decay_rate= learning_rate / epochs
adam= Adam(lr=learning_rate, decay=decay_rate)

# Create Model
model = Sequential()
model.add(Dense(512, activation=activation_function, input_dim=13))
model.add(Dense(512, activation=activation_function))
model.add(Dense(512, activation=activation_function))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ["accuracy"])
tbCallBack = TensorBoard(log_dir='./lab2_1', histogram_freq=0, write_graph=True, write_images=True)
hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=b_size,callbacks=[tbCallBack])

# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# %load_ext tensorboard
# %tensorboard --logdir lab2_1

# HyperParameters2
activation_function="relu"
learning_rate=0.2
epochs=50
b_size=32
decay_rate= learning_rate / epochs
sgd= SGD(lr=learning_rate, decay=decay_rate)

# Create Model
model = Sequential()
model.add(Dense(512, activation=activation_function, input_dim=13))
model.add(Dense(512, activation=activation_function))
model.add(Dense(512, activation=activation_function))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ["accuracy"])
tbCallBack = TensorBoard(log_dir='./lab2_2', histogram_freq=0, write_graph=True, write_images=True)
hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=b_size,callbacks=[tbCallBack])

# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# %load_ext tensorboard
# %tensorboard --logdir lab2_2