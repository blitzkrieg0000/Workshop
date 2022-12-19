import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn import model_selection, metrics
from keras import layers

#https://machinelearningmastery.com/neural-network-models-for-combined-classification-and-regression/

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv'
dataframe = pd.read_csv(url, header=None)

print(dataframe.shape)
print(dataframe.head())

X, y = dataframe.iloc[:, 1:-1], dataframe.iloc[:, -1]
X, y = X.astype('float'), y.astype('float')
n_features = X.shape[1]

#!###########
#! REGRESSION
#!###########


# X-Y
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=1)


# Model oluşturma
model = keras.Sequential()
model.add(layers.Dense(20, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
model.add(layers.Dense(10, activation='relu', kernel_initializer='he_normal'))
model.add(layers.Dense(1, activation='linear'))


# Model Compile
model.compile(loss='mse', optimizer='adam')


# Modeli Eğitme
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=2)


# evaluate on test set
yhat = model.predict(X_test)
error = metrics.mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % error)







































































































