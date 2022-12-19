import pandas as pd
import numpy as np


import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


(train_features, train_labels), (test_features, test_labels) = keras.datasets.boston_housing.load_data()
pd.DataFrame(train_features).head()






































