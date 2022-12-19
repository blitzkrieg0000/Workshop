import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn import metrics
from tensorflow import keras


#! BOSTON HOUSING VERİ SETİNİ YÜKLE
(train_features, train_labels), (test_features, test_labels) = keras.datasets.boston_housing.load_data()

print("\n-> Veri seti x :\n",
    pd.DataFrame(train_features).head()
)

print("\n-> Veri seti y :\n",
    pd.DataFrame(train_labels).head()
)


feature_size = len(train_features[0])

#! Model Oluşturma
# Veri seti çok büyük olmadığı için temel bir model bu işi çözebilir.
model = keras.Sequential()
model.add(layers.Dense(10, activation="relu", input_shape=[feature_size]))
model.add(layers.Dense(15, activation="relu", input_shape=[10]))
model.add(layers.Dense(1))


#! Modeli Derleme
model.compile(optimizer="adam", loss="mse", metrics=["mse"])


#! Modeli Eğitme
history = model.fit(train_features, train_labels, epochs=1000, verbose=0)


#! Modeli Test Etme
pred = model.predict(test_features)

print("\n-> Loss :\n",
    metrics.mean_squared_error(pred, test_labels)
)


print("\n->  :\n",
    model.predict([[0.02177,  82.5,   2.03,  0.0,  0.415,  7.610,   15.7,  6.2700,   2.0,  348.0,  14.7,  395.38,   3.11]])
)


































