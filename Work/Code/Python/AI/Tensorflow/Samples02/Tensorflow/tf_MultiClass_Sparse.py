# Tensorflow'un warning loglarından kurtulmak için
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow import keras
import matplotlib.pyplot as plt


[ds_raw_train, ds_raw_test], info = tfds.load('mnist', split=['train[:10%]','test[:10%]'], as_supervised=True, with_info=True)
print("Number of samples in train : ", ds_raw_train.cardinality().numpy(), " in test : ", ds_raw_test.cardinality().numpy())
print("Number of classes/labels: ",info.features["label"].num_classes)
print("Names of classes/labels: ",info.features["label"].names)
labels= info.features["label"].names


def show_samples(dataset):
    fig = plt.figure(figsize=(16, 16))
    columns = 3
    rows = 3
  
    print(columns*rows,"samples from the dataset")
    i=1
    for a,b in dataset.take(columns*rows): 
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(a))
        #plt.imshow(a.numpy())
        plt.title("image shape:"+ str(a.shape)+" ("+str(b.numpy()) +")" )
    i = 1+i
    plt.show()
# show_samples(ds_raw_test)


#VGG16 expects min 32 x 32 
def resize_scale_image(image, label):
    image = tf.image.resize(image, [32, 32])
    image = image/255.0
    image = tf.image.grayscale_to_rgb(image)
    return image, label


ds_train_resize_scale = ds_raw_train.map(resize_scale_image)
ds_test_resize_scale = ds_raw_test.map(resize_scale_image)
# show_samples(ds_test_resize_scale)


batch_size = 64 
ds_train_resize_scale_batched = ds_train_resize_scale.batch(batch_size, drop_remainder=True ).cache().prefetch(tf.data.experimental.AUTOTUNE)
ds_test_resize_scale_batched = ds_test_resize_scale.batch(batch_size, drop_remainder=True ).cache().prefetch(tf.data.experimental.AUTOTUNE)
print("Number of batches in train: ", ds_train_resize_scale_batched.cardinality().numpy())
print("Number of batches in test: ", ds_test_resize_scale_batched.cardinality().numpy())

print(type(ds_test_resize_scale_batched))

base_model = keras.applications.VGG16( weights='imagenet', input_shape=(32, 32, 3), include_top=False)
base_model.trainable = False


number_of_classes = 10
inputs = keras.Input(shape=(32, 32, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
initializer = tf.keras.initializers.GlorotUniform(seed=42)

activation =  None  # tf.keras.activations.sigmoid or softmax
outputs = keras.layers.Dense(number_of_classes, kernel_initializer=initializer, activation=activation)(x) 
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # default from_logits=False
              metrics=[keras.metrics.SparseCategoricalAccuracy()])


model.fit(ds_train_resize_scale_batched, validation_data=ds_test_resize_scale_batched, epochs=3)


ds = ds_test_resize_scale
print("Test Accuracy: ", model.evaluate(ds.batch(batch_size=10))[1])
predictions= model.predict(ds.batch(batch_size=10).take(1))
y=[]
print("10 Sample predictions:")
for (pred,(a,b)) in zip(predictions,ds.take(10)):
    print("predicted: " , np.argmax(pred), "Actual Label: "+labels[b.numpy()]+" ("+str(b.numpy()) +")", " True" if (np.argmax(pred)==b.numpy()) else " False" )
    y.append(b.numpy())