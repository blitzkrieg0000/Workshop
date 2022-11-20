# Tensorflow'un warning loglarından kurtulmak için
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from keras import applications, Input, layers, activations
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
    for a, b in dataset.take(columns*rows): 
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(a))
        #plt.imshow(a.numpy())
        plt.title("image shape:"+ str(a.shape)+" ("+str(b.numpy()) +")" )
    i = 1+i
    plt.show()
# show_samples(ds_raw_test)


def one_hot(image, label):
    label = tf.one_hot(label, depth=10)
    return image, label

#VGG16 expects min 32 x 32 
def resize_scale_image(image, label):
    image = tf.image.resize(image, [32, 32])
    image = image/255.0
    image = tf.image.grayscale_to_rgb(image)
    return image, label


ds_train_resize_scale = ds_raw_train.map(resize_scale_image)
ds_test_resize_scale = ds_raw_test.map(resize_scale_image)
# show_samples(ds_test_resize_scale)

ds_train_resize_scale_one_hot= ds_train_resize_scale.map(one_hot)
ds_test_resize_scale_one_hot= ds_test_resize_scale.map(one_hot)


batch_size = 64 
ds_train_resize_scale_one_hot_batched=  ds_train_resize_scale_one_hot.batch(64)
ds_test_resize_scale_one_hot_batched = ds_test_resize_scale_one_hot.batch(64)


base_model = applications.VGG16(weights='imagenet', input_shape=(32, 32, 3), include_top=False)
base_model.trainable = False


number_of_classes = 10
inputs = Input(shape=(32, 32, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)

initializer = tf.initializers.GlorotUniform(seed=42)
activation = activations.softmax # None  #  tf.activations.sigmoid or softmax

outputs = layers.Dense(number_of_classes, kernel_initializer=initializer, activation=activation)(x) 
 
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(), # default from_logits=False
              metrics=[tf.keras.metrics.CategoricalAccuracy()])


model.fit(ds_train_resize_scale_one_hot_batched, validation_data=ds_test_resize_scale_one_hot_batched, epochs=20)


ds = ds_test_resize_scale_one_hot
print("Test Accuracy: ", model.evaluate(ds.batch(batch_size=10))[1])
print("10 Sample predictions ")
predictions= model.predict(ds.batch(batch_size=10).take(1))
y=[]
for (pred,(a,b)) in zip(predictions,ds.take(10)):
  print("predicted: " , (pred), "Actual Label: "+str(b.numpy()) , " True" if (np.argmax(pred)==np.argmax(b.numpy())) else " False" )
  print()
  y.append(b.numpy())