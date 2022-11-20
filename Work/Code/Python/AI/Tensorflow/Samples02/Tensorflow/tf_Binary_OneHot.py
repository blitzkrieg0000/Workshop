# Tensorflow'un warning loglarından kurtulmak için
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

def LoadData():
    ds_raw_train, ds_raw_test = tfds.load('horses_or_humans', split=['train','test'], as_supervised=True)
    print("Number of samples in train : ", ds_raw_train.cardinality().numpy(), " in test : ",ds_raw_test.cardinality().numpy())
    return ds_raw_train, ds_raw_test

def show_samples(dataset):
    fig=plt.figure(figsize=(14, 14))
    columns = 3
    rows = 3
  
    print(columns*rows,"samples from the dataset")
    i=1
    for a,b in dataset.take(columns*rows): 
        fig.add_subplot(rows, columns, i)
        plt.imshow(a)
        #plt.imshow(a.numpy())
        # plt.title("image shape:"+ str(a.shape)+" Label:"+str(b.numpy()) )
        # i=i+1
    plt.show()


#VGG16 expects min 32 x 32 
def resize_scale_image(image, label):
    image = tf.image.resize(image, [32, 32])
    image = image/255.0
    return image, label

def one_hot(image, label):
  label = tf.one_hot(label, depth=2)
  return image, label


if '__main__' == __name__:
    ds_raw_train, ds_raw_test = LoadData()
    ds_train_resize_scale = ds_raw_train.map(resize_scale_image)
    ds_test_resize_scale = ds_raw_test.map(resize_scale_image)
    print(ds_test_resize_scale)
    #show_samples(ds_test_resize_scale)

    batch_size = 64 
    ds_train_resize_scale_one_hot = ds_train_resize_scale.map(one_hot)
    ds_test_resize_scale_one_hot = ds_test_resize_scale.map(one_hot)

    # show_samples(ds_test_resize_scale_one_hot)
    ds_train_resize_scale_one_hot_batched = ds_train_resize_scale_one_hot.batch(batch_size)
    ds_test_resize_scale_one_hot_batched = ds_test_resize_scale_one_hot.batch(batch_size)

    # VGG16 expects min 32 x 32
    # Do not include the ImageNet classifier at the top.
    base_model = keras.applications.VGG16(weights='imagenet', input_shape=(32, 32, 3), include_top=False)  
    base_model.trainable = False


    inputs = keras.Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)

    #! Binary Classification OneHot [1, 0]
    activation = None  #  tf.keras.activations.sigmoid or softmax
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    outputs = keras.layers.Dense(2, kernel_initializer=initializer, activation=activation)(x) #2 çıkış çünkü OneHot --> [1, 0]
    
    model = keras.Model(inputs, outputs)


    model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.BinaryCrossentropy(from_logits=True), # default from_logits=False #CategoricalCrossentropy de kullanılabilir fakat BinaryCrossentropy kadar kesin sonuç vermeyecektir çünkü C.C.E. birden fazla sınıflar için daha uygun olduğundan yanlış ölçümler yapacaktır.
            metrics=[keras.metrics.CategoricalAccuracy()]
    )

    model.fit(ds_train_resize_scale_one_hot_batched, validation_data=ds_test_resize_scale_one_hot_batched, epochs=100)
    model.summary()
    
    model.evaluate(ds_test_resize_scale_one_hot_batched)


