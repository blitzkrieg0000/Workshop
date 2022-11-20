# Tensorflow'un warning loglarından kurtulmak için
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import (Input, Model, activations, applications, layers, losses,metrics, optimizers)

# Data Pipeline
df = pd.read_csv("Dataset/miml_dataset/miml_labels_1.csv")
print(df.head())
LABELS = ["desert", "mountains", "sea", "sunset", "trees"]


data_dir = pathlib.Path("Dataset/miml_dataset")
filenames = list(data_dir.glob('images/*.jpg'))
fnames = []
for fname in filenames:
    fnames.append(str(fname))
ds_size = len(fnames)
print("Klasördeki veri sayısı: ", ds_size)


number_of_selected_samples = 2000
filelist_ds = tf.data.Dataset.from_tensor_slices(fnames) # [:number_of_selected_samples] Dosya adları verilir.
ds_size = filelist_ds.cardinality().numpy()
print("Dataset için seçilen veri sayısı: ", ds_size)


def get_label(file_path):
    parts = tf.strings.split(file_path, '/')
    file_name = parts[-1]
    labels = df[df["Filenames"]==file_name][LABELS].to_numpy().squeeze()
    return tf.convert_to_tensor(labels)


IMG_WIDTH, IMG_HEIGHT = 64 , 64
def decode_img(img):
    #color images
    img = tf.image.decode_jpeg(img, channels=3) 
    #convert unit8 tensor to floats in the [0,1]range
    img = tf.image.convert_image_dtype(img, tf.float32) 
    #resize 
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]) 


def combine_images_labels(file_path: tf.Tensor):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# Veriyi train ve test olarak böl
train_ratio = 0.80
dsTrain = filelist_ds.take(ds_size*train_ratio)
dsTest = filelist_ds.skip(ds_size*train_ratio)

BATCH_SIZE = 64
dsTrain = dsTrain.map(lambda x: tf.py_function(func=combine_images_labels, inp=[x], Tout=(tf.float32, tf.int64)),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False
            )


dsTest = dsTest.map(lambda x: tf.py_function(func=combine_images_labels, inp=[x], Tout=(tf.float32,tf.int64)),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )


def covert_onehot_string_labels(label_string,label_onehot):
    labels=[]
    for i, label in  enumerate(label_string):
        if label_onehot[i]:
            labels.append(label)
    if len(labels)==0:
        labels.append("NONE")
    return labels


def show_samples(dataset):
    fig = plt.figure(figsize=(16, 16))
    columns = 3
    rows = 3
    print(columns*rows,"samples from the dataset")
    i=1
    for a,b in dataset.take(columns*rows): 
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(a))
        plt.title("image shape:"+ str(a.shape)+" ("+str(b.numpy()) +") " + str(covert_onehot_string_labels(LABELS, b.numpy())))
        i=i+1
    plt.show()

show_samples(dsTest)


#buffer_size = dsTrain_resize_scale.cardinality().numpy()/10
#ds_resize_scale_batched=ds_raw.repeat(3).shuffle(buffer_size=buffer_size).batch(64, )

dsTrain_batched = dsTrain.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE) 
dsTest_batched = dsTest.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)

print("Number of batches in train: ", dsTrain_batched.cardinality().numpy())
print("Number of batches in test: ", dsTest_batched.cardinality().numpy())


#! MODEL
base_model = applications.VGG16(
    weights='imagenet',             # Load weights pre-trained on ImageNet.
    input_shape=(64, 64, 3),        # VGG16 expects min 32 x 32
    include_top=False)              # Do not include the ImageNet classifier at the top.
base_model.trainable = False

number_of_classes = 5

inputs = Input(shape=(64, 64, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
initializer = tf.keras.initializers.GlorotUniform(seed=42)

activation =  activations.sigmoid #None  # tf.keras.activations.sigmoid or softmax

outputs = layers.Dense(number_of_classes, kernel_initializer=initializer, activation=activation)(x) 
model = Model(inputs, outputs)


model.compile(optimizer=optimizers.Adam(), loss=losses.BinaryCrossentropy(), metrics=[metrics.BinaryAccuracy()])

model.fit(dsTrain_batched, validation_data=dsTest_batched, epochs=100)


#! TEST
ds = dsTest_batched
print("Test Accuracy: ", model.evaluate(ds)[1])

predictions= model.predict(ds.batch(batch_size=10).take(1))
print("A sample output from the last layer (model) ", predictions[0])
y=[]
print("10 Sample predictions:")
for (pred,(a,b)) in zip(predictions,ds.take(10)):
    pred[pred>0.5]=1
    pred[pred<=0.5]=0
    print("predicted: " ,pred, str(covert_onehot_string_labels(LABELS, pred)), "Actual Label: ("+str(covert_onehot_string_labels(LABELS,b.numpy())) +")")
    y.append(b.numpy())
