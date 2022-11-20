import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import datetime
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras


physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.compat.v1.reset_default_graph()
# tf.keras.backend.clear_session()

LABELS = ["desert", "mountains", "sea", "sunset", "trees"]
df = pd.read_csv("Datasets/miml_dataset/miml_labels_1.csv")
print(df.head())


data_dir = pathlib.Path("Datasets/miml_dataset/")
filenames = list(data_dir.glob('images/*.jpg'))
fnames=[]
for fname in filenames:
    fnames.append(str(fname))

ds_size = len(fnames)
print("Number of images in folders: ", ds_size)


BATCH_SIZE = 64
IMG_WIDTH, IMG_HEIGHT = 64, 64
filelist_ds = tf.data.Dataset.from_tensor_slices(fnames)
ds_size = filelist_ds.cardinality().numpy()
print("Number of selected samples for dataset: ", ds_size)



def get_label(file_path):
    parts = tf.strings.split(file_path, '/')
    file_name = parts[-1]

    labels = df[df["Filenames"]==file_name][LABELS].to_numpy()
    print(f"Label:> {labels}")
    labels = labels.squeeze()
    print(f"Label Shape:> {labels.shape}")
    return tf.convert_to_tensor(labels)


def combine_images_labels(file_path: tf.Tensor):
    print(F"File PATH: {file_path}")
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3) 
    img = tf.image.convert_image_dtype(img, tf.float32) 
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    print(img.shape, label.shape)
    return img, label


train_ratio = 0.80
ds_train = filelist_ds.take(ds_size*train_ratio)
ds_test = filelist_ds.skip(ds_size*train_ratio)

#GRAPH EXECUTION
combine_images_labels_graph = tf.function(combine_images_labels)

ds_train = ds_train.map(lambda x: combine_images_labels_graph(x), #tf.py_function(func=combine_images_labels, inp=[x], Tout=(tf.float32, tf.int64)),
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=False)


ds_test = ds_test.map(lambda x: combine_images_labels_graph(x),
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=False)



def covert_onehot_string_labels(label_string,label_onehot):
    labels=[]
    for i, label in  enumerate(label_string):
        if label_onehot[i]:
            labels.append(label)
    if len(labels)==0:
        labels.append("NONE")
    return labels


def show_samples(dataset):
    fig=plt.figure(figsize=(16, 16))
    columns = 3
    rows = 3
    print(columns*rows,"samples from the dataset")
    i=1
    for a, b in dataset.take(columns*rows):
        print(a,b)
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(a))
        #plt.title("image shape:"+ str(a.shape)+" ("+str(b.numpy()) +") "+ str(covert_onehot_string_labels(LABELS,b.numpy())))
        i=i+1
    plt.show()

# show_samples(ds_test)

# ...2]]], shape=(64, 64, 3), dtype=float32) tf.Tensor([], shape=(0, 5), dtype=int64)
# ValueError: `logits` and `labels` must have the same shape, received ((None, 5) vs (None, 0, 5)).


#buffer_size = ds_train_resize_scale.cardinality().numpy()/10
#ds_resize_scale_batched=ds_raw.repeat(3).shuffle(buffer_size=buffer_size).batch(64, )
ds_train_batched=ds_train.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE) 
ds_test_batched=ds_test.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)
print("Number of batches in train: ", ds_train_batched.cardinality().numpy())
print("Number of batches in test: ", ds_test_batched.cardinality().numpy())



base_model = keras.applications.VGG16(weights='imagenet', input_shape=(64, 64, 3), include_top=False)
base_model.trainable = False


number_of_classes = 5
inputs = keras.Input(shape=(64, 64, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
initializer = tf.keras.initializers.GlorotUniform(seed=42)

activation =  tf.keras.activations.sigmoid #None  # tf.keras.activations.sigmoid or softmax

outputs = keras.layers.Dense(number_of_classes, kernel_initializer=initializer, activation=activation)(x) 
model = keras.Model(inputs, outputs)



model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()]
)


model.fit(ds_train_batched, validation_data=ds_test_batched, epochs=100)



ds = ds_test_batched
print("Test Accuracy: ", model.evaluate(ds)[1])

ds = ds_test
predictions= model.predict(ds.batch(batch_size=10).take(1))
print("A sample output from the last layer (model) ", predictions[0])
y=[]
print("10 Sample predictions:")
for (pred,(a,b)) in zip(predictions,ds.take(10)):
    pred[pred>0.5]=1
    pred[pred<=0.5]=0
    print("predicted: " ,pred, str(covert_onehot_string_labels(LABELS, pred)),  
        "Actual Label: ("+str(covert_onehot_string_labels(LABELS,b.numpy())) +")")
    y.append(b.numpy())
