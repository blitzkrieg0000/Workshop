import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds


#Parameters
dataset_path = "Datasets/flower_photos"  # Veriseti klasörü yolu
batch_size = 32                          # Eğitimde kullanılacak veri girişinin "batch" boyutu
validation_split = 0.2  
img_height = 180
img_width = 180



#! Check Data Path
data_dir = pathlib.Path(dataset_path)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)



#! 1-)Load Data: "Veriler tek bir klasör içerisindedir ve her bir sınıf, alt klasör olacak şekilde yerleştirilir."
#*  ├── flower_photos
#*  │   ├── daisy
#?  │       ├── 1.jpg
#?  │       ├── 2.jpg
#*  │   ├── dandelion
#*  │   ├── roses
#*  │   ├── sunflowers
#*  │   └── tulips

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split,
    subset="both",
    seed=123,
    image_size=(img_height, img_width),
    interpolation="bilinear",
    batch_size=batch_size
)# -> <class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>



#Show Details
class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")



#! 2-) Preprocessing
normalization_layer = tf.keras.layers.Rescaling(1.0/255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))



#! 3-) Dataset Tuning
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)



#* 1-) PrepareModel
num_classes = 5
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])


#* 2-) Set model parameters
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


#* 3-) Train Model
model.fit(train_ds,validation_data=val_ds,epochs=100)


