import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1]).cache().prefetch(tf.data.experimental.AUTOTUNE)



#dataset #<PrefetchDataset element_spec=TensorSpec(shape=(), dtype=tf.int32, name=None)>


#!Iterate Data
for elem in dataset:
    print(elem.numpy())

it = iter(dataset)
print(next(it).numpy())


#!Apply Process on All Data
#reduce(Başlangıç, lambda state, value: state + value),
#state : saved return
#value : iterate all items
data = dataset.reduce(0, lambda state, value: state + value).numpy() 
print(data)

#!Show DataSpects
dataset.element_spec



#! Dataset from a tensor(SparseTensor)
# Dataset containing a sparse tensor.
#tf.SparseTensor(indices, values, dense_shape)
tensor = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
dataset4 = tf.data.Dataset.from_tensors(tensor)
dataset4 = dataset4.cache().prefetch(tf.data.experimental.AUTOTUNE)
dataset4.element_spec

#!MAPPING

dataset1 = tf.data.Dataset.from_tensor_slices( tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32) )


for z in dataset1:
  print(z.numpy())

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))


for a, (b,c) in dataset3:
  print(f'shapes: {a.shape}, {b.shape}, {c.shape}')