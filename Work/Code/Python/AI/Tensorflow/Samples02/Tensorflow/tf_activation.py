# Tensorflow'un warning loglarından kurtulmak için
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf



#! 1-SIGMOID --> Binary Classification, MultiLabel Classification
#* Sigmoid çıkış olarak giren değerleri 0 ve 1 arasına sıkıştırır.
#* +5 ve -5 girdilerden daha büyük ve daha küçük değerleri 0 ve 1 e yakınsatır.
arr = [-20, -1.0, 0.0, 1.0, 20]
tensor = tf.constant(arr, dtype=tf.float32) # Listeden Tensor tipine çevir. (Tensor: Tensorflow un kulandığı matris tipi.)
pred = tf.keras.activations.sigmoid(tensor)
results = pred.numpy()
print(f"{arr}\n<--Sigmoid-->\n{results}\n")


#! 2-SOFTMAX --> Multiclass Classification
#* Sigmoidden farklı olarak girdi olarak verilen tüm değerlerin toplamı "1" olacak şekilde 1 ve 0 arasına sıkıştırır.
#* Multiclass Classification larda bu yüzden yararlıdır.
arr = [[-20, -1.0, 4.5],[0.0, 1.0, 20]]
tensor = tf.constant(arr, dtype=tf.float32) # Listeden Tensor tipine çevir. (Tensor: Tensorflow un kulandığı matris tipi.)
pred = tf.keras.activations.softmax(tensor)
results = pred.numpy()
print(f"{arr}\n<--Softmax-->\n{results}\n")
