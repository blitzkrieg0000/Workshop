#% Paketleri yükle
#$ pip install tensorflow
#$ pip install tensorflow-gpu

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras

#* BİLGİ
#(vscode extension: "aaron-bond.better-comments")
#? Tensorflow v2.0 dan itibaren çalışma modu olarak "Eager Execution" kullanır.
#? Eğer kodumuzun "Graph Execution" modunda çalışmasını istiyorsak, manuel olarak fonksiyonlarımızı değiştirmemiz gerekir.
#? Eager Execution modu herhangi bir tensorflow kütüphanesi ile yapılan işlemin tüm kaynağı kullanıp bir an önce bitirilmesi ile alakalıdır.
#? Graph Execution da ise birden çok işlem uzun süreli yapıalcaksa; bu mod kodu hızlandırmaya ve kaynakları verimli şekilde kullandırmaya yarar.
#? Bu modların gelişmesinde PyTorch(Facebook) dan esinlenmiş olunup; rekabetten doğan bir mimari ile son kullanıcıyı buluşturmuştur.
#? Kısa süreli işlemlerde kod yazarken EagerExecution kullanılması tavsiye edilir. Zaten Default olarak TF2.0 dan sonra, tüm fonksiyonlar bu mod ile çalışır.  
#? Ancak küçük işlemlerde (örnek: sadece bir katmanlı bir sinir ağı, basit matris işlemleri...) hız ve kaynak verimliliğini arttırmak için "Graph Execution" kullanılırsa kod daha da yavaş çalışacaktır.


#! Modeli oluştur
inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(64, activation="relu")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)


#! Optimizer oluştur
optimizer = keras.optimizers.SGD(learning_rate=1e-3)


#! Loss fonksiyonu oluştur
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)


#! Veri setini ayarla
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)


#! Training
epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Tüm verisetini döndür
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # İşlemleri kaydetmek için "GradientTape" contexti açıldı.
        with tf.GradientTape() as tape:
            #! a-) FeedForward yap
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            #! b-) Verilen Minibatch girdisi için loss hesapla.
            loss_value = loss_fn(y_batch_train, logits)

        #! c-) "GradientTape" ile loss a göre tüm eğitilebilir parametrelerin gradyanını hesapla.
        grads = tape.gradient(loss_value, model.trainable_weights)

        #! d-) Ağırlıklar hesaplanan gradyanalra göre güncellendi.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if step % 200 == 0:
            print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
            print("Seen so far: %s samples" % ((step + 1) * batch_size))



































































            