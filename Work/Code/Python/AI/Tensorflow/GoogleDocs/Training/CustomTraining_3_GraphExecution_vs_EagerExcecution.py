import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

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
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs)


#! Optimizer oluştur
optimizer = keras.optimizers.SGD(learning_rate=1e-3)


#! Loss fonksiyonu oluştur
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)


#! Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()


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



@tf.function
def train_step(x_batch_train, y_batch_train):
    with tf.GradientTape() as tape:
        #! a-) FeedForward yap
        logits = model(x_batch_train, training=True)

        #! b-) Verilen Minibatch girdisi için loss hesapla.
        loss_value = loss_fn(y_batch_train, logits)

    #! c-) "GradientTape" ile loss a göre tüm eğitilebilir parametrelerin gradyanını hesapla.
    grads = tape.gradient(loss_value, model.trainable_weights)

    #! d-) Ağırlıklar hesaplanan gradyanalra göre güncellendi.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    #! e-) Training Metric'leri her minibatchten sonra hesapla ve güncelle.
    # Training Metric'ler, modele eğitim için giren input verisinin; batch (32, 3, 1920,1080) veya tek bir tane (1, 3, 1920,1080) verinin
    #olmasına bakmaksızın her tür feedforward ve ardından gelen backward(eğitim) işlemleri sonrası hesaplanarak, ana metricleri (accuracy vs.) günceller.
    # Kısaca her bir veri ile eğitim sonrası training metricler hesaplanır.
    train_acc_metric.update_state(y_batch_train, logits)
    return loss_value


@tf.function
def test_step(x_batch_val, y_batch_val):
    #! a-) Feedforward yap fakat training deki gibi ağırlıkları güncelleme.
    val_logits = model(x_batch_val, training=False)

    #! b-)Validation Metric'leri güncelle.
    val_acc_metric.update_state(y_batch_val, val_logits)



#! Training
epochs = 2
for epoch in range(epochs):
    print("\nEpoch başladı %d" % (epoch,))
    start_time = time.time()

    #! Tüm verisetini döndür
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)

        if step % 200 == 0:
            print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
            print("Eğitilen veri: %d samples" % ((step + 1) * batch_size))

    #! Her epoch sonunda Training Metric'leri göster.
    train_acc = train_acc_metric.result()
    print("Her epochtaki Training acc: %.4f" % (float(train_acc),))

    #! Her epoch sonunda Training Metric'leri sıfırla. 
    # Tipik olarak eğitim kuralları gereği her epoch sonrası training accuracy sıfırlanmak zorundadır.
    train_acc_metric.reset_states()

    #! Validation veri seti ile validation metriclerin hesaplanması yapılır.
    # Training metriclerden farklı olarak her bir feedforward sonrası _eğitim yapılmadan_ yani ağırlıklar güncellenmeden 
    #metric hesabı yapılır.
    #Her epoch sonrasında yapılır.
    #  Validation ın amacı, modelin tüm eğitim setini tükettikten sonra; yapılan eğitim bu model, bu hiperparametreleri kullanarak
    #öğreniyor mu yoksa ezberliyor mu ya da öğrenmiyor mu (Overfitting, Underfitting)? Kanısına varabilmektir.
    #Validation değerlerine göre parametreler değiştirilir.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)
    
    #! Validation Metric'leri göster.
    val_acc = val_acc_metric.result()

    #! Validation Metric'leri sıfırla.
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Geçen zaman: %.2fs" % (time.time() - start_time))




























