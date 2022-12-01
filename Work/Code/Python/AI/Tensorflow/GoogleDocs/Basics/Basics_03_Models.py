import tensorflow as tf


#! Bir modül/model oluştur.
class MyModule(tf.Module):
    def __init__(self, value):
        self.weight = tf.Variable(value)

    @tf.function
    def multiply(self, x):
        return x * self.weight

mod = MyModule(3)
result = mod.multiply(tf.constant([1, 2, 3]))
print(result)


#! Modeli Kaydet
save_path = 'Basics/saved_model'
tf.saved_model.save(mod, save_path)

#! Modeli Yükle ve Kullan
reloaded = tf.saved_model.load(save_path)
results = reloaded.multiply(tf.constant([1, 2, 3]))
print(results)