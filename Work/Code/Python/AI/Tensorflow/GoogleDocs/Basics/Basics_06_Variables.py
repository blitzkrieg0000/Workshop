import tensorflow as tf


#! Değişken üretme
my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)
bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])


#! Değişken atama
a = tf.Variable([2.0, 3.0])
b = tf.Variable(a)
a.assign([5, 6])  #Tamamen değiştirir.

print(a.numpy())
print(b.numpy())

# Ekleme yapar
print(a.assign_add([2,3]).numpy())  # Ekleme
print(a.assign_sub([7,9]).numpy())  # Çıkartma



#! Yaşam döngüleri, adlandırma ve izlemeYaşam döngüleri, adlandırma ve izleme
# Variable lara aynı ismi verebiliriz. Fakat farklı tensorler tutacaktır.
a = tf.Variable(my_tensor, name="Mark")
b = tf.Variable(my_tensor + 1, name="Mark")

#Element bazında eşit değillerdir.
print(a == b)

#Not: Variable isimleri, model(örnek: sinir ağı) kaydedildiğinde ve geri yüklendiğinde korunur.
#Bir model oluşturduğumuzda variable isimleri otomatik olarak oluşturulur. 

#Bazı variable'lar ın gradyan hesabı gereksizdir. Bu yüzden bu hesaplama özelliği kapatılabilir.
step_counter = tf.Variable(1, trainable=False)


#! Otomatik olarak kodlar GPU ile çalışılır. Eğer CPU da çalışmasını istiyorsak bu ContextManager'ı çağırmalıyız.
with tf.device('CPU:0'):
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
print(c)

# Eğer farklı cihazlarda; örneğin CPU ve GPU raminde bilgi tutacaksak ve işlem yapacaksak bu gecikmeye sebep olacaktır.
with tf.device('CPU:0'):
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.Variable([[1.0, 2.0, 3.0]])
with tf.device('GPU:0'):
    # Element-wise multiply
    k = a * b
print(k)

# NOT: "tf.config.set_soft_device_placement" özelliği varsayılan olarak açık olduğundan GPU olmayan bir cihaz üzerinde bu kod çalışmaya devam edecektir.




















































































