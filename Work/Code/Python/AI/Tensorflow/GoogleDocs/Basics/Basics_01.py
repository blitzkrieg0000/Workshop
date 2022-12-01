import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print("TensorFlow bu bilgisayarda GPU'yu kullanabiliyor.")
else:
    print("TensorFlow bu bilgisayarda GPU'yu kullanamıyor.")


# 2x3 tipinde bir python listesi tanımlıyoruz.
arr:list[list[float]] = [
                            [1., 2., 3.],
                            [4., 5., 6.]
                        ]


#! 1-) Tensor tipi obje tanımlama
# Tensor objeleri sabittir, değiştirilemezler.
x = tf.constant(arr)
print("\n$-> Tensor Özellikleri(boyut, tip): ", x.shape, x.dtype)

#! Tensor İşlemleri
print("\n--> Toplama:\n", 
    x + x
)

print("\n--> Nokta Çarpım:\n",
    x * x
)

print("\n--> Matris Çarpımı:\n",
    x @ tf.transpose(x)    #Matris çarpımı için transpoze aldık. [(2,3) vs (3,2)]
)

print("\n--> Concat:\n",
    tf.concat([x, x, x], axis=0)
)

print("\n--> SoftmaxFunction('tf.nn' module: neural network):\n",
    tf.nn.softmax(x, axis=-1)
)

print("\n--> Elemanlar toplamı:\n",
    tf.reduce_sum(x)
)



#! 2-) Variable tipi obje tanımlama
# Tensorflow weightsleri depolamak için Variable değişkeni kullanır çünkü değiştirilebilirler.
var = tf.Variable(
        [0.0, 0.0, 0.0]
    )

#! Variable içeriğini değiştirme
var.assign([1, 2, 3])
print("\n--> Variable Objesi:\n", var)

#! Tüm değişkenlere ekleme yapma (ağırlık güncelleme yaparken etkili olabilir.)
var.assign_add([10, -10, 10])
print("\n--> Toplanmış Variable Objesi:\n", var)

#! Variable değerini Tensor olarak almaya yarar.
print("\n--> Variable Objesi Tensor değeri:\n", type(var), "-->" ,type(var.read_value()))








































































































