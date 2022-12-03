import tensorflow as tf
import numpy as np


#! Düzenli Tensor tipleri
# Tensorler değiştirilemez değişkenlerdir. Ancak yeni bir atama yapılarak değiştirilir.
# Boyutları tüm eksenlerde aynı olmalıdır. Yani farklı uzunluktaki listeleri tutamaz 
# -> Şunun gibi olamaz: [[1,2], [1,2,3,4]] bu matrisin boyutu ne; kaça kaç: ???

# tf.constant ile değişken tanımlamak varsayılan olarak int32 tipinde bir tensor oluşturur.
# No axes, boyutsuz, skaler bir değer
rank_0_tensor = tf.constant(4)


#! Float sayılardan oluşan liste verdiğimizde 1 boyutlu bir matris gibi davranır ve tipi float32 dir.
# shape: (3)
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)


#! (2,2) boyutlu ve tipi bizim seçtiğimiz bir tensor oluşturabiliriz.
rank_2_tensor = tf.constant(
                            [
                                [1, 2],
                                [3, 4],
                                [5, 6]
                            ],
                            dtype=tf.float16
                )


#! Örnek: (3,2,5) shape inde bir tensor veya matris
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])


#! Tesorleri tekrardan numpy objesine (array ine) dönüştürebiliriz.
rank_3_tensor.numpy()


#! Hatırlatma: Tensorler ile temel matematiksel işlemler yapılabilir.
print("\n=> Tensorler ile temel matematiksel işlemler yapılabilir.")
a = tf.constant([[1, 2],
                 [3, 4]])

b = tf.constant([[1, 1],
                 [1, 1]]) # tf.ones([2,2]) ile de oluşturulabilir.

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")


#! Basic Operations
print("\n=> Basic Operations")
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# Max değer
print(f"Max: {tf.reduce_max(c)}")

# Max değer indis
print(f"Max değerin indisi: {tf.math.argmax(c)}")

# softmax
print(f"Softmax {tf.nn.softmax(c)}")


#! Herhangi bir tensorflow fonksiyonu Tensor objesi gerektirir. Array'leri tensor'a çevirmek için:
converted = tf.convert_to_tensor([1,2,3])


#! Boyut kavramları
print("\n=> Boyut kavramları")
# shape: Bir tensor'ün boyutudur yani her bir eksende kaç eleman barındırdığıdır: [3,1920,1080]
# size: Bir tensor'ün her eksenindeki toplam eleman sayısıdır.
# rank: Bir tensor'ün kaç boyutlu olduğudur.
# dimension & axis : Belirli eksen sayısı.

# 0'lardan oluşan (3, 2, 4, 5) shape, inde rankı (boyut sayısı) 4 olan bir matris oluşturduk.
rank_4_tensor = tf.zeros([3, 2, 4, 5])
print(f"rank_4_tensor shape : {rank_4_tensor.shape}")
print(f"rank_4_tensor rank : {tf.rank(rank_4_tensor)}")           #tf.rank ile bir tensor,ün kaç boyutlu olduğu bulunabilir.
print(f"rank_4_tensor dimension : {rank_4_tensor.ndim}")
print(f"rank_4_tensor '-1'. eleman : {rank_4_tensor.shape[-1]}")
# NOT: Eğer fonksiyonlarda "axis=-1" şeklinde bir ifade görürseniz, bu ifade en son sıradaki shape'i referans almak içindir.
# Python daki indis kullanımına benzer.
# Yani yukarıdaki örnekte axis=-1 demek "5" anlamına gelir.


#! Tensorlarda çoğunlukla kullanılan boyutlandırma önceliği nasıldır?
# [batch, height, width, features] şeklindedir.
#batch: Birden fazla resmi temsil eder ya da bir sisteme giren ayrı eleman sayısıdır.
#height ve width: Normal bir matrisin satır ve sütunudur.
#features: Bir RGB resmin 3 kanalı gibi düşünülebilir. Bu sayı sinir ağının ara katmanlarında artar. Kanal sayısı
#[batch, features, height, width] şekilde benimsenip, kanal sayısı önce geldiğinde buna "channel first" denir.
#Fakat orijinal kullanımda bellek alanları sıralı şekilde olacaktır.(This way feature vectors are contiguous regions of memory.)


#! Tensor'ü yeniden boyutlandırmak
print("\n=> Tensor'ü yeniden boyutlandırmak")
x = tf.constant([[1], [2], [3]])
# shape attribute'u "list" olarak ele alınabilir.
print(f"shape as list: {x.shape.as_list()}",)

# yeniden boyutlandırma
reshaped = tf.reshape(x, [1, 3])
print(f"reshaped: {reshaped}")


#! Transpoze
print("\n=> Transpoze")
x = tf.constant([[[1, 2, 3, 4], [5, 6, 7, 8]],[[1, 2, 3, 4], [5, 6, 7, 8]],[[1, 2, 3, 4], [5, 6, 7, 8]]])
tx = tf.transpose(x, [0,1,2])
print("before transpose shape: ", x.shape)
print("aft transpose shape: ", tx.shape)
print(x, tx)


#! Tensor çoğaltma
print("=> Broadcasting: ", tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))


#! tf.convert_to_tensor
print("\n=> tf.convert_to_tensor")
x = tf.convert_to_tensor(x)
print(x)


#! Ragged Tensors : Düzensiz Tensorler
print("\n=> Ragged Tensors : Düzensiz Tensorler")
ragged_list = [[0, 1, 2, 3], [4, 5], [6, 7, 8],[9]]

# Normal tensor'e çevirme yöntemleri ile düzensiz tensorler çevrilemez
try:
    tensor = tf.constant(ragged_list)
except Exception as e:
    print(f"{type(e).__name__}: {e}")


# Düzensiz Tensorler bu şekilde dönüştürülür.
ragged_tensor = tf.ragged.constant(ragged_list)
print("ragged_tensor: ", ragged_tensor)



#! String tensor
print("\n=> String tensor")
# Tek elemanlı bir stringi ayırmak için
scalar_string_tensor = tf.constant("Gray wolf")
print("String Normal: ",tf.strings.split(scalar_string_tensor, sep=" "))

# Çok boyutlu bir arrayde, ayırrma sonucu her bir eleman için farklı boyutlar çıkacağı için RaggedTensor tipine dönüşecektir.
tensor_of_strings = tf.constant(["Gray wolf", "Quick brown fox", "Lazy dog"])
splitted = tf.strings.split(tensor_of_strings)
print("String Ragged: ", splitted, splitted.shape)


#! Stringden, Sayıya dönüştürme
print("\n=> Stringden, Sayıya dönüştürme")
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))


#! StringText to Number
print("\n=> StringText to Number")
text = tf.constant("ABCD")

byte_strings = tf.strings.bytes_split(text)
print("Byte strings:", byte_strings)

byte_ints = tf.io.decode_raw(text, tf.uint8)
print("Bytes:", byte_ints)


#! Unicode karakter çevirme
print("\n=> unicode_bytes")
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")
print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)



#!Sparse tensors: Seyrek Tensor
print("\n=> Sparse tensors: Seyrek Tensor")
# Seyrek tensörler, değerleri bellek açısından verimli bir şekilde dizine göre depolar
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
print(sparse_tensor, "\n")

# sparse tensorleri, dense'e (normal tensor tensor'e) çevirebiliriz.
print(tf.sparse.to_dense(sparse_tensor))
























