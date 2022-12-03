import matplotlib.pyplot as plt
import tensorflow as tf


#! Otomatik türev (Auto Differentiation)
# Kısa Örnek
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2
    dy_dx = tape.gradient(y, x)
    print(dy_dx.numpy())


# İç içe yazılarak yüksek dereceli türev hesabı yapılabilir.
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

#* 2-buradaki manager ise d2x/dx2 işlemi gerçekleştirmek için t1.gradient in yaptığı hesapları takip ediyor.
with tf.GradientTape() as t2:

    #* 1-içerideki bu manager dy/dx işlemini aşağıda t1.gradient fonksiyonunu çağırarak gerçekleştiriyor.
    with tf.GradientTape() as t1:
        y = x * x * x

    # t2 ContextManager'ı t1 bandını kapsadığı için içerideki t1 değişkeni ile yapılan işlemleri, t2 takip edebilir.
    
    dy_dx = t1.gradient(y, x)   # T1, "y" fonksiyonunu ve "x" değişkenini izleyerek dy/dx ifadesini hesapladı.
d2y_dx2 = t2.gradient(dy_dx, x) # T2 ise yapılan bu dy/dx işleminin de gradyanını otomatik hesapladı.
# Sonuç olarak GradientTape ContextManager'ları iç içe çağrıldığında, içerideki manager'ın yaptığı işlemleri bir üstteki manager takip etti.

print('dy_dx:', dy_dx.numpy())  # 3 * x**2 => 3.0
print('d2y_dx2:', d2y_dx2.numpy())  # 6 * x => 6.0


#! Tensor ile
w = tf.Variable(tf.random.normal((3, 2)), name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = [
        [1., 2., 3.]
    ]

with tf.GradientTape(persistent=True) as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y**2)
    [dl_dw, dl_db] = tape.gradient(loss, [w, b])
    print(dl_dw)

    # dictionary şeklinde girdi verilebilir.
    # my_vars = {
    #     'w': w,
    #     'b': b
    # }
    # grad = tape.gradient(loss, my_vars)
    # grad['b']


#! Fonksiyon ile
x = tf.Variable(1.0)

def f(x):
    y = x**2 + 2*x - 5
    return y

with tf.GradientTape() as tape:
    y = f(x)

g_x = tape.gradient(y, x)  # g(x) = dy/dx
print("g_x: ", g_x)


#! Tensorflow modeli ile
layer = tf.keras.layers.Dense(2, activation='relu')
x = tf.constant([[1., 2., 3.]])

with tf.GradientTape() as tape:
    y = layer(x)
    loss = tf.reduce_mean(y**2)

# Tüm eğitilebilir parametrelere göre gradyan hesabı
grad = tape.gradient(loss, layer.trainable_variables)

for var, g in zip(layer.trainable_variables, grad):
    print(f'{var.name}, shape: {g.shape}')


#! GradientTape in izlediği değerler, eğitilebilir olmalıdır.
# Eğitilebilir bir değer
x0 = tf.Variable(3.0, name='x0')

# Eğitilemez
x1 = tf.Variable(3.0, name='x1', trainable=False)

# Eğitilemez değer: A variable + tensor returns a tensor.
x2 = tf.Variable(2.0, name='x2') + 1.0

# Eğitilemez değer:
x3 = tf.constant(3.0, name='x3')

with tf.GradientTape() as tape:
    y = (x0**2) + (x1**2) + (x2**2)

[g1, g2, g3, g4] = tape.gradient(y, [x0, x1, x2, x3])
print([g1, g2, g3, g4])


#! Bir değişkeni izlemeye al
x = tf.constant(3.0)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = x**2

# dy = 2x * dx
dy_dx = tape.gradient(y, x)
print("Watch: ", dy_dx.numpy())


#! İzlemeyi kapat
x0 = tf.Variable(0.0)
x1 = tf.Variable(10.0)

with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(x1)
    y0 = tf.math.sin(x0)
    y1 = tf.nn.softplus(x1)
    y = y0 + y1
    ys = tf.reduce_sum(y)


# dys/dx1 = exp(x1) / (1 + exp(x1)) = sigmoid(x1)
grad = tape.gradient(ys, {'x0': x0, 'x1': x1})
print('dy/dx0:', grad['x0'])
print('dy/dx1:', grad['x1'].numpy())


#! Ara değerler için
x = tf.constant(3.0)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = x * x
    z = y * y

# Ara değer y ye göre z nin gradyanı:
# dz_dy = 2 * y and y = x ** 2 = 9
print("Ara değer: ", tape.gradient(z, y).numpy())


#! Kalıcı gradient hesabı
x = tf.constant([1, 3.0])
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = x * x
    z = y * y

print(tape.gradient(z, x).numpy())  # [4.0, 108.0] (4 * x**3 at x = [1.0, 3.0])
print(tape.gradient(y, x).numpy())  # [2.0, 6.0] (2 * x at x = [1.0, 3.0])
del tape   # "tape" değişken referansını sil


#! GRADIENT TAPE NOTLARI
# GradientTape context ini kullanırken küçük bir overhead vardır.
# GradientTape, backward esnasında giriş ve çıkış değerleri dahil, orta değerler ile (modelin ara katmanlarındaki değerler)
#hafızada tutar.
# Verimlilik için ReLU gibi operasyonları tutmaya ihtiyaç yoktur. Forwardpass de zaten bu değerler atılır. Persistent=True
#ise hiçbir değer atılmayacak ve hafıza bir miktar şişecektir.



#! Hedef scaler değilse
# Gradyan hesabı skalerler için geçerlidir.
#Eğer hedef scaler değilse, gradyenlerin toplamı hesaplanır.
x = tf.Variable(2.)

with tf.GradientTape() as tape:
    y = x * [3., 4.]

print("\n", tape.gradient(y, x).numpy())


#! Element wise gradient
#! Her bir eleman için bağımsız türev alınır.
x = tf.linspace(-10.0, 10.0, 200+1)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.nn.sigmoid(x)

dy_dx = tape.gradient(y, x)

plt.plot(x, y, label='y')
plt.plot(x, dy_dx, label='dy/dx')
plt.legend()
_ = plt.xlabel('x')





































