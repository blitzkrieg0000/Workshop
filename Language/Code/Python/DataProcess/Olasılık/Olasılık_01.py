from scipy.stats import bernoulli
import numpy as np
import numpy as np


# Olayların olabilirliğinin sayısal ifadesine olasılık denir.
"""
    Dağılım: 
       Gerçekleşen olayların sayısal karşılığıdır.

    Olasılık Dağılımı:
      Olayların sayısal karşılıklarının gerçekleşme olasılıkları ile birlite sunulmasıdır.

"""

#! Kesikli Olasılık Dağılımı (1-0, Kategorik, Yazı tura...)
# -> Bernoulli
# -> Binom
# -> Poisson

#! Sürekli Olasılık Dağılımı (Aralıklar, Maaş, Kazanç...)
# -> Normal Dağılım
# -> Uniform Dağılım
# -> Üstel Dağılım


# Amacımız olasılıklar ile belirsizlikleri ortadan kaldırmaya çalışmaktır.


#! 1-BERNOULLI DAĞILIMI
# Başarılı-Başarısız, Olumlu-Olumsuz şeklinde iki sonuçla olaylar ile ilgilenildiğinde kullanılan kesikli olasılık dağılımıdır.
p = 0.6              # Tura ihtimalinin 0.6 olması durumunu ele alalım.
rv = bernoulli(p)

sonuc = rv.pmf(k=1)  # probability mass function (olasılık kütle fonksiyonu)
print("\n-> Tura ihtimaline 1 dersek :\n",
    rv.pmf(k=1)
)
print("\n-> Yazı ihtimaline 0 dersek :\n",
    rv.pmf(k=0)
)



#! Büyük Sayılar Yasası
# Bir rassal değişkenin (deneyler sonucu çıkmış sonuçların) 
#-uzun vadede kararlılığını- tanımlayan olasılık teoremidir.

# Yazı-Tura olasılığı niçin %50 kabul edilir? (1 durum gelebilir / toplam durum 2)
# Çünkü uzun vadede bu yasaya göre ortalma olarak 0 ve 1 gelme durumu %50 ye yaklaşır.


rng = np.random.RandomState(123) #Random ifade üretirken her çalıştırmada üretilen değerlerin aynı olduğunu garantiler.
for i in np.arange(1, 21):
    deney_sayisi = 2**i
    yazi_turalar = rng.randint(0, 2, size=deney_sayisi)
    yazi_olasiliklari = np.mean(yazi_turalar)
    print(f"Atış Sayısı: {deney_sayisi} --- Yazı Olasılığı: {yazi_olasiliklari:.2f}")






