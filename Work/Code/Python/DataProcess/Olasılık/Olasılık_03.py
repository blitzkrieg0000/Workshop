import numpy as np
from scipy.stats import poisson


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


#? Poisson Örnek Soru: Bir üniversitede 5000 not girişinde 5 tane notun yanlış girilmesi olasılığı nedir?
# Not: Dağılımın Poisson olduğu biliniyor. Lambda = 0.2
# f(5; 0.2) = ((0.2^5) * (e^-0.2)) / 5! = 0.00000218328201

#! 3-POISSON DAĞILIMI
# Belirli bir zaman aralığında belirli bir alanda, nadiren
#rastlanan olayların, olasılıklarını hesaplamak için kullanılır.

# Eğer bir olayın gerçekleşme olasılığı nadir ise:
# n>50 ve n*p<5 olması gerekir.(n: rassal deneme sayısı, p: olasılık değeri)
 

#?Problem: 
#? Hatalı ilan girişi olasılıkları hesaplanmak isteniyor.
#? Bir yıl öncesinden ölçümler yapılıyor.
#? Dağılım biliniyor "Poisson" ve Lambda = 0.1 (ortalama hata sayısı)
#? Hiç hata olmaması (0), 3 hata, 5 hata olması durumlarının olasılıkları nedir?

_lambda = 0.1
rv = poisson(mu=_lambda)

print("\n-> Hiç hata olmama ihtimali :\n",
    rv.pmf(k = 0)
)
print("\n-> 3 hata olmama ihtimali :\n",
    rv.pmf(k = 3)
)
print("\n-> 5 hata olmama ihtimali :\n",
    rv.pmf(k = 5)
)















































































