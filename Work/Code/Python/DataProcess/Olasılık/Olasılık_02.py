import numpy as np
from scipy.stats import binom


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



#! 2-BINOM DAĞILIMI
# Gerçekleşme olasılığı bilinen bir olayın "n" denemede ki
#gerçekleşme olasılığını bulmak için kullanılır.

#? Soru: Reklamı 100 kişi gördüğünde 1,5,10,25 kişinin tıklama olasılığı nedir.
# Dağılım ve tıklanma olasılığı biliniyor: p=0.01
#-> f(1; 100, 0.01) = C(100,1) 0.01^2 (1-0.01)^(100-1) = 0.37

p = 0.01           # olma olasılığı
n = 100            # deneme sayısı
rv = binom(n, p)

print("\n-> Cevap :\n",
    rv.pmf(1),     
    rv.pmf(5),      # 100 denemede, 5' inin gerçekleşme olasılığı
    rv.pmf(10),
    rv.pmf(25)
)
























































































