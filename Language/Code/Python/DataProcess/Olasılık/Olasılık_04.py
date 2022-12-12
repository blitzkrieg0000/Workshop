from scipy.stats import norm

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


#! 1-Normal Dağılım
# Normal dağıldığı bilinen sürekli rassal değişkenler için olasılık hesaplaması için kullanılır.

#?  Soru: Bir yatırım/toplantı öncesinde gelecek ay ile ilgili satışların belirli
#? değerde gerçekleşmesi olasılıkları belirlenmek isteniyor.
#? Dağılımın normal olduğu biliniyor.
#? Aylık ortalama satış sayısı: 80K, standart sapması: 50K
#? 90K' dan fazla satış yapma olasılığı nedir?


# cdf: cumulative density function
# 1 den çıkıyoruz çünkü gerçekleşme olasılığı 1 ve istenilen değerin olma olasılığına kadar hesaplama yapılıyor.Hesaplanacak değerden fazla olma olasılığını ise 1 den çıkara buluyoruz.
print("\n-> 90' dan fazla satış yapma olasılığı :\n",
    1-norm.cdf(90, 80, 5) # istenilen değer, ortalama, standart sapma
)

print("\n-> 70' dan fazla satış yapma olasılığı :\n",
    1-norm.cdf(70, 80, 5) # istenilen değer, ortalama, standart sapma
)
print("\n-> 70' dan az satış yapma olasılığı :\n",
    norm.cdf(50, 80, 5) # istenilen değer, ortalama, standart sapma
)
print("\n-> 70 ile 90 arası satış yapma olasılığı :\n",
    norm.cdf(90, 80, 5)-norm.cdf(70, 80, 5)
)





































































