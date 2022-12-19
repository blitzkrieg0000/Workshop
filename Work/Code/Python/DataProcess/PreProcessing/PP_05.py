import missingno as msno
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

V1 = np.array([1,3,6,np.NaN, 7,1,np.NaN, 9, 15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
V4 = np.array([np.NaN]*9)
df = pd.DataFrame({
    "V1": V1,
    "V2": V2,
    "V3": V3,
    "V4": V4
})


print("\n-> Orijinal Veri Seti :\n",
    df
)


#!##########################################################
#! Eksik Öğeleri Görselleştirme ve Diğer Mücadele Yöntemleri
#!##########################################################
# Grafikler için plt.show yorum satırlarını açınız.

#! 1-Değişkenlerdeki eksiklik sayısını grafikte gösterme
msno.bar(df)
# plt.show(block=False)


#! 2-Yapısal bozukluklar matrisini grafikte gösterme
msno.matrix(df)
# plt.show(block=False)


#! 3-Heatmap, eksikliği değerlendirmek için bir yöntem olarak kullanılabilir.
# nullity korelasyonu varsa(boşluk korelasyonu) bunu bu grafik gösteriyor.
# "nullity correlation" değeri 1'e yaklaşırsa iki değişkenin eksikliği, yüksek korelasyon
#ile beraber, birlikte meydana geliyor demektir çünkü korelasyon iki değişken arasındaki ilişkiyi gösterir.
# Yani bir değişkende eksiklik varsa diğer değişkende de yüksek ihtimalle eksiklik olacaktır demektir.
msno.heatmap(df)
# plt.show(block=False)



#! Eksik Veri Silme-2
print("\n-> N/A verilerin silinmesi :\n",
    df.dropna()
)
print("\n-> Tüm değerleri eksik olan satırları sil :\n",
    df.dropna(how="all")
)
print("\n-> En az 1 tane özelliği(sütunu) boş olan gözlemi(satırı) sil :\n",
    df.dropna(axis=0) # axis=0: satırları seç -> sütun boyunca demek
)
print("\n-> Tüm değerleri N/A olan bir değişkeni(sütunu) sil :\n",
    df.dropna(axis=1, how="all")
)



#! Sayısal Değişkenlerde Değer Atama-2
print("\n-> N/A değerlerini 0 ile doldur :\n",
    df["V1"].fillna(0)
)

print("\n-> Tüm tablo sütunları için N/A değerleri ortalama değerler ile doldur :\n",
    df.apply(lambda x: x.fillna(x.mean()), axis=0)
)

print("\n-> Tüm N/A değerleri ortalama ile doldur :\n",
    df.fillna(df.mean())
)

print("\n-> V1-V2 arası N/A değerleri ortalama ile doldur :\n",
    df.fillna(df.mean()["V1":"V2"])
)

print("\n-> v3 N/A değerleri medyan ile doldur :\n",
    df.fillna(df.median())
)

print("\n-> Where Fonksiyonu ile :\n",
    df.where(pd.notna(df), df.mean(), axis="columns")
)


#!###########################################
#! Kategorik Değişken Kırılımında Değer Atama
#!###########################################
# Elimizdeki verilerdeki özellikleri, diğer kategorik değişkenlerce indirgeyip indirgeyemediğimize bakmamız gerekir.
#Yani N/A değişkenlerin ortalama ve medyan ile doldurulmasını, kategorik cinsten daha doğru biçimde yapmak gerekir.
# Örneğin ayrı departmanlardan oluşan bir maaş tablosunda ki bir eksikliği kapatmak için, tüm departmanların ortalama maaşını
#almaktansa, ilgili departmanın aldığı maaş ortalaması ile o departmandaki eksikliği onarmak daha doğru bir yaklaşım olacaktır.

V1 = np.array([1, 3, 6, np.NaN, 7, 1, np.NaN, 9, 15])
V2 = np.array([7, np.NaN, 5, 8, 12, np.NaN, np.NaN, 2, 3])
V3 = np.array([np.NaN, 12, 5, 6, 14, 7, np.NaN, 2, 31])
V4 = np.array(["IT", "IT", "IK", "IK", "IK", "IK", "IK", "IT", "IT"])

df = pd.DataFrame({
    "maas" : V1,
    "V2" : V2,
    "V3" : V3,
    "departman" : V4
})


print("\n-> Çalışan-Maaş Dataset :\n",
    df
)

print("\n-> Departmanların maaş ortalaması :\n",
    df.groupby("departman")["maas"].mean()
)

# Transform kullanılmasının sebebi önceki array ile aynı boyutta çıktı almak içindir.
print("\n-> Eksik maaşları ilgili departman ortalamasına göre doldur :\n",
    df["maas"].fillna(df.groupby("departman")["maas"].transform("mean"))
)

# Aggregated : (sum, mean, median... toplanmış tek elemanlı sonuç) 
# "maas" sütununa göre transform işlemi uygula ancak departman ile gruplayarak bu işi yap
#"transfrom" kullandığımız için "maas" sütunu ile aynı boyutta bir çıktı ver.
#Transform işlemi içindeki fonksiyon normalde aggregated bir sonuç döndüremez. Yani tek bir elemandan sorumludur. Tüm sütunu ele alamaz. 
#Ancak groupby ile kullanılırsa istinai olarak ilgili tüm sütun üzerinde çalışır ve sonucu ele aldığı sütunun boyutu kadar göstererek bir değer döndürür.
print("\n-> Aggregated return'ü olan bir fonksiyon ile Transform kullanılabildi çünkü groupby Transform için Series sağlıyor:\n",
    df.groupby("departman")["maas"].transform(lambda x : x.mean())
)
































