import numpy as np
import pandas as pd


#!###########################
#! I-HIZLI EKSİK VERİ ANALİZİ
#!###########################
V1 = np.array([1,3,6,np.NaN, 7,1,np.NaN, 9, 15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
df = pd.DataFrame({
    "V1": V1,
    "V2":V2,
    "V3":V3
})

print("\n-> Veri Seti :\n",
    df
)
print("\n-> Sütunlardaki Eksik Değerler :\n",
    df.isnull()
)
print("\n-> Sütunlardaki Eksik Değer Sayısı :\n",
    df.isnull().sum()
)
print("\n-> Veri Setindeki Toplam Eksik :\n",
    df.isnull().sum().sum()
)
print("\n-> Sütunlardaki Eksik Olmayan Değerlerin Sayısı :\n",
    df.notnull().sum()
)
print("\n-> Hangi elemanlarda en az 1 eksik var :\n",
    df[df.isnull().any(axis=1)]                  # any: Bir dizideki en az 1 eleman True ise True döner.
)
print("\n-> Hangi elemanlar tamamen dolu :\n",
    df[df.notnull().all(axis=1)]                 # all: Bir dizideki tüm elemanlar True ise True döner.
)


#! 1-Basit olarak: Eksik değerlerin direkt silinmesi: dropna
print("\n-> Eğer en az 1 tane N/A değer varsa o gözlem birimi silinir: :\n",
    df.dropna() # inplace=True argumanı girersek kalıcı olur.
)


#! 2-Basit olarak: Değer ile doldurma: fillna
df_v1_mean = df["V1"].mean()
print("\n-> B-Basit değer atama orijinal V1 :\n",
    df["V1"]
)

print("\n-> Sütunu kendi ortalaması ile doldur :\n",
    df["V1"].fillna(df_v1_mean)
)

# apply sütun bazında işlem yapıyor(axis=0).
#Her sütunun N/A değerlerini kendi ortalamasıyla dolduruyor.
print("\n-> Tüm değişkenler üzerine uygula :\n",
    df.apply(lambda x: x.fillna(x.mean()), axis=0) # x: Sütun, axis=0 yani satırlar boyunca demektir. Bu da sütunlara tekabül ediyor.
)











