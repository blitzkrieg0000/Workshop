import pandas as pd 
import numpy as np
from sklearn import preprocessing

#!##################################################
#! Değişken Standardizasyonu (Veri Standardizasyonu)
#!##################################################

# Değikenin yayılım ve dağılım bilgisi değişsede
#değişkenin veriseti içindeki konumu ve özü değişmeyecektir.

V1 = np.array([1,3,6,5,7])
V2 = np.array([7,7,5,8,12])
V3 = np.array([6,12,5,6,14])
df = pd.DataFrame({
    "V1" : V1,
    "V2" : V2,
    "V3" : V3
})
df = df.astype(np.float64)

print("\n-> Dataset :\n",
    df.head()
)


#! 1-Standardizasyon
# Standart Normal dağılıma çevirmeyi temsil eder.
print("\n-> Standardizasyon :\n",
    preprocessing.scale(df)
)


#! 2-Normalizasyon
# Değerleri 0 ile 1 arasına hapseder.
print("\n-> Normalizasyon :\n",
    preprocessing.normalize(df)
)


#! 3-Min-Max Dönüşümü
# İstenilen iki değer aralığına verileri sıkıştırmaya yarar.
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
print("\n-> Min-Max :\n",
    scaler.fit_transform(df)
)




















































