import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from ycimpute.imputer import EM
from sklearn.impute import KNNImputer
#!######################################################
#! Makine Öğrenmesine Dayalı Eksik Değişkenleri Doldurma
#!######################################################


df:pd.DataFrame = sb.load_dataset("titanic")
df = df.select_dtypes(include=["float64", "int64"])

print("\n-> Dataseti Görüntüle :\n",
    df.head()
)

print("\n-> Sütunlardaki eksik sayısı-1 :\n",
    df.isnull().sum()
)



#! 1-KNN ile tahmine dayalı doldurma
var_names = list(df)   # Sütun adlarını tut
n_df = np.array(df.copy()) #Numpy array e çevir

imputer = KNNImputer(n_neighbors=4, weights="uniform")
dff = imputer.fit_transform(n_df)

dff = pd.DataFrame(dff, columns=var_names)  # DataFrame'e çevir

print("\n-> İşlemden sonra sütunlardaki eksik sayısı-2 :\n",
    dff.isnull().sum()
)


# #! 2-RandomForest ile tahmine dayalı doldurma (DEPRECATED OLMUŞ)
# var_names = list(df)   # Sütun adlarını tut
# n_df = np.array(df.copy())

# clf = iterforest.MissForest(n_estimators=300)
# dff = clf.complete(n_df)

# dff = pd.DataFrame(dff, columns=var_names)  # Tekrar DataFrame'e çevir

# print("\n-> İşlemden sonra sütunlardaki eksik sayısı-2 :\n",
#     dff.isnull().sum()
# )


#! 3-EM algoritması ile doldurma
var_names = list(df)   # Sütun adlarını tut
n_df = np.array(df.copy()) #Numpy array e çevir

imputer = EM()
dff = imputer.complete(n_df)
dff = pd.DataFrame(dff, columns=var_names)  # DataFrame'e çevir
print("\n-> İşlemden sonra sütunlardaki eksik sayısı-3 :\n",
    dff.isnull().sum()
)











