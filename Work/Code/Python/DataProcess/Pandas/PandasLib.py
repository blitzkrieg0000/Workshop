#%%
import pandas as pd
import numpy as np
import seaborn as sb
# NOT: Pandas veri tipi numpy den farklı olarak indexleri de tutar.

arr = [[1, 2, 3, 4, 5],[1,2,3,4,5]]
df = pd.Series(arr)

print("Veri : ", df.axes)
print("Eleman sayısı : ", df.size)
print("Boyut : ", df.shape)
print("Kaç boyutlu : ", df.ndim)

#%%
#! İçerideki arrayi çekmek için
df.values


#%%
#! Baştan n-eleman alma
df.head(5)


#%%
#! Sondan n-eleman alma
df.tail(3)


#%%
#! Index isimlendirme
series = pd.Series([15, 38, 32, 44, 56], index=[1, 3, 5, 7, 9])
series.head()

series = pd.Series([15, 38, 32, 44, 56], index=["a", "b", "c", "d", "e"])
series["a":"c"]


#%%
#! Dictionary ile series oluşturma
series = pd.Series({"a":11, "b":22, "c":33})
series
# series.keys()


#%%
#! DataFrame üretme
print("\n=> DataFrame üretme")
liste = np.random.normal(0.0, 1.0, (3,3))

dFrame = pd.DataFrame(liste, columns=["col_1", "col_2", "col_3"])
dFrame


#%%
#! DataFrame isimlendirme
dFrame.columns
dFrame.columns = ["col_a", "col_b", "col_c"]
dFrame.columns


#%%
#! Bilgi
dFrame.shape, dFrame.axes


#%%
#! Locate: stop dahil
# print(dFrame.loc[0:3])


#%%
#! iLocate stop dahil değil
# print(dFrame.iloc[0:3])


#%%
#! Şartlı sorgu
dFrame[dFrame.col_a > 0]
dFrame.loc[dFrame["col_a"] > 0, ["col_a"]]
dFrame[dFrame["col_a"] > 0][["col_a", "col_b"]] # İki parantez


#%%
#! Concatenate
print("\n=> Concatenate : ")
liste1 = pd.DataFrame(np.random.normal(0.0, 1.0, (3,3)))
liste1.columns = ["a", "b", "c"]
liste2 = pd.DataFrame(np.random.normal(0.0, 1.0, (3,3)))
liste2.columns = ["a", "b", "c"]
yeniListe = pd.concat([liste1, liste2], axis=0, ignore_index=True) # ignore index: indexlerin sırasını yeniden düzenler ve 0 dan başlatır.
yeniListe


#%%
#! Sütunları farklı olarak Concatenate
print("\n=> Sütunları farklı olarak Concatenate : ")
liste1 = pd.DataFrame(np.random.normal(0.0, 1.0, (3,3)))
liste1.columns = ["a", "b", "c"]
liste2 = pd.DataFrame(np.random.normal(0.0, 1.0, (3,3)))
liste2.columns = ["a", "b", "e"]    # "e", "c" farklı
yeniListe = pd.concat([liste1, liste2], axis=0, ignore_index=True)
yeniListe


#%%
#! Eşleşen sütunları birleştir.
print("\n=> Eşleşen sütunları birleştir. : ")
liste1 = pd.DataFrame(np.random.normal(0.0, 1.0, (3,3)))
liste1.columns = ["a", "b", "c"]
liste2 = pd.DataFrame(np.random.normal(0.0, 1.0, (3,3)))
liste2.columns = ["a", "d", "e"]
yeniListe = pd.concat([liste1, liste2], axis=0, ignore_index=True, join="inner")
yeniListe


#%%
#! Merge
print("\n=> Merge : ")
liste1 = pd.DataFrame(
    {
        "ilk": [1, 2, 3],
        "ikinci": ["a", "b", "c"]
    }
)

liste2 = pd.DataFrame(
    {
        "ilk": [1, 2, 3],
        "durum": ["aa", "bb", "cc"] # Farklı bir sütun var.
    }
)
yeniListe = pd.merge(liste1, liste2, on="ilk")
yeniListe


#%%
#! Many2One
print("\n=> Many2One : ")
liste1 = pd.DataFrame(
    {
        "ilk": [1, 2, 2, 2, 3],
        "ikinci": ["a", "b", "c", "d", "e"]
    }
)

liste2 = pd.DataFrame(
    {
        "ilk": [1, 2, 3],
        "durum": ["aa", "bb", "cc"] # "durum" adında farklı bir sütun var ve boyutu daha az.
    }
)
liste3 = pd.merge(liste1, liste2)   #Join gibi "ilk" sütunundakiler eşleşti
liste3


#%%
#! PANDAS İŞLEMLER \w seaborn
df:pd.DataFrame = sb.load_dataset("planets")
print("planets dataset: \n", df.head())
print("planets dataset SHAPE: ", df.shape)

# İstatistikler
print("MEAN: ", df["mass"].mean())
print("STD: ", df["mass"].std())
print("Variance: ", df["mass"].var())

# Her şeyi istatistikle
print(df.describe().T) # T : transpose

# Eksik veri barındıran gözlemleri sil
print(df.dropna().describe().T) # T : transpose

# Gruplama
print(df.groupby("method")["orbital_period"].mean())


#%%
#! Gruplama
df = pd.DataFrame(
    {
        "gruplar": ["A","B","C","B","C","B","A","A","B","C"],
        "veriler": [10, 20, 30 ,40, 15, 18 ,28 , 36, 42, 58]
    }
)

df_group = df.groupby("gruplar")
df_group.mean()


#%%
#! Aggregate
df = pd.DataFrame(
    {
        "gruplar" : ["A","B","C", "A", "B", "C"],
        "degisken1" : [10, 23, 33, 22, 11, 99],
        "degisken2" : [100, 253, 333, 262, 111, 969]
    },
    columns=["gruplar", "degisken1", "degisken2"]
)

# Aggregate: farklı fonksiyonlar ile çalıştırmak
# df.groupby("gruplar").mean()
df.groupby("gruplar").aggregate(["min", np.median, max])
# df.groupby("gruplar").aggregate({"degisken1": "max", "degisken2": "min"})


#%%
#! Filter
# Verilen şarta uyan değerleri döndürür.
def my_filter(x):
    return x["degisken1"].std() > 3

df = pd.DataFrame(
    {
        "gruplar": ["A","B","C", "A", "B", "C"],
        "degisken1" : [10,23,33,22,11,99],
        "degisken2": [100,253,333,262,111,969]
    },
    columns=["gruplar", "degisken1", "degisken2"]
)

df.groupby("gruplar").filter(my_filter)


#%%
#! Transform: 
# Aggregated sonuç üretemez.
# Vektör bazında çalışır. 
# Giriş boyutu ile çıkış boyutu aynı olmalıdır.
# Tek bir Series ile çalışır.

df_a = df.iloc[:, 1:3]
transformed = df_a.transform(lambda x : (x-x.mean())/ x.std() )
print(transformed)

try:
    df_a.transform(lambda x : x.std())
except Exception as e:
    print("\n", e, "\n")


#%%
#! Apply
# Aggregated sonuçlar döndürebilir.
# Çoklu Series destekler.
df.apply(np.sum)


#%%
#! Pivot
df:pd.DataFrame = sb.load_dataset("titanic")
df.head()
df.groupby("sex")["survived"].mean()
df.groupby(["sex", "class"])[["survived"]].aggregate("mean").unstack()



#%%
#! Pivot-2
# df.pivot_table("survived", index="sex",columns="class")

age = pd.cut(df["age"], [10, 18, 90])
age.head()
df.pivot_table(values=["survived"], index=["sex", age], columns=["class"])



# %%
#! Dışarıdan veri okuma
import pandas as pd

# CSV
csv = pd.read_csv("DataProcess/dataset/ornekcsv.csv", sep=";")


# TXT
# İlk satırı sütun ismi gibi okur.
text = pd.read_csv("DataProcess/dataset/duz_metin.txt")
print(text)


# EXCEL
excel = pd.read_excel("DataProcess/dataset/ornekx.xlsx")
print(excel.head())




# %%
