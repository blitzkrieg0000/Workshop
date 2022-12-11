# Veri Görselleştirme
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

planet = sb.load_dataset("planets")
df:pd.DataFrame = planet.copy()


#! Veri seti hakkında info alma
print("\n=>Veri seti hakkında info alma :\n")
print(df.info())

# Obje tipini kategoriye dönüştürme
df["method"] = pd.Categorical(df["method"])
print("\n-> Obje tipini kategoriye dönüştürme :\n",
    df.dtypes.head()
)

# İstatistikler
print("\n-> İstatistikler :\n",
    df.describe().T
)


#! Eksik Değerler Üzerinde İşlem
print("\n=> Eksik Değerler Üzerinde İşlem : \n")

print("\n-> Hiç eksik var mı? :\n",
    df.isnull().values.any()
)

# Hangi değişkenlerde eksik değer var?
print("\n-> Hangi değişkenlerde eksiklik var :\n",
    df.isnull().sum()
)

# Eksik değerlere 0 ver.
# print(df["orbital_priod"].fillna(0, inplace=True))

# Eksik değerleri ortalama ile doldur.
# print(df["mass"].fillna(df.mass.mean(), inplace=True))


#! Kategorik değişkenleri anlamak
print("\n|#> Kategorik değişkenleri anlamak :\n")

kat_df = df.select_dtypes(include=["category"])
print("\n -> Kategorik değişken seçme: \n",
    kat_df.head()
)

print("\n-> Kategorik değişkenleri Unique olarak ele almak: \n",
    kat_df['method'].unique()
)

print("\n-> Frenkanslara erişmek :\n",
    kat_df["method"].value_counts()
)

print("\n-> Kategorik değişken sayısı:\n",
    kat_df["method"].value_counts().count()
)

# df["method"].value_counts().plot.barh()
# plt.show()


#! select_dtypes ile belirli tipteki değişkenleri seçme
print("\n|#> select_dtypes ile belirli tipteki değişkenleri seçme  :\n")
df_num = df.select_dtypes(include=["float64", "int64"])
print("\n-> NoCategorical: select_dtypes :\n",
    df_num["distance"].describe().T
)


#! Genel olarak bir veri nasıl incelenir?
print("\n|#> Genel olarak bir veri nasıl incelenir? :\n")

diamonds = sb.load_dataset("diamonds")
df = diamonds.copy()
print("\n Diamond Dataset :\n",
    df.head()
)

print("\nDiamond Dataset Info :\n",
    df.info()
)

print("\nDiamond Dataset Describe :\n",
    df.describe().T
)

print("\nDiamond Dataset 'cut' kategorik sınıfı Frekansı :\n",
    df["cut"].value_counts()
)

print("\nDiamond Dataset 'color' kategorik sınıfı Frekansı :\n",
    df["color"].value_counts()
)


#! Ordinal kategorik sınıf tanımlama, (kategoriler arası sınıf farkı varsa)
df['cut'] = df['cut'].astype(pd.api.types.CategoricalDtype(ordered=True))
print("\nOrdinal tanımlama :\n",
    df['cut']
)

print("\n-> df['cut'].dtype :\n",
    df['cut'].dtype
)

print("\n-> Ordinal olup olmadığı :\n",
    df['cut'].head(1)
)

# Ordinal tanımlamayı istediğimiz sıraya göre yapma
cut_kategoriler = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
df['cut'] = df['cut'].astype(pd.api.types.CategoricalDtype(categories=cut_kategoriler, ordered=True))
print("\n-> Ordinal sıralamanın istediğimiz gibi olup olmadığı :\n",
    df['cut'].head(1)
)


#! Veri Görselleştirme
# Pandas ile
(df["cut"]
    .value_counts().plot.bar()
    .set_title("Cut Frekansları"))
plt.show()


# Seaborn ile
sb.barplot(x="cut", y=df["cut"].index, data=df)
plt.show()
sb.barplot(x=df["cut"], y=df["cut"].index)
plt.show()


#! Çaprazlamalar
# "price" ve "cut" sütunlarını çaprazlayarak arasındaki ilişkiyi açıklama
sb.catplot(x=df["cut"], y=df["price"])
plt.show()

# "cut" değerlerine karşılık gelen "color" değerleri ile beraber,
#"cut" değerlerinin "price" a göre sınıflandırılması
sb.barplot(x=df["cut"], y=df["price"], hue=df["color"])
plt.show()

# barplot grafiğinde "price" değeri değişmiş gibi görünsede
#mean ile işlem yapılacaktır ve ortalama değerler görünecektir.
print("\n-> groupby([cut, color])[price].mean() :\n",
    df.groupby(["cut", "color"])["price"].mean().unstack()
)


#! Sayısal değişkenler için histogramın önemi
# bins: Histogram için aralık
# kde: Olasılık yoğunluk fonksiyonunu çiz
sb.displot(df['price'], bins=100, kde=False)
plt.show()

#Sadece olasılık yoğunluğunu göstermek için
sb.distplot(df['price'], bins=100, hist=False)
plt.show()

#Sadece olasılık yoğunluğunun altını göstermek için
sb.kdeplot(df['price'], fill=True)
plt.show()


#! Çaprazlamalar ile veri derinliği
(sb.FacetGrid(df, hue="cut", height=7, xlim=(0, 10000))
    .map(sb.kdeplot, "price", shade=True)
    .add_legend()
)
plt.show()

# Çaprazlamalar sonucu "color" verisinin ne kadar iyi ayırt edici olabileceğini anlatan bir yapı
sb.catplot(x="cut", y="price", hue="color", kind="point", data=df)
plt.show()


















