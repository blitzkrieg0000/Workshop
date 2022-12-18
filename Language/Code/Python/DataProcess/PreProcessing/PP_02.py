import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import numpy as np




#!##############################################
#! I-TEK DEĞİŞKENLİLERDE AYKIRI GÖZLEM BELİRLEME
#!##############################################
# Veride genel eğilimin oldukça dışına çıkan ya da diğer gözlemlerden
#oldukça farklı olan gözlemlere aykırı gözlem denir.

# Genellenebilirlik kaygısı ile oluşturulan kural setlerini ya da fonksiyonları yanıltır.
# Yanlılığa sebep olur.


#! 1-Aykırı gözlemler nasıl ayırt edilir ve neye göre aykırı olduğu söylenir?
#? A-Sektör Bilgisi: 
# Örneğin emlak sektöründe Türkiye'de %90 lik kesimin 2+1 ev olduğu bilgisini biliyorsak.
#Yapacağımız makine öğrenmesi için hazırlayacağımız veri setine
#1000m^2 lik olan evleri katmak aykırı gözlem olacaktır.

# Örneğin ilk okul için yaş bilgilerine göre kişileri listeliyorsak,
#Çok düşük ihtimal ile 60 yaşındaki bir kişiyi hesaba katmak aykırı gözlem olacaktır.


#? B-Standart Sapma Yaklaşımı
# Bir değişkenin ortalaması üzerine aynı değişkenin standart sapması hesaplanarak eklenir.
#1,2 ya da 3 standart sapma değeri ortalama üzerine eklenerek ortaya çıkan bu değer eşik değeri olarak
#düşünülür ve bu değerden yukarıda ya da aşağıda olan değerler aykırı olarak nitelendirilir.
# [Eşik değer = Ortalama +- n x Standart Sapma]


#? C-"z-Skoru" Yaklaşımı
# Standart sapma yöntemine benzer çalışır. Değişken standart normal dağılıma uyarlanır yani standartlaştırılır.
# Dağılımın sağında ve solunda örneğin +-2.5 değerine göre eşik değer verilir ve bu değerin altında ve
#üstünde olan değerler aykırı gözlem olarak nitelendirilir.


#? D-BoxPlot (Interquartile range - IQR) Yöntemi
# En sık kullanılan yöntemlerden birisidir. Değişkenler küçükten büyüğe sıralanır. Çeyrekliklerine(yüzdeliklerine) göre
# Q1-Q3 değerlerine göre eşik değerleri hesaplanır ve bu şekilde aykırı gözlemler tanımlanır.
#IQR = 1.5*(Q3-Q1)
#Alt Eşik = Q1 - IQR
#Üst Eşik = Q3 + IQR


original_df:pd.DataFrame = sb.load_dataset("diamonds")
original_df = original_df.select_dtypes(include=['float64', 'int64']) # Tip olarak sayısal olanları seçiyoruz. Kategorikleri ele almıyoruz.

#! 2-Basit Eksik (N/A) Değer Çıkartma (Doğrusu bu şekilde direkt yapmıyoruz)
original_df = original_df.dropna()  # Eksik değerler varsa, o gözlemleri çıkart

print("\n-> Diamonds Dataset :\n",
    original_df.head()
)

df_table = original_df['table']

print("\n-> Table Değişkeni :\n",
    df_table
)

#! 3-Boxplot ile aykırı değerleri gözlemleme
sb.boxplot(x=df_table)
plt.show()


#! 4-Aykırı değerlerin eşiğini boxplot yöntemi ile belirleme
Q1 = df_table.quantile(0.25)
Q3 = df_table.quantile(0.75)
IQR = Q3-Q1
k = 1.5     # Kat sayı
alt_sinir = Q1 - k*IQR
ust_sinir = Q3 + k*IQR


# Aykırı değerleri görüntüleyelim
print("\n-> Küçük aykırı değerler :\n",
    df_table[df_table<alt_sinir]
)

print("\n-> Büyük aykırı değerler :\n",
    df_table[df_table>ust_sinir]
)

print("\n-> Tüm aykırı değerler :\n",
    df_table[(df_table<alt_sinir) | (df_table>ust_sinir)]
)

print("\n-> Aykırı indexleri görüntüle :\n",
    df_table[(df_table<alt_sinir) | (df_table>ust_sinir)].index
)

aykiri_index = df_table[(df_table<alt_sinir) | (df_table>ust_sinir)].index


#* Aykırı Değerlerden Kurtulma Yaklaşımları
#? A-Silme işlemi
df_table = pd.DataFrame(df_table) # Seaborn Series -> Pandas DataFrame
print("\n-> Shape :\n",
    df_table.shape
)

# "~"" işareti ile koşulu sağlayanlar dışındakileri getir.
cleaned_df = df_table[~((df_table<alt_sinir) | (df_table>ust_sinir))]

print("\n-> Temizlendikten sonra :\n",
    cleaned_df.shape
)


#? B-Ortalama ile doldurma
df_table = original_df['table'].copy()
df_table[aykiri_index] = df_table.mean()


#? C-Baskılama Yöntemi
# Üst sınırdaysa üst sınır; alt sınırdaysa alt sınır değerine eşitlenirler.
df_table = original_df['table'].copy()
df_table[df_table<alt_sinir] = alt_sinir
df_table[df_table>ust_sinir] = ust_sinir



#!###############################################
#! II-ÇOK DEĞİŞKENLİLERDE AYKIRI GÖZLEM BELİRLEME
#!###############################################
# Hesaplamalarda birden fazla özelliğin bir arada ele alındığı durumlarda
#Aykırı olan gözlemlerden kaçınılmaya çalışılır. Örneğin A değikeni B değişkenine göre
#Olasılığın çok küçük bir miktarında bir değer alıyorsa, bu A-B çifti aykırı bir gözlemdir.
# Mesela veri tabanında bir öğrenci 17 yaşındayken 3 üniversite okumuşsa; gerçekleşse bile çok düşük olasılıklı
#olacağı için diğer verileri saptıracaktır. Bu yüzden 17 yaş ve 3 üniversite aykırı gözlemdir.


#! 1-Local Outlier Factor (LOF) Yöntemi ile aykırı gözlemleri tespit etme
# Noktaların komşuluk yoğunluğuna göre seyrek olan değişkenler çıkarılır. 

df = original_df.copy()

# Komşuluk oranına göre yoğunluk skorlarını belirlemek için bir obje
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_    #Negatif skor değerlerini al

print("\n-> LOF Puanlar :\n",
    np.sort(df_scores)[0:20]
)

# 13. eleman da büyük bir atlama noktası olduğu için
#keyfi olarak 13. elemanı eşik değeri olarak seçtik
esik_deger = np.sort(df_scores)[13]
aykiri_olmayan_lof_index = df_scores > esik_deger
aykiri_lof_index = df_scores < esik_deger

#? A-Silme Yöntemi
yeni_df = df[aykiri_olmayan_lof_index]

print("\n-> Aykırı olmayan değerler :\n",
    yeni_df
)


#? B-Baskılama Yöntemi
baski_deger = df[df_scores == esik_deger]
aykirilar = df[aykiri_lof_index]            # df[~aykiri_olmayan_lof_index]

# Indexleri kaldır, numpy array e çevir
np_baski_deger = baski_deger.to_records(index=False)
print("\n-> Baskılanmış değerler :\n",
    np_baski_deger
)

df.iloc[aykiri_lof_index, :] = baski_deger

print("\n-> Baskılanmış DF :\n",
    df[aykiri_lof_index]
)

print("\n-> Son Baskılanmış DF özet :\n",
    df.head()
)



