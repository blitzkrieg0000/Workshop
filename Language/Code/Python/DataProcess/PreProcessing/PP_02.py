
#! 1-AYKIRI GÖZLEM

# Veride genel eğilimin oldukça dışına çıkan ya da diğer gözlemlerden
#oldukça farklı olan gözlemlere aykırı gözlem denir.

# Genellenebilirlik kaygısı ile oluşturulan kural setlerini ya da fonksiyonları yanıltır.
# Yanlılığa sebep olur.


#! Aykırı gözlemler nasıl ayırt edilir ve neye göre aykırı olduğu söylenir?
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
# [Eşik değer = Ortalama + n x Standart Sapma]


#? C-"z-Skoru" Yaklaşımı
# Standart sapma yöntemine benzer çalışır. Değişken standart normal dağılıma uyarlanıri yani standartlaştırılır.
#Örneğin sonrasında dağılımın sağında ve solunda +-2.5 değerine göre eşik değer verilir ve bu değerin altında ve
#üstünde olan değerler aykırı gözlem olarak nitelendirilir.


#? D-BoxPlot (Interquartile range - IQR) Yöntemi
# En sık kullanılan yöntemlerden birisidir. Değişkenler küçükten büyüğe sıralanır. Çeyrekliklerine(yüzdeliklerine) göre
# Q1-Q3 değerlerine göre eşik değerleri hesaplanır ve bu şekilde aykırı gözlemler tanımlanır.
#IQR = 1.5*(Q3-Q1)
#Alt Eşik = Q1 - IQR
#Üst Eşik = Q3 + IQR



































