
#!###################
#! Eksik Veri Analizi
#!###################
# İnceleme esnasında tablo birleştirme ve veri dönüştürme vb. durumlarda ortaya çıkan sorunlardır.
#Silme ve doldurma yaklaşımları dikkat edilmesi gereken yaklaşımlardır. Çeşitli problemler ortaya çıkarabilmektedir.


#? Eksiklik olan veriye bağlı tüm değerleri direkt silme.
# Eksik değere sahip gözlemlerin rassallığının kontrol edilmesi ve o şekilde yok edilmesi mantıklıdır.
# Yani rastgele olmayan; belirli bir şarta göre meydana gelen eksik verileri silmenin etkileri, bizim ele aldığımız konuya göre değişir.
# Eksik gözlemlerin veri setinden direk çıkarılabilmesi için veri setindeki eksiğin bazı durumlarda kısmen bazı durumlarda tamamen
#rastlantısal olarak meydana gelmesi gerekir.
# Eğer eksiklikler değişkenler ile ilişkili olarak ortaya çıkan yapısal problemler ile meydana gelmiş ise bu durumda yapılacak silme
#işlemleri ciddi ölçüde "Yanlılıklara" sebep olacaktır.
# Eğer rastgele oluşmayan veriler varsa incelememiz gerekir.

#* I-> Veri setindeki eksikliğin yapısal bir eksiklik olup olmadığı bilinmesi gerekir:
# Örnek: "Kredi kartı" olmayan birisinin "kredi kartı harcaması" bilgisi mevcut olması beklenemez.
# Bu değerlere sahip gözlemleri ele aldığımızda, tüm satırın silinmesi ve diğer bilgilerin hiçe sayılması yanlış bir yaklaşım olacaktır.

#* II-> N/A her zaman bir eksiklik anlamına gelmez:
# Örnek: "Kredi kartı" olupta "kredi kartı harcaması" N/A olan birisinin bilgisi eksik sayılmaz çünkü daha harcama yapmamış olabilir.
#Bu durumda N/A değeri 0 kabul edilebilir yani ölçülebilir bir değere dönüştürülebilir.

#* III-> Bilgi kaybı:
# Örnek: Bir gözlem için feature sayısı 100 olsun. Eğer feature'lardan birisi boş ise ve 99 değer doluysa; tek bir tane değişkenden kurtulmak için
#99 adet anlamlı olabilecek bilgiyi silmek mantıksız olacaktır.


#? Eksik Veri Türleri
# a-> Tümüyle Raslantısal Kayıp
#   - Diğer verilerle alakasız olarak tamamen raslantısal gelişen kayıplar. (Direkt silinebilecek bir veri türüdür.)
# b-> Raslantısal Kayıp
#   - Diğer veri türlerine bağlı oluşabilen kayıptır.
# c-> Raslantısal Olmayan Kayıp
#   - Göz ardı edilemeyecek kayıptır. Diğer değişkenler ve yapısal problemler ile bağlantılı olarak ortaya çıkmış olabilir.


#? Eksiklik Testleri
#-> Görsel Teknikler
#-> Bağımsız Örneklem T-Testi
#-> Korelasyon Testi
#-> Little MCAR Testi


#? Silme Yöntemleri
# Mesela 70% Eksiklik varsa silinecek...

#-> Gözlem ya da değişken silme yöntemi
#-> Liste bazında silme yöntemi
#-> Çiftler bazında silme yöntemi


#? Doldurma Yöntemleri
#-> Ortanca, ortalama, medyan ile doldurma
#-> En benzer birime atama (Hot Deck)
#-> Dış kaynaklı atama


#? Tahmine Dayalı Yöntemler ile doldurma
#-> Makine Öğrenmesi
#-> EM Algoritması
#-> Çoklu Atama YÖntemi







