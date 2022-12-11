import numpy as np
np.random.seed(0)

"""İstatistik, veri-biliminin "bilim", makine öğrenmesinin "öğrenme" kısmıdır."""

#! Örnek Teorisi:
"""
    Örneklem: Bir ana kitle içerisinden, popülasyonu temsil eden alt kümedir.
"""

#! Merkezi Limit Teoremi
"""
    Bağımsız ve aynı dağılıma sahip rassal değişkenlerin toplamı ya da,
    aritmetik ortalaması yaklaşık olarak normal dağılmaktadır.
"""

# 1-Popülasyon oluştur
populasyon = np.random.randint(0, 80, 10000)
print("\n-> Populasyon Ortalaması :\n",
    populasyon.mean()
)

# 2-Popülasyondan rastgele örneklem al
orneklem = np.random.choice(a=populasyon, size=100)
print("\n-> Örneklem-1 Ortalaması :\n",
    orneklem.mean()
)
#? Sonuç: Örneklem ve Populasyon'ın ortalaması hemen hemen birbirine yakın çıktı!


#! Örneklem dağılımı
# Alınan örneklemler ne kadar doğru?

# 1-Bunun için birden fazla örneklem alalım
orneklemler = np.array([np.random.choice(a=populasyon, size=100) for _ in range(10)])

#? Sonuç: Merkezi limit teoremine göre tüm alınan örneklerin ortalaması ana kitleye yakın çıktı.
print("\n-> orneklemler :\n",
    orneklemler.mean()
)












































