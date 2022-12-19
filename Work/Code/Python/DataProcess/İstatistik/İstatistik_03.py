import numpy as np
import statsmodels.stats.api as sms
np.random.seed(0)


#! Güven Aralığı bulma

# Rastgele fiyat veri seti oluştur
fiyatlar = np.random.randint(10, 110, 1000)


print("\n-> Ortalama fiyat :\n",
    fiyatlar.mean()
)

# Tüm fiyat veri setine göre kullanıcıların ödemek istedikleri değer
#istatistiksel olarak 95% güven aralığı ile şu aralıktadır:
print("\n-> Confidence Interval :\n",
    sms.DescrStatsW(fiyatlar).tconfint_mean()
)
























