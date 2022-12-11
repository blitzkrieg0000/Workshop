# Veri Görselleştirme 10
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import pandas_datareader as pdr


tips = pdr.get_data_yahoo("AAPL", start="2016-01-01", end="2019-08-25")
df:pd.DataFrame = tips.copy()

#! Özet
print("\n-> Özet-1 :\n",
    df.head()
)
print("\n-> Özet-2 :\n",
    df.describe().T
)
print("\n-> Özet-3 :\n",
    df.dtypes
)
print("\n-> Özet-4 :\n",
    df.shape
)

closing = df['Close']


#! BASİT ZAMAN SERİLERİ GÖRSELLEŞTİRME
plt.figure()
closing.plot()
plt.show()
"""NOT :
        Kategorik değerlerin nominal veya ordinal olup olmadığını
    belirttiğimiz gibi zaman değerlerini de zaman-tarih değişkeni şeklinde 
    belirtmemiz gerekir ki doğru veriler ile çalışabilelim.
"""

closing.index = pd.DatetimeIndex(closing.index) #DateTimeIndex olması gerekir, değilse:

print("\n->  :\n",
    closing.index 
)















