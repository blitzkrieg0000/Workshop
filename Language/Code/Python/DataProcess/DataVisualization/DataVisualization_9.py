# Veri Görselleştirme 9
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

tips = sb.load_dataset("fmri")
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


#! Çizgi Grafik
# Sinyaller ve IoT cihaz verileri için kullanılır.




















