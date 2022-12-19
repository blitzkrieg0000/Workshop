# Veri Görselleştirme 3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

tips = sb.load_dataset("tips")
df:pd.DataFrame = tips.copy()

#! Özet
print("\n-> Özet-1 :\n",
    df.head()
)
print("\n-> Özet-2 :\n",
    df.describe().T
)


#! Violin grafiği: 
"""
        Yoğunluk grafiğinin ve box grafiğinin kesişimi gibi düşünülebilir.
    Dağılım hakkında bilgi sunar.
    (catplot yeni bir figure üretiyor. plt.figure() yapmaya gerek yok)
"""
sb.catplot(y="total_bill", kind="violin", data=df)
plt.show(block=False)

sb.catplot(x="day", y="total_bill", kind="violin", data=df)
plt.show(block=False)

# Kırılım == Çaprazlama == Boyut ekleme
sb.catplot(x="day", y="total_bill", kind="violin", hue="sex", data=df)
plt.show()








































































