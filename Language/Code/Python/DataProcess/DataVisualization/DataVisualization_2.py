# Veri Görselleştirme 2
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


#! 1-Kategorik değişkenin frekanslarını saydıralım.
print("\n-> 1-Kategorik değişkenin frekanslarını saydıralım (Cinsiyet) :\n",
    df["sex"].value_counts()
)

print("\n-> 1-Kategorik değişkenin frekanslarını saydıralım (Sigara içme durumu):\n",
    df["smoker"].value_counts()
)


#! 2-Box Plot: Aykırı verileri gözlemlememizde çok yardımcı bir grafiktir.
plt.figure()
sb.boxplot(x=df["total_bill"]) # Dikey grafik için: orient="v" 
plt.show(block=False)

#? Hangi günler daha fazla kazanılıyor ?
plt.figure()
sb.boxplot(x="day", y="total_bill", data=df)
plt.show(block=False)

#? Sabah mı akşam mı daha fazla kazanılıyor ?
plt.figure()
sb.boxplot(x="time", y="total_bill", data=df)
plt.show(block=False)

#? Kişi sayısı kaç ?
plt.figure()
sb.boxplot(x="size", y="total_bill", data=df)
plt.show(block=False)

#? Hangi günlerde cinsiyete göre ödeme sayısı daha fazla ?
plt.figure()
sb.boxplot(x="day", y="total_bill", hue="sex", data=df)
plt.show()


























