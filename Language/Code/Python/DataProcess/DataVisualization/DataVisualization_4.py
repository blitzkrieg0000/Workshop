# Veri Görselleştirme 4
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


#! Korelasyon grafiği
# Korelasyon iki veri arasındaki ilişki demektir.
#ScatterPlot(Saçılım grafiği sayısal değişkenler arasındaki ilişkiyi gösterir.)
#Birden fazla sayısal değişkenin birlikte oluşturduğu yapılar ele alınır.

plt.figure()
sb.scatterplot(x="total_bill", y="tip", data=df)
plt.show(block=False)

#! Çaprazlamalar
plt.figure()
sb.scatterplot(x="total_bill", y="tip", hue="time", data=df)
plt.show(block=False)

plt.figure()
sb.scatterplot(x="total_bill", y="tip", hue="time", style="time", data=df)
plt.show(block=False)

plt.figure()
sb.scatterplot(x="total_bill", y="tip", hue="day", style="time", data=df)
plt.show(block=False)

# Sürekli değişken de eklersek. "size" kişi sayısı
plt.figure()
sb.scatterplot(x="total_bill", y="tip", hue="size", size="size", data=df)
plt.show()

































































