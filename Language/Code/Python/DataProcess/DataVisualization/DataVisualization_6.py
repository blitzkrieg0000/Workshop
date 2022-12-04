# Veri Görselleştirme 6
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

tips = sb.load_dataset("iris")
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


#! Scatter Plot MATRİSİ

sb.pairplot(df)
plt.show(block=False)

sb.pairplot(df, hue="species")
plt.show(block=False)

sb.pairplot(df, hue="species", markers=["o", "s", "D"])
plt.show(block=False)

#Grafiğe doğru ekleme
sb.pairplot(df, hue="species", kind="reg")
plt.show()

















