# Veri Görselleştirme 7
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

tips = sb.load_dataset("flights")
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

print("\n-> passengers :\n",
    df["passengers"].describe()
)



#! HEATMAP (ISI HARİTASI)
#Zaman serilerini gözlemlerken önemlidir.
df = df.pivot("month", "year", "passengers")
print("\n-> Pivot :\n",
    df
)

#Pivot ile çalışır.
plt.figure()
sb.heatmap(df);
plt.show(block=False)

plt.figure()
sb.heatmap(df, annot=True, fmt="d");
plt.show(block=False)

plt.figure()
sb.heatmap(df, annot=True, fmt="d", linewidths=0.5); # cbar=False
plt.show()
















