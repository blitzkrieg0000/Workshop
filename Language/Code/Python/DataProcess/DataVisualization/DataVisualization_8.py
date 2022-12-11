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

print("\n-> Signal :\n",
    df.groupby("timepoint")["signal"].count()
)

print("\n-> Signal :\n",
    df.groupby("timepoint")["signal"].describe()
)


#! Çizgi Grafik
"""
        Sinyaller ve IoT cihaz verileri için kullanılır.
    Genellikle zaman serisine bağlı makine verileri ile çalışılır.
"""

plt.figure()
sb.lineplot(x="timepoint", y="signal", data=df)
plt.show(block=False)

plt.figure()
sb.lineplot(x="timepoint", y="signal", hue="event", data=df)
plt.show(block=False)

plt.figure()
sb.lineplot(x="timepoint", y="signal", hue="event", style="event", data=df)
plt.show(block=False)

plt.figure()
sb.lineplot(
    x="timepoint", 
    y="signal", 
    hue="event",
    style="event", 
    markers=True,
    dashes=False,
    data=df
)
plt.show(block=False)

plt.figure()
sb.lineplot(
    x="timepoint", 
    y="signal", 
    hue="region",
    style="event", 
    data=df
)
plt.show()










