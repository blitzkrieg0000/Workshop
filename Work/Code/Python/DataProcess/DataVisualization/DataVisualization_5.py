# Veri Görselleştirme 5
# Doğrusal İlişki Gösterilmesi
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

#! Linear Plot
# (lmplot oto figure: plt.figure yazmaya gerek yok)
#İki değişken arasındaki doğrusal ilişkiyi gösterir.
sb.lmplot(x="total_bill", y="tip", data=df)
plt.show(block=False)

sb.lmplot(x="total_bill", y="tip", hue="smoker", data=df)
plt.show(block=False)

sb.lmplot(x="total_bill", y="tip", hue="smoker", col="time", data=df)
plt.show(block=False)

sb.lmplot(x="total_bill", y="tip", hue="smoker", col="time", row="sex", data=df)
plt.show()



























