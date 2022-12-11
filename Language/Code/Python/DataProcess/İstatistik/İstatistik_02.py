import seaborn as sb
import pandas as pd
import researchpy as rp

# Betimsel İstatistik


# Varyans: 
#   Ortalama etrafındaki dağılımın ölçüsüdür.
#   *Standart sapmanın karesidir.


#! Kovaryans
"""
    İki değişken arasındaki ilişkinin değişkenlik ölçüsüdür.
"""

#! Korelasyon
"""
        iki değişken arasındaki ilișkinin anlamlı olup olmadığını,
    ilişkinin şiddetini ve yönünü ifade eden istatistiksel bir tekniktir.
"""

#? Sonuç: Korelasyon, iki veri arasında ilişki olup olmadığını, varsa şidetini, yönünü ölçerken; 
#?kovaryans bu ilişkinin değişkenliğinin ne kadar olduğunu ölçer


tips = sb.load_dataset("tips")
df:pd.DataFrame = tips.copy()

print("\n-> Tips Dataset :\n",
    df.head()
)

print("\n-> Veri setine genel bakış :\n",
    df.describe().T
)

#researchpy kütüphanesi ile özet
print("\n-> Özet(Numeric) :\n",
    rp.summary_cont(df[["total_bill", "tip", "size"]])
)

print("\n-> Özet(Categorical) :\n",
    rp.summary_cat(df[["sex", "smoker", "day"]])
)

print("\n-> Kovaryans 'tip' vs 'total_bill' :\n",
    df[["tip", "total_bill"]].cov()
)

print("\n-> Korelasyon 'tip' vs 'total_bill' :\n",
    df[["tip", "total_bill"]].corr()
)































