import requests, zipfile 
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.fft import fft, fftfreq

# Dosyaları indir ve çıkar
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
# r = requests.get(url, allow_redirects=True)
# z = zipfile.ZipFile(io.BytesIO(r.content))
# z.extractall(path='tmp/')

#change dir to the folder with data
os.chdir('dataset/UCI HAR Dataset/train/Inertial Signals')

# Toplam ivmeden yerçekiminin çıkarılmasıyla elde edilen vücut ivme sinyali.
df = pd.read_csv('body_acc_x_train.txt', header=None, delim_whitespace=True) # Her satır 128 elemanlı bir vektör gösterir.

# ilk gözlemi alalım
acquisition = df[0]

acquisition.plot(figsize = [15,8])
plt.title("Body Acceleration X", size = 20, color = 'b')
plt.xlabel("Samples", size=15)
plt.ylabel("Acceleration", size=15)
plt.grid()
plt.show()


FEATURES = ['MIN','MAX','MEAN','RMS','VAR','STD','POWER','PEAK','P2P','CREST FACTOR','SKEW','KURTOSIS',
            'MAX_f','SUM_f','MEAN_f','VAR_f','PEAK_f','SKEW_f','KURTOSIS_f']

def features_extraction(df): 
    Max=[]
    Min=[]
    Mean=[]
    Rms=[]
    Var=[]
    Std=[]
    Power=[]
    Peak=[]
    Skew=[]
    Kurtosis=[]
    P2p=[]
    CrestFactor=[]
    FormFactor=[]
    PulseIndicator=[]
    Max_f=[]
    Sum_f=[]
    Mean_f=[]
    Var_f=[]
    Peak_f=[]
    Skew_f=[]
    Kurtosis_f=[]
    
    
    X = df.values
    # Zamana Göre Sinyal
    Min.append(np.min(X))
    Max.append(np.max(X))
    Mean.append(np.mean(X))
    Rms.append(np.sqrt(np.mean(X**2)))
    Var.append(np.var(X))
    Std.append(np.std(X))
    Power.append(np.mean(X**2))
    Peak.append(np.max(np.abs(X)))
    P2p.append(np.ptp(X))
    CrestFactor.append(np.max(np.abs(X))/np.sqrt(np.mean(X**2)))
    Skew.append(stats.skew(X))
    Kurtosis.append(stats.kurtosis(X))
    FormFactor.append(np.sqrt(np.mean(X**2))/np.mean(X))
    PulseIndicator.append(np.max(np.abs(X))/np.mean(X))


    # Frekansa Göre Sinyal
    ft = fft(X)
    S = np.abs(ft**2)/len(df)
    Max_f.append(np.max(S))
    Sum_f.append(np.sum(S))
    Mean_f.append(np.mean(S))
    Var_f.append(np.var(S))
    
    Peak_f.append(np.max(np.abs(S)))
    Skew_f.append(stats.skew(X))
    Kurtosis_f.append(stats.kurtosis(X))


    df_features = pd.DataFrame(
        index = [FEATURES],
        data = [Min,Max,Mean,Rms,Var,Std,Power,Peak,P2p,CrestFactor,Skew,Kurtosis, Max_f,Sum_f,Mean_f,Var_f,Peak_f,Skew_f,Kurtosis_f])
    
    return df_features


results = features_extraction(df)

print(results.head(5))













