import numpy as np

a = np.random.rand(3,2)

# Normalised [0,1]
b = (a - np.min(a))/np.ptp(a)

# Normalised [0,255] as integer: don't forget the parenthesis before astype(int)
c = (255*(a - np.min(a))/np.ptp(a)).astype(int)        

# Normalised [-1,1]
d = 2.*(a - np.min(a))/np.ptp(a)-1