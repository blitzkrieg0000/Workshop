import numpy as np


arr = np.array([[5, 4, 100, 11, -4], [3, 62, 52, 71, 42], [53, 5, 9, 72, 42], [43, 13, 12, 47, 42]])


print(arr[arr[..., 2].argsort(axis=0)])

