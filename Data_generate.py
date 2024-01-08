
import numpy as np
data = np.random.normal(loc=8, scale=4, size=100000)
data = np.clip(data, 0, 1023)
data = np.round(data).astype(int)
np.savetxt('data.txt', data, fmt='%d')

