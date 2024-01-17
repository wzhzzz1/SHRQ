'''
import numpy as np
data = np.random.normal(loc=8, scale=4, size=100000)
data = np.clip(data, 0, 1023)
data = np.round(data).astype(int)
np.savetxt('data.txt', data, fmt='%d')
'''
import math

print(math.log(0.00020))

print(math.log(3.3227081775015936e-05))