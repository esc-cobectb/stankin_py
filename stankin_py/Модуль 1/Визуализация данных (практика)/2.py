# Created on : 30 июл. 2024 г., 16:15:49
# Author     : Илья


import matplotlib.pyplot as plt
import numpy as np


y = 200 + np.random.randn(10)
x = [x for x in range(len(y))]

plt.plot(x,y, '-')
plt.fill_between(x,y,195)

plt.show()
    
