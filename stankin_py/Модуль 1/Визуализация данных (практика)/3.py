# Created on : 30 июл. 2024 г., 16:15:49
# Author     : Илья


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
x,y,z = axes3d.get_test_data()

print(x,y,z)