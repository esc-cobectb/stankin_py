# Created on : 30 июл. 2024 г., 16:15:49
# Author     : Илья


import matplotlib.pyplot as plt

x =  [1,2,3,4,5,6]
y1 = [2,4,6,4,2,1]
y2 = [3,6,8,5,3,2]
plt.plot(x, y1, label='y1')
plt.plot(x, y2, label='y2')
plt.plot()

plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.title('Тестовый линейный график')
plt.legend()

plt.show()

