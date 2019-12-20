import json
import matplotlib.pyplot as plt
import numpy as np

t = json.load(open('algotimes.json'))
ta = t[0]
tg = t[1]
plt.figure(figsize=(12, 8))
x = np.arange(10000, 100001, 10000)
plt.plot(x, ta, label='algorithm')
plt.plot(x, tg, label='general')
plt.xticks(np.arange(11)*10000)
plt.xlabel('Размер', fontsize=15)
plt.ylabel('Время', fontsize=15)
plt.title('Зависимость времени работы методов от размера датасета', fontsize=15)
plt.legend()
plt.savefig('./img/test_algo')