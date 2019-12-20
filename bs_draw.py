import json
import matplotlib.pyplot as plt
import numpy as np
import time

from numpy import loadtxt
from simple_grad_descend import SGD

V = 20

##############################################

dataset_name = './data/default_data.csv'
labels_name = './data/default_labels.csv'

X = loadtxt(dataset_name, delimiter=',')
y = loadtxt(labels_name, delimiter=',') 
start_sgd, stop_sgd = 0, 0

start_sgd += time.time()
SGD(X, y)
stop_sgd += time.time()
k = stop_sgd - start_sgd

##############################################

logs = json.load(open('logs.json'))
t = logs['algorithm_time']
v = logs['version']
ta = sorted([t[i] / k for i in range(len(v)) if v[i] == V])

h1b1 = []
h1b2 = []
h2b1 = []
h2b2 = []

i = 0
while i < len(ta):
    h1b1.append(ta[i])
    i += 1
    h1b2.append(ta[i])
    i += 1
    h2b1.append(ta[i])
    i += 1
    h2b2.append(ta[i])
    i += 1

x = np.arange(1, 21, 2)

plt.figure(figsize=(12, 8))

plt.plot(x, h1b1, label='b = 4')
plt.plot(x, h1b2, label='b = 16')
plt.plot(x, h2b1, label='b = 32')
plt.plot(x, h2b2, label='b = 256')

plt.xlabel('Количество работников', fontsize=15)
plt.ylabel('Ускорение по сравнению с SGD', fontsize=15)
plt.title("Зависимость ускорения от mini-batch size", fontsize=15)
plt.legend()
plt.savefig('./img/test_bs_h')