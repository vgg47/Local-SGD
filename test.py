import argparse
import json
import functools
import matplotlib.pyplot as plt
import numpy as np
import time

from data_generator import create_data
from gradient_computing import mse_metric, mse_grad, L
from nesterov import nesterov_descent
from numpy import loadtxt
from scipy.optimize import minimize
from simple_grad_descend import SGD

V = 21

##########################

dataset_name = input('Введи название для файла, хранящего датасет, например data.csv:\n')
labels_name = input('Введи название для файла, хранящего целевую переменную, например labels.csv:\n')
step = input('Размер датасета\n')

repeat = input('Сколько раз повторять запуск алгоритма (результаты усредняются):\n')
draw = int(input('Вы хотите получить рисунок? 0 = Нет, 1 = Да:\n'))

if not dataset_name:
    dataset_name = './data/default_data.csv'
if not labels_name:
    labels_name = './data/default_labels.csv'
if not step:
    step = 10000
else:
    step = int(step)
if not repeat:
    repeat = 5
else:
    repeat = int(repeat)

##########################

def mse_test(w, X=None, y=None):
    return mse_metric(X, y, w)

def mse_grad_test(w, X=None, y=None):
    return mse_grad(X, y, w)

X = loadtxt(dataset_name, delimiter=',')
y = loadtxt(labels_name, delimiter=',') 
n = y.shape[0]

size = step
comparsion = {'bfgs': [], 'sgd': [], 'nesterov': [], 'size': []}

if draw == 1:
    while(size <= n):
        print(f'Counting...Size={size}')
        
        init = np.ones(X.shape[1])
        X_part = X[:size]
        y_part = y[:size]
        mse = functools.partial(mse_test, X=X_part, y=y_part)
        grad = functools.partial(mse_grad_test, X=X_part, y=y_part)
        start_bfgs, stop_bfgs = 0, 0
        start_sgd, stop_sgd = 0, 0
        start_nest, stop_nest = 0, 0
        for _ in range(repeat):
            #BFGS
            start_bfgs += time.time()
            minimize(mse, init, method='BFGS', jac=grad)
            stop_bfgs += time.time()

            #SGD
            start_sgd += time.time()
            SGD(X_part, y_part)
            stop_sgd += time.time()

            #Nesterov
            start_nest += time.time()
            nesterov_descent(mse, L(X_part), init, grad)
            stop_nest += time.time()

        comparsion['bfgs'].append((stop_bfgs-start_bfgs) / repeat)
        comparsion['sgd'].append((stop_sgd - start_sgd) / repeat)
        comparsion['nesterov'].append((stop_nest - start_nest) / repeat)
        comparsion['size'].append(size)
        size += step

    print(comparsion)

    ##############################################

    logs = json.load(open('logs.json'))
    t = logs['algorithm_time']
    v = logs['version']
    ta = sorted([t[i] for i in range(len(v)) if v[i] == V])

    plt.figure(figsize=(12, 8))
    x = comparsion['size']
    plt.plot(x, ta, label='LSGD')
    plt.plot(x, comparsion['bfgs'], label='BFGS')
    plt.plot(x, comparsion['sgd'], label='SGD')
    plt.plot(x, comparsion['nesterov'], label='Nesterov')
    plt.xticks(np.arange(11)*10000)
    plt.xlabel('Размер', fontsize=15)
    plt.ylabel('Время', fontsize=15)
    plt.title('Зависимость времени работы методов от размера датасета', fontsize=15)
    plt.legend()
    plt.savefig('./img/test')



mse = functools.partial(mse_test, X=X, y=y)
grad = functools.partial(mse_grad_test, X=X, y=y)

init = np.ones(X.shape[1])

L = np.max(np.linalg.eig(2 * X.T @ X)[0])
a = time.time()
x = nesterov_descent(mse, L, init, grad)
b = time.time()
print(b-a)
print(mse(x))