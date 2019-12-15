import argparse
import functools
import matplotlib.pyplot as plt
import numpy as np
import time

from data_generator import create_data
from gradient_computing import mse_metric
from numpy import loadtxt
from scipy.optimize import minimize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d',
                        help='Путь к файлу с dataset', default='./data/default_data.csv')
    parser.add_argument('--label', '-l',
                        help='Путь к файлу с label', default='./data/default_labels.csv')
    return parser.parse_args()

def mse_test(w, X=None, y=None):
    return mse_metric(X, y, w)

def experiment(mse_test, X, y, method, init=None, sample_size=5):
    if init is None:
        init = np.ones(X.shape[1])
    size = 10
    k = 1
    n = X.shape[0]
    times = []
    sizes = []
    while(size <= n):
        print(size)
        mse = functools.partial(mse_test, X=X[:size], y=y[:size])
        sample = []
        for _ in range(sample_size):
            start = time.time()
            minimize(mse, init, method=method)
            stop = time.time()
            sample.append(stop-start)
        times.append(np.array(sample).mean())
        sizes.append(k)
        size *= 10
        k += 1

    return np.array(sizes), np.array(times)


args = parse_args()
X = loadtxt(args.dataset, delimiter=',')
y = loadtxt(args.label, delimiter=',')  
plt.figure(figsize=(15, 10))
x_plot, y_plot = experiment(mse_test, X, y, 'nelder-mead')
plt.plot(x_plot, y_plot, label='nelder-mead')
x_plot, y_plot = experiment(mse_test, X, y, 'powell')
plt.plot(x_plot, y_plot, label='powell')
plt.xticks(np.arange(5)+1, np.array(['$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$']))
plt.xlabel('Размер', fontsize=15)
plt.ylabel('Время', fontsize=15)
plt.title('Зависимость времени работы методов от размера датасета', fontsize=15)
plt.legend()
plt.savefig('./img/test')
#mse = functools.partial(mse_test, X=X, y=y)
#ans1 = minimize(mse, init)
#ans2 = minimize(mse, init, method='nelder-mead')
#ans3 = minimize(mse, init, method='powell')
#print(ans1.fun, ans2.fun, ans3.fun)
