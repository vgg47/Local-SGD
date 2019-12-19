import argparse
import functools
import matplotlib.pyplot as plt
import numpy as np
import time

from data_generator import create_data
from gradient_computing import mse_metric, mse_grad
from nesterov import nesterov_descent
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

def mse_grad_test(w, X=None, y=None):
    return mse_grad(X, y, w)

def experiment(mse_test, X, y, method, init=None, sample_size=1000, kwargs={}):
    if init is None:
        init = np.ones(X.shape[1])
    size = 2500
    n = X.shape[0]
    times = [0]
    sizes = [0]
    while(size <= n):
        print(size)
        mse = functools.partial(mse_test, X=X[:size], y=y[:size])
        if kwargs and 'jac' in kwargs:
            kwargs['jac'] = functools.partial(mse_grad_test, X=X[:size], y=y[:size])
        sample = []
        for _ in range(sample_size):
            start = time.time()
            minimize(mse, init, method=method, **kwargs)
            stop = time.time()
            sample.append(stop-start)
        times.append(np.array(sample).mean())
        sizes.append(size)
        size += 5*1000

    return np.array(sizes), np.array(times)


def make_plot(args):
    X = loadtxt(args.dataset, delimiter=',')
    y = loadtxt(args.label, delimiter=',')  
    plt.figure(figsize=(12, 8))
    #x_plot, y_plot = experiment(mse_test, X, y, 'nelder-mead')
    #plt.plot(x_plot, y_plot, label='nelder-mead')
    #x_plot, y_plot = experiment(mse_test, X, y, 'powell')
    #plt.plot(x_plot, y_plot, label='powell')
    x_plot, y_plot = experiment(mse_test, X, y, 'BFGS', kwargs= {'jac':None})
    plt.plot(x_plot, y_plot, label='BFGS')
    plt.xticks(np.arange(11)*10000)
    plt.xlabel('Размер', fontsize=15)
    plt.ylabel('Время', fontsize=15)
    plt.title('Зависимость времени работы методов от размера датасета', fontsize=15)
    plt.legend()
    plt.savefig('./img/test')


args = parse_args()
#make_plot(args)
X = loadtxt(args.dataset, delimiter=',')
y = loadtxt(args.label, delimiter=',')  
mse = functools.partial(mse_test, X=X, y=y)
grad = functools.partial(mse_grad_test, X=X, y=y)
#ans1 = minimize(mse, init)
#ans2 = minimize(mse, init, method='nelder-mead')
#ans3 = minimize(mse, init, method='powell')
init = np.ones(X.shape[1])
#ans4 = minimize(mse, init, method='BFGS')
#print(ans4)
a = time.time()
ans5 = minimize(mse, init, method='BFGS', jac=grad)
b = time.time()
print(b-a)
#print(ans5)
#print(ans5)
#print(ans1.fun, ans2.fun, ans3.fun)
L = 1#np.max(np.linalg.eig(2 * X.T @ X)[0])
a = time.time()
x = nesterov_descent(mse, L, init, grad)
b = time.time()
print(b-a)
print(mse(x))