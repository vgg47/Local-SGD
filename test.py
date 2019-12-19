import argparse
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d',
                        help='Путь к файлу с dataset', default='./data/default_data.csv')
    parser.add_argument('--label', '-l',
                        help='Путь к файлу с label', default='./data/default_labels.csv')
    parser.add_argument('--step', type=int, default=10000)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--draw', type=int, default=0)
    return parser.parse_args()

def mse_test(w, X=None, y=None):
    return mse_metric(X, y, w)

def mse_grad_test(w, X=None, y=None):
    return mse_grad(X, y, w)

args = parse_args()
X = loadtxt(args.dataset, delimiter=',')
y = loadtxt(args.label, delimiter=',') 
n = y.shape[0]
size = args.step
comparsion = {'bfgs': [], 'sgd': [], 'nesterov': [], 'size': []}

if args.draw == 1:
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
        for _ in range(args.repeat):
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

        comparsion['bfgs'].append((stop_bfgs-start_bfgs) / args.repeat)
        comparsion['sgd'].append((stop_sgd - start_sgd) / args.repeat)
        comparsion['nesterov'].append((stop_nest - start_nest) / args.repeat)
        comparsion['size'].append(size)
        size += args.step

    print(comparsion)
    plt.figure(figsize=(12, 8))
    x = comparsion['size']
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






#make_plot(args)
#ans1 = minimize(mse, init)
#ans2 = minimize(mse, init, method='nelder-mead')
#ans3 = minimize(mse, init, method='powell')
init = np.ones(X.shape[1])
#ans4 = minimize(mse, init, method='BFGS')
#print(ans4)
#a = time.time()
#ans5 = minimize(mse, init, method='BFGS', jac=grad)
#b = time.time()
#print(b-a)
#print(ans5)
#print(ans5)
#print(ans1.fun, ans2.fun, ans3.fun)
L = np.max(np.linalg.eig(2 * X.T @ X)[0])
a = time.time()
x = nesterov_descent(mse, L, init, grad)
b = time.time()
print(b-a)
print(mse(x))