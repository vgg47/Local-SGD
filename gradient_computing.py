# В этом файлике реализуются методы для вычисления градиентов
# и их синхронизации

import numpy as np 
import scipy.stats as sts 

full_data = np.loadtxt('./data/data.csv', delimiter=',')
full_labels = np.loadtxt('./data/label.csv', delimiter=',')

def mse_metric(X, y, w):
    ''' X - dataset
        y - target feature
        w - weights
    '''
    return (y - X @ w).transpose().dot(y - X @ w) / (2 * len(y))

def mae_metric(y, y_pred):
    return sum(np.abs(y - y_pred)) / len(y)

def predict(data, weights):
    return data @ weights

def mse_grad(X, y, w):
    ''' X - dataset
        y - target feature
        w - weights
    '''
    return (-X.transpose() @ y +  X.transpose() @ X.dot(w)) / X.shape[0]

def gradient_step(X, y, w, step_size):
    ''' Делает шаг градиентного спуска '''
    return w - step_size * 2. * X @ (X @ w - y) / y.size


def sync(w_new):
    # заглушка
    return w_new

def choose_step_size(cur_step):
    # заглушка
    return 0.01 / cur_step