# В этом файлике реализуются методы для вычисления градиентов
# и их синхронизации

import numpy as np 
import scipy.stats as sts 


def mse_metric(X, y, w):
    ''' X - dataset
        y - target feature
        w - weights
    '''
    return (y - X @ w).T.dot(y - X @ w)

def mae_metric(y, y_pred):
    return sum(np.abs(y - y_pred))

def predict(data, weights):
    return data @ weights

def mse_grad(X, y, w):
    ''' X - dataset
        y - target feature
        w - weights
    '''
    return X.T @ (X @ w - y) / X.shape[0]

def gradient_step(X, y, w, batch_idxs, step_size):
    ''' Делает шаг градиентного спуска '''
    return w - step_size * 2. * X[batch_idxs].T @ (X[batch_idxs] @ w - y[batch_idxs]) / len(batch_idxs)

def sync(w_new, comm):
    all_w = comm.gather(w_new, root=0)
    if comm.Get_rank() == 0:
        w_new = np.mean(all_w, axis=0)
    w_new = comm.bcast(w_new, root=0)
    return w_new


def L(X):
    '''
    Считает константу Липшица
    как максимальное собственное значение
    '''
    return np.max(np.linalg.eig(2 * X.T @ X)[0])


def mu(X):
    '''
    Считает константу мю сильной выпуклости
    как минимальное собственное значение
    '''
    return np.min(np.linalg.eig(2 * X.T @ X)[0])


def a(X, max_gap):
    k = L(X) / mu(X)
    return max(16 * k, max_gap)


def stepsize(X, cur_step, max_gap):
    '''
    Размер шага = 4 / myu(a + t), где
    a = max(16k, H) - параметр сдвига,
    k = L / myu
    '''
    return 4 / (mu(X) * (a(X, max_gap) + 1))
