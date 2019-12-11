import numpy as np
import random
import scipy.stats as sts
import sys
import time

from console_args import console_args
from gradient_computing import gradient_step, sync, choose_step_size, mse_metric
from mpi4py import MPI
from numpy import loadtxt
from scipy.spatial import distance

args = console_args()

steps_number = args.steps
communications_number = args.sync
min_weight_dist = args.precision
batch_size = args.batch_size

start_time = time.process_time()

 # загружаем данные из файлов
X = loadtxt(args.dataset, delimiter=',')
y = loadtxt(args.label, delimiter=',')  


X = np.hstack([np.ones((X.shape[0], 1)), X])

feature_number = X.shape[1]

np.random.seed(17)

w = np.hstack([np.arange(1, feature_number + 1)])
# w = np.hstack([1 , np.random.rand(feature_number - 1)])
print(w)
print(f'mse for random weights {mse_metric(X, y, w)}')

data_loading_time = time.process_time()
print(f'data loading time is {data_loading_time - start_time}')



weight_dist = np.inf
cur_step = 0
# работа алгоритма завершается, если  шаг градиентного метода меньше
# заданного значения или же после определенного количества шагов 
while (cur_step < steps_number and weight_dist > min_weight_dist):
    batch_idxs = np.random.randint(X.shape[0], size=batch_size)

    # выбираем размер шага (learning rate)
    step_size = choose_step_size(cur_step) # тут заглушка!!!
    # делаем шаг
    w_new  = gradient_step(X, y, w, batch_idxs, step_size)
    # если текущий таймстемп лежит в множестве синхронизируемых, то синхронизируемся))
    # смотрим на то, как сильно изменились веса
    weight_dist = distance.euclidean(w, w_new)
    w = w_new
    cur_step += 1


final_time = time.process_time()
print(f'algorithm time is {final_time - data_loading_time}')
print(f'general time is {final_time - start_time}')
print(f'final value mse after {steps_number} for 1 worker is {mse_metric(X, y, w)}')