import numpy as np
import random
import scipy.stats as sts
import sys
import time

from console_args import console_args
from gradient_computing import gradient_step, sync, mse_metric, stepsize
from numpy import loadtxt
from scipy.spatial import distance

args = console_args()
min_weight_dist = args.precision

start_time = time.process_time()

 # загружаем данные из файлов
X = loadtxt(args.dataset, delimiter=',')
y = loadtxt(args.label, delimiter=',')  

def SGD(X, y, min_weight_dist=10**-8, batch_size=1, steps_number=10**7, delta=0.1):
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    np.random.seed(17)
    w = np.hstack([1 , np.random.rand(X.shape[1] - 1)])

    step_size = 0.01
    weight_dist = np.inf
    cur_step = 0
    while (cur_step < steps_number and weight_dist > min_weight_dist and mse_metric(X, y, w) > delta):
        batch_idxs = np.random.randint(X.shape[0], size=batch_size)

        # делаем шаг
        w_new  = gradient_step(X, y, w, batch_idxs, step_size)
        # смотрим на то, как сильно изменились веса
        weight_dist = distance.euclidean(w, w_new)
        w = w_new
        cur_step += 1
