import numpy as np
import random
import json
import scipy.stats as sts
import sys
import time

from console_args import console_args
from gradient_computing import gradient_step, sync, choose_step_size, mse_metric, stepsize
from mpi4py import MPI
from numpy import loadtxt
from scipy.spatial import distance

VERSION = 21

args = console_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
steps_number = args.steps
communications_number = args.sync
min_mse = args.precision
# batch_size = 50
batch_size = args.batch_size

if rank == 0:  
    start_time = time.process_time()
    # генерируем последовательность таймстемпов для синхронизаций
    sync_timestamps = set(np.arange(1, steps_number, communications_number, dtype=int))

    # загружаем данные из файлов
    full_data = loadtxt(args.dataset, delimiter=',')
    full_labels = loadtxt(args.label, delimiter=',')    

    full_data = np.hstack([np.ones((full_data.shape[0], 1)), full_data])

    feature_number = full_data.shape[1]
    np.random.seed(17)

    w = np.hstack([1 , np.random.rand(feature_number - 1)])
    print(f'initial weights: {w}')
    print(f'mse for random weights {mse_metric(full_data, full_labels, w)}')


    # определяем размер батча для каждого воркера
    shard_size = int(full_data.shape[0] / comm.Get_size())
    
    data_loading_time = time.process_time()
    print(f'data loading time is {data_loading_time - start_time}')
    # рассылаем данные по воркерам
    for idx in range(1, comm.size):
        
        # отправляем размер батча и количество фичей        
        comm.send(shard_size, dest=idx, tag=75)
        comm.send(feature_number, dest=idx, tag=76)
        
        # отправляем начальные веса, таймстемпы, данные и метки !!!! здесь можно заменить на gathering
        comm.Send(w, dest=idx, tag=74) 
        comm.Send(full_data[idx * shard_size: (idx + 1)  * shard_size], dest=idx, tag=78)
        comm.Send(full_labels[idx * shard_size: (idx + 1)  * shard_size], dest=idx, tag=79)
    
    # определяем данные для обработки нулевым воркером
    X = full_data[0: shard_size]
    y = full_labels[0: shard_size]
    
else:
    # принимаем размер батча и количество фичей
    shard_size = comm.recv(source=0, tag=75)
    feature_number = comm.recv(source=0, tag=76)
    
    # принимаем данные и метки
    w = np.empty(feature_number, dtype=np.float64)
    sync_timestamps = None
    X = np.empty((shard_size, feature_number), dtype=np.float64)
    y = np.empty(shard_size, dtype=np.float64)
    
    comm.Recv(w, source=0, tag=74)
    comm.Recv(X, source=0, tag=78)
    comm.Recv(y, source=0, tag=79)
    

sync_timestamps = comm.bcast(sync_timestamps, root=0)

# здесь начинается градиентный спуск
data_sending_time = time.process_time()
if rank == 0:
    print(f'data sending time is {data_sending_time - data_loading_time}')

cur_mse = 1 
cur_step = 0
stopping_criterion = True

# работа алгоритма завершается, если  мсе меньше
# заданного значения или же после определенного количества шагов 
while cur_step < steps_number and stopping_criterion:
    batch_idxs = np.random.randint(X.shape[0], size=batch_size)

    # выбираем размер шага (learning rate)
    step_size = stepsize(X, cur_step, communications_number)
    # делаем шаг
    w  = gradient_step(X, y, w, batch_idxs, step_size)
    # если текущий таймстемп лежит в множестве синхронизируемых, то синхронизируемся))
    if cur_step + 1 in sync_timestamps:
        w = sync(w, comm)
    # смотрим на то, как сильно изменились веса
    cur_mse = mse_metric(X, y, w)
    if rank == 0 and cur_mse > min_mse:
        stopping_criterion = comm.bcast(False, root=0)
    cur_step += 1

w = sync(w, comm)

final_time = time.process_time()
if rank == 0:
    # print(f'algorithm time is {final_time - data_sending_time}')
    # print(f'general time is {final_time - start_time}')
    # print(f'final value mse after {steps_number} for {comm.size} workers is {mse_metric(full_data, full_labels, w)}')
    with open('logs.json') as logfile:
        logs = json.load(logfile)
    logs['general_time'].append(final_time - start_time)
    logs['steps'].append(steps_number) # cur_step?
    logs['algorithm_time'].append(final_time - data_sending_time)
    logs['mse'].append(mse_metric(full_data, full_labels, w))
    logs['version'].append(VERSION)
    logs['workers'].append(comm.size)
    with open('logs.json', 'w') as logfile:
        json.dump(logs, logfile)
