import numpy as np
import random
import scipy.stats as sts
import sys

from console_args import console_args
from gradient_computing import gradient_step, sync, choose_step_size, mse_metric
from mpi4py import MPI
from numpy import loadtxt
from scipy.spatial import distance
import time

args = console_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
steps_number = args.steps
communications_number = args.sync
min_weight_dist = args.precision


if rank == 0:  
    start_time = time.process_time()
    # генерируем последовательность таймстемпов для синхронизаций
    # sync_timestamps = np.sort(sts.randint.rvs(low=0, high=steps_number, size=communications_number))
    sync_timestamps = set(np.linspace(1, steps_number, communications_number, dtype=int))

    # загружаем данные из файлов
    full_data = loadtxt(args.dataset, delimiter=',')
    full_labels = loadtxt(args.label, delimiter=',')    

    full_data = np.hstack([np.ones((full_data.shape[0], 1)), full_data])

    feature_number = full_data.shape[1]
    np.random.seed(17)
    w = np.hstack([1 , np.random.rand(feature_number - 1)])
    print(w)
    print(f'mse for random weights {mse_metric(full_data, full_labels, w)}')


    # определяем размер батча для каждого воркера
    batch_size = int(full_data.shape[0] / comm.Get_size())
    
    data_loading_time = time.process_time()
    print(f'data loading time is {data_loading_time - start_time}')
    # рассылаем данные по воркерам
    for idx in range(1, comm.size):
        
        # отправляем размер батча и количество фичей        
        comm.send(batch_size, dest=idx, tag=75)
        comm.send(feature_number, dest=idx, tag=76)
        
        # отправляем начальные веса, таймстемпы, данные и метки !!!! здесь можно заменить на gathering
        comm.Send(w, dest=idx, tag=74) 
        # comm.Send(sync_timestamps, dest=idx, tag=77) 
        comm.Send(full_data[idx * batch_size: (idx + 1)  * batch_size], dest=idx, tag=78)
        comm.Send(full_labels[idx * batch_size: (idx + 1)  * batch_size], dest=idx, tag=79)
    
    # определяем данные для обработки нулевым воркером
    X = full_data[0: batch_size]
    y = full_labels[0: batch_size]
    
else:
    # принимаем размер батча и количество фичей
    batch_size = comm.recv(source=0, tag=75)
    feature_number = comm.recv(source=0, tag=76)
    
    # принимаем данные и метки
    w = np.empty(feature_number, dtype=np.float64)
    sync_timestamps = None
    X = np.empty((batch_size, feature_number), dtype=np.float64)
    y = np.empty(batch_size, dtype=np.float64)
    
    comm.Recv(w, source=0, tag=74)
    # comm.Recv(sync_timestamps, source=0, tag=77)
    comm.Recv(X, source=0, tag=78)
    comm.Recv(y, source=0, tag=79)
    

sync_timestamps = comm.bcast(sync_timestamps, root=0)

# тутачки начинается градиентный спуск
data_sending_time = time.process_time()
if rank == 0:
    print(f'data sending time is {data_sending_time - data_loading_time}')

weight_dist = np.inf
cur_step = 0
# работа алгоритма завершается, если  шаг градиентного метода меньше
# заданного значения или же после определенного количества шагов 
while (cur_step < steps_number and weight_dist > min_weight_dist):
    # выбираем размер шага (learning rate)
    step_size = choose_step_size(cur_step) # тут заглушка!!!
    # делаем шаг
    w_new  = gradient_step(X, y, w, step_size)
    # если текущий таймстемп лежит в множестве синхронизируемых, то синхронизируемся))
    if cur_step + 1 in sync_timestamps:
        w_new = sync(w_new, comm) # тут заглушка!!!
    # смотрим на то, как сильно изменились веса
    weight_dist = distance.euclidean(w, w_new)
    w = w_new
    cur_step += 1

w_new = sync(w_new, comm)

final_time = time.process_time()
if rank == 0:
    print(f'algorithm time is {final_time - data_sending_time}')
    print(f'general time is {final_time - start_time}')
    print(f'final value mse after {steps_number} for {comm.size} is {mse_metric(full_data, full_labels, w)}')