import numpy as np
import random
import scipy.stats as sts
import sys

from console_args import console_args
from gradient_computing import gradient_step, sync, choose_step_size, mse_metric
from mpi4py import MPI
from numpy import loadtxt
from scipy.spatial import distance


args = console_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
steps_number = args.steps
communications_number = args.precision

if rank == 0:  
    
    # генерируем последовательность таймстемпов для синхронизаций
    # sync_timestamps = np.sort(sts.randint.rvs(low=0, high=steps_number, size=communications_number))
    sync_timestamps = np.sort(np.array(random.sample(range(steps_number), communications_number)))
    
    # загружаем данные из файлов
    full_data = loadtxt('./data/data.csv', delimiter=',')
    full_labels = loadtxt('./data/label.csv', delimiter=',')    

    full_data = np.hstack([np.ones((full_data.shape[0], 1)), full_data])

    feature_number = full_data.shape[1]
    np.random.seed(17)
    w = np.hstack([1 , np.random.rand(feature_number - 1)])
    print(w)

    # определяем размер батча для каждого воркера
    batch_size = int(full_data.shape[0] / comm.Get_size())
    
    # рассылаем данные по воркерам
    for idx in range(1, comm.size):
        
        # отправляем размер батча и количество фичей        
        comm.send(batch_size, dest=idx, tag=75)
        comm.send(feature_number, dest=idx, tag=76)
        
        # отправляем начальные веса, таймстемпы, данные и метки !!!! здесь можно заменить на gathering
        comm.Send(w, dest=idx, tag=74) 
        comm.Send(sync_timestamps, dest=idx, tag=77) 
        comm.Send(full_data[idx * batch_size: (idx + 1)  * batch_size], dest=idx, tag=78)
        comm.Send(full_labels[idx * batch_size: (idx + 1)  * batch_size], dest=idx, tag=79)
    
    # определяем данные для обработки нулевым воркером
    X = full_data[0: batch_size]
    y = full_labels[0: batch_size]
    print('from ', rank, X.shape, y.shape)
    
else:
    # принимаем размер батча и количество фичей
    batch_size = comm.recv(source=0, tag=75)
    feature_number = comm.recv(source=0, tag=76)
    
    # принимаем данные и метки
    w = np.empty(feature_number, dtype=np.float64)
    sync_timestamps = np.empty(communications_number, dtype=np.float64)
    X = np.empty((batch_size, feature_number), dtype=np.float64)
    y = np.empty(batch_size, dtype=np.float64)
    
    comm.Recv(w, source=0, tag=74)
    comm.Recv(sync_timestamps, source=0, tag=77)
    comm.Recv(X, source=0, tag=78)
    comm.Recv(y, source=0, tag=79)
    
    
# тутачки начинается градиентный спуск

min_weight_dist = 10 ** -6 # тут заглушка!!!
weight_dist = np.inf
cur_step = 0
print(mse_metric(X, y, w))
# работа алгоритма завершается, если  шаг градиентного метода меньше
# заданного значения или же после определенного количества шагов 
while (cur_step < steps_number and weight_dist > min_weight_dist):
    # выбираем размер шага (learning rate)
    step_size = choose_step_size(cur_step) # тут заглушка!!!
    # делаем шаг
    w_new  = gradient_step(X, y, w, step_size)
    # если текущий таймстемп лежит в множестве синхронизируемых, то синхронизируемся))
    if cur_step + 1 in sync_timestamps:
        w_new = sync(w_new) # тут заглушка!!!
    # смотрим на то, как сильно изменились веса
    weight_dist = distance.euclidean(w, w_new)
    w = w_new
    if rank == 0 and cur_step % 100 == 0:
        print(mse_metric(X, y, w))

    cur_step += 1

