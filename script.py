from mpi4py import MPI
from numpy import loadtxt
import scipy.stats as sts
import sys

import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
steps_number = int(sys.arvg[0])
communications_number = int(sys.argv[1])

if rank == 0:  
    
    # генерируем последовательность таймстемпов для синхронизаций
    sync_timestamps = np.sort(sts.randint.rvs(low=0, high=steps_number, size=communications_number))
    
    # загружаем данные из файлов
    full_data = loadtxt('data.csv', delimiter=',')
    full_labels = loadtxt('label.csv', delimiter=',')
    
    # определяем размер батча для каждого воркера
    batch_size = int(full_data.shape[0] / comm.Get_size())
    
    # рассылаем данные по воркерам
    for idx in range(1, comm.size):
        
        # отправляем размер батча и количество фичей        
        comm.send(batch_size, dest=idx, tag=77)
        comm.send(full_data.shape[1], dest=idx, tag=77)
        
        # отправляем таймстемпы, данные и метки !!!! здесь можно заменить на gathering
        comm.Send(sync_timestamps, dest=idx, tag=77) 
        comm.Send(full_data[idx * batch_size: (idx + 1)  * batch_size], dest=idx, tag=77)
        comm.Send(full_labels[idx * batch_size: (idx + 1)  * batch_size], dest=idx, tag=77)
    
    # определяем данные для обработки нулевым воркером
    data = full_data[0: batch_size]
    labels = full_labels[0: batch_size]
    print('from ', rank, data.shape, labels.shape)
    
else:
    # принимаем размер батча и количество фичей
    batch_size = comm.recv(source=0, tag=77)
    feature_number = comm.recv(source=0, tag=77)
    
    # принимаем данные и метки
    sync_timestamps = np.empty(communications_number, dtype=np.float64)
    data = np.empty((batch_size, feature_number), dtype=np.float64)
    labels = np.empty(batch_size, dtype=np.float64)
    
    comm.Recv(sync_timestamps, source=0, tag=77)
    comm.Recv(data, source=0, tag=77)
    comm.Recv(labels, source=0, tag=77)
    
    # вычисляем гра
    grad = compute_gradient(data)
    