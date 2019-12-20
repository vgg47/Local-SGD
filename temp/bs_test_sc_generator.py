'''
Генератор bash скрипта для запуска на кластере
'''

'''
Запуск с изменяемыми параметрами mini-batch size, sync
'''
# for worker in range(1, 21, 2):
#     print ("mpiexec --oversubscribe -n " + 
#         str(worker) + " python3 script.py --sync " + 
#         str(256) + " -b " + str(4))
#     print ("mpiexec --oversubscribe -n " + 
#         str(worker) + " python3 script.py --sync " + 
#         str(256) + " -b " + str(16))
#     print ("mpiexec --oversubscribe -n " + 
#         str(worker) + " python3 script.py --sync " + 
#         str(256) + " -b " + str(32))
#     print ("mpiexec --oversubscribe -n " + 
#         str(worker) + " python3 script.py --sync " + 
#         str(256) + " -b " + str(256))

'''
Запуск алгоритма на срезках разного размера
'''
for size in range(10000, 100001, 10000):
    print ("mpiexec --oversubscribe -n " + 
        str(4) + " python3 script.py --sync " + 
        str(1000) + " -b " + str(256) + " -d ./parts/dataset_part_" +
        str(size) + " -l ./parts/label_part_" + str(size))
    