# файлик генерит игрушечный датасет для обучения и дебага
from numpy import asarray
from numpy import savetxt
import numpy as np 
import scipy.stats as sts 
import sys

dataset_size = int(input('Введи количество строк в генерируемом датасете'))
feature_number = int(input('Введи количество фичей в генерируемом датасете'))
dataset_name = input('Введи название для файла, хранящего датасет, например data.csv')
labels_name = input('Введи название для файла, хранящего целевую переменную, например labels.csv')
print('Созданные файлы находятся в директории ./data')

def f(x):
    return x @ sts.uniform.rvs(size=feature_number)
    
X = np.random.rand(dataset_size, feature_number)
y = f(X) + sts.norm(0, scale=0.1).rvs(size=dataset_size)

# save to csv file
savetxt('./data/' + dataset_name, X, delimiter=',')
savetxt('./data/' + labels_name, y, delimiter=',')
