# файлик генерит игрушечный датасет для обучения и дебага
from numpy import asarray
from numpy import savetxt
import numpy as np 
import scipy.stats as sts 
import sys

dataset_size = int(input())
feature_number = int(input())

def f(x):
    return x @ sts.uniform.rvs(size=feature_number)
    
X = np.random.rand(dataset_size, feature_number)
y = f(X) + sts.norm(0, scale=0.1).rvs(size=dataset_size)

# save to csv file
savetxt('./data/data.csv', X, delimiter=',')
savetxt('./data/label.csv', y, delimiter=',')
