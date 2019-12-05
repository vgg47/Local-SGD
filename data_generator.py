# файлик генерит игрушечный датасет для обучения и дебага

import numpy as np 
import scipy.stats as sts 
from numpy import asarray
from numpy import savetxt

dataset_size = int(sys.argc[0])
feature_number = int(sys.argc[1])

def f(x):
    return x @ sts.uniform.rvs(size=feature_number)
    
X = np.random.rand(dataset_size, feature_number)
y = f(X) + sts.norm(0, scale=0.1).rvs(size=dataset_size)

# save to csv file
savetxt('./data/data1.csv', X, delimiter=',')
savetxt('./data/label1.csv', y, delimiter=',')