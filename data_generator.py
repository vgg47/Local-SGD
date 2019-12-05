# файлик генерит игрушечный датасет для обучения и дебага

import numpy as np 
import scipy.stats as sts 
from numpy import asarray
from numpy import savetxt


def f(x):
    return x @ sts.uniform.rvs(size=feature_number)
    
feature_number = sts.randint(10000)
X = np.random.rand(100000, feature_number)
y = f(X) + sts.norm(0, scale=0.3).rvs(size=100000)

# save to csv file
savetxt('./data/data1.csv', X, delimiter=',')
savetxt('data/label1.csv', y, delimiter=',')