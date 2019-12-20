# файлик генерит игрушечный датасет для обучения и дебага
from numpy import asarray
from numpy import savetxt
import numpy as np 
import scipy.stats as sts 
import sys
import os

def f(x, feature_number):
    return x @ sts.uniform.rvs(size=feature_number)

def create_data(dataset_size, feature_number):
    X = sts.uniform.rvs(loc=1, scale=1, size=(dataset_size, feature_number))
    y = sts.uniform.rvs(loc=0, scale=1, size=dataset_size)
    # y = f(X, feature_number) + sts.norm(0, scale=0.1).rvs(size=dataset_size)
    return X, y

def main():
    dataset_size = int(input('Введи количество строк в генерируемом датасете:\n'))
    feature_number = int(input('Введи количество фичей в генерируемом датасете:\n'))
    dataset_name = input('Введи название для файла, хранящего датасет, например data.csv:\n')
    labels_name = input('Введи название для файла, хранящего целевую переменную, например labels.csv:\n')
    print('Созданные файлы находятся в директории ./data')

    X, y = create_data(dataset_size, feature_number)

    if not os.path.exists('./data'):
        os.makedirs('./data')

    # save to csv file
    savetxt('./data/' + dataset_name, X, delimiter=',')
    savetxt('./data/' + labels_name, y, delimiter=',')

if __name__ == "__main__":
    main()
