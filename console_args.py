'''
Консольное взаимодействие
'''
import argparse
import sys
import re        
import importlib

'''
Аргументы для запуска из консоли
@return: доступ к ним
'''
def console_args():
    parser = argparse.ArgumentParser(description='Stohastic Gradient Descent',
                                     prog='sgd', fromfile_prefix_chars='@')
    parser.add_argument('--steps', '-s', type=int,
                        help='Запуск алгоритма на указанное количество шагов', default=10**3)
    parser.add_argument('--sync', type=int,
                        help='WIP', default=10**2)
    parser.add_argument('--dataset', '-d',
                        help='Путь к файлу с dataset', default='./data/default_data.csv')
    parser.add_argument('--label', '-l',
                        help='Путь к файлу с label', default='./data/default_labels.csv')
    parser.add_argument('--metrics', '-m',
                        help='python-script, содержащий реализацию \
                        функции metrics для метрики и grad(X, y, w) для \
                        производной')
    parser.add_argument('--batch-size', '-b', type=int,
                        help='Размер batch', default=16)
    parser.add_argument('--precision', '-p', type=float,
                        help='тудуду', default=10 ** -1)
    return parser.parse_args()
