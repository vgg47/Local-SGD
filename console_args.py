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
def parse_args():
    parser = argparse.ArgumentParser(description='Stohastic Gradient Descent',
                                     prog='sgd', fromfile_prefix_chars='@')
    parser.add_argument('--steps', '-s', type=int,
                        help='Запуск алгоритма на указанное количество шагов', default=10**6)
    parser.add_argument('--sync',
                        help='WIP')
    parser.add_argument('--dataset', '-d',
                        help='Путь к файлу с dataset', required=True)
    parser.add_argument('--label', '-l',
                        help='Путь к файлу с label', required=True)
    parser.add_argument('--metrics', '-m',
                        help='python-script, содержащий реализацию \
                        функции metrics для метрики и grad для \
                        её производной')
    parser.add_argument('--precision', '-p', type=int,
                        help='тудуду', default=1)
    return parser.parse_args()

'''
Проверка ввода
'''
def console_args():
    args = parse_args()
    
    # print(args.dataset)
    # print(args.label)
    # print(args.metrics)

    m = importlib.import_module(args.metrics)
    func = m.metrics
    # func()

    return args

console_args()