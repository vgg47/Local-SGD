import argparse
import numpy as np
import os

from numpy import savetxt, loadtxt


def parse_args():
    parser = argparse.ArgumentParser(description='Data Generator',
                                     prog='data gen', fromfile_prefix_chars='@')
    parser.add_argument('--step', '-s', type=int, default=10000)
    parser.add_argument('--dataset', '-d',
                        help='Путь к файлу с dataset', default='./data/default_data.csv')
    parser.add_argument('--label', '-l',
                        help='Путь к файлу с label', default='./data/default_labels.csv')
    return parser.parse_args()


args = parse_args()

X = loadtxt(args.dataset, delimiter=',')
y = loadtxt(args.label, delimiter=',')
n = y.shape[0]
if not os.path.exists('./parts'):
    os.makedirs('./parts')
size = args.step

while(size <= n):
    print(f'Creating part with {size} elements...')
    X_part = X[:size]
    y_part = y[:size]
    savetxt('./parts/' + f'dataset_part_{size}', X_part, delimiter=',')
    savetxt('./parts/' + f'label_part_{size}', y_part, delimiter=',')
    size += args.step