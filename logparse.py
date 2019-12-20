import numpy
import json

V = 8

logs = json.load(open('logs.json'))
t1 = logs['general_time']
t2 = logs['algorithm_time']
v = logs['version']
tg = sorted([t1[i] for i in range(len(v)) if v[i] == V])
ta = sorted([t2[i] for i in range(len(v)) if v[i] == V])
t = [ta, tg]
json.dump(t, open('algotimes.json', 'w'))
