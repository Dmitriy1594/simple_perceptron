import csv
import os
import shutil
import random
from math import tanh, cos, exp, cosh
import itertools

def data_to_scv(path_with_name, data):
    with open(f'{path_with_name}', 'w', newline='') as csvfile:
        fieldnames = ['k', 'w', 'y', 'E']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for d in data:
            writer.writerow(d)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def round(x):
    if x >= 0.5:
        return 1
    elif x < 0.5:
        return 0

def think(net):
    return round((1/2)*(tanh(net) + 1))

def net(W, X):
    net = 0
    for (w,x) in zip(W[1:], X[1:]):
        net += w*x
    net += W[0]
    return net

def delta(t, y):
    return t - y

def delta_w(nu, delta, net, x):
    # der = 1 - (1 / (2 * (cos(net)**2)))
    # der = 0.5 / (cosh(net) ** 2)
    der = 0.5 - 0.5 * (tanh(net))**2
    return nu * delta * der * x

def recount_W(W, X, d, n, nu):
    for (i, w) in enumerate(W):
            W[i] += delta_w(nu, d, n, X[i])
    return W

def totalError(Y, F):
    E = 0
    for (y, f) in zip(Y, F):
        if y != f:
            E += 1
    return E

# function settings
def write_Data(file, k, Y, W, E):
    file.write('k = ' + str(k) + '\n')
    file.write('Y = (' + str(Y)[1:-1] + '),\n')
    file.write('W = (' + str([float(j) for j in ['%.3f' % i for i in W]])[1:-1] + '), E = ' + str(E) + '\n')
    file.write('\n\n')

def IntToByte(x):
    n = '' if x > 0 else '0'
    while x > 0:
        y = str(x % 2)
        n = y + n
        x = int(x / 2)
    return n

def bin_generation(n):
    X = list()
    count = n**2
    for i in range(0, count):
        X.append([int(x) for x in IntToByte(i)])
        while len(X[i]) < 4:
            X[i].insert(0, 0)
        X[i].insert(0, 1)
    return X

import matplotlib.pyplot as plt

def drawGraph(E, k, name = 'E(k)'):
    plt.plot(k[1:], E[1:], marker = 'o')
    plt.xlabel('Era  k')
    plt.ylabel('Error  E')
    plt.axis([0, k[-1]+1, 0, max(E[1:])+1])
    plt.title('E(k)')
    plt.grid(True)
    plt.savefig('plt_{0}.png'.format(name))
    plt.clf()
