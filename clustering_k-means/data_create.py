import numpy
from sklearn.datasets import  make_blobs
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import math
from sklearn.preprocessing import StandardScaler



def blobs(count):
    X1, Y1 = make_blobs(n_samples=300, n_features=2, centers=count, cluster_std=0.6, random_state=145)
    return X1[:, 0], X1[:, 1]

def make_circles(count, x, y, r1, w):          # Создаем кольцо с центорм в х, у, внутренним радиусом r и толщиной w.
    X, Y = [], []                              # Состоящее из count точек
    for i in range(count):
        a = rd.uniform(0, 360)
        r = rd.uniform(r1, r1 + w)
        x1 = r * math.cos(math.pi * a / 180)
        y1 = r * math.sin(math.pi * a / 180)
        X.append(x + x1)
        Y.append(y + y1)
    return X, Y

def nested():            # 2 кольца друг в друге
    COUNT = 300
    x1, y1 = make_circles(COUNT, 50, 50, 40, 10)
    x2, y2 = make_circles(COUNT, 50, 50, 20, 10)
    X, Y = x1 + x2, y1 + y2
    return X, Y

def nearby():            # 2 кольца рядом
    COUNT = 300
    x1, y1 = make_circles(COUNT, 50, 50, 40, 10)
    x2, y2 = make_circles(COUNT, 160, 50, 40, 10)
    return x1 + x2, y1 + y2

def nested_nearby():     # nested + nearby
    COUNT = 300
    x1, y1 = make_circles(COUNT, 50, 50, 40, 10)
    x2, y2 = make_circles(COUNT, 50, 50, 20, 10)
    x3, y3 = make_circles(COUNT, 140, 50, 40, 10)
    x4, y4 = make_circles(COUNT, 140, 50, 20, 10)
    return x1 + x2 + x3 + x4, y1 + y2 + y3 + y4



if __name__ == '__main__':
    figure = plt.figure()

    '''x, y = make_circles(500, 50, 50, 40, 10)
    plt.scatter(x, y, marker="o", s=25, edgecolor="k")
    x, y = make_circles(500, 50, 50, 20, 10)
    plt.scatter(x, y, marker="o", s=25, edgecolor="k")'''
    x, y = nested_nearby()
    plt.scatter(x, y, marker="o", s=25, edgecolor="k")
    '''x, y = make_circles(500, 150, 50, 30, 10)
    plt.scatter(x, y, marker="o", s=25, edgecolor="k")'''
    plt.show()