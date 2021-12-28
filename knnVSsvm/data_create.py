import numpy
from sklearn.datasets import make_moons, make_circles, make_classification, make_gaussian_quantiles, make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def gaussian(count):
    X1, Y1 = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=count, random_state=1)
    return X1, Y1

def blobs(count):
    X1, Y1 = make_blobs(n_samples=500, n_features=2, centers=count, random_state=1)
    return X1, Y1

def moons():
    X1, Y1 = make_moons(n_samples=500, noise=0.3, random_state=1)
    return X1, Y1

def circles():
    X1, Y1 = make_circles(n_samples=500, noise=0.2, factor=0.5, random_state=1)
    return X1, Y1

def usual():
    X1, Y1 = make_classification(
        n_samples=500, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )
    rng = np.random.RandomState(2)
    X1 += 2 * rng.uniform(size=X1.shape)
    return X1, Y1

if __name__ == '__main__':
    figure = plt.figure(figsize=(10, 10))
    '''plt.subplot(323)
    X2, Y2 = make_classification(n_features=2, n_redundant=0, n_informative=2)
    plt.scatter(X2[:, 0], X2[:, 1], marker="o", c=Y2, s=25, edgecolor="k")

    plt.subplot(324)
    X1, Y1 = make_classification(
        n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1
    )
    plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")

    plt.subplot(325)
    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    X = StandardScaler().fit_transform(X)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")

    gaussian(5)
    blobs(5)'''
    x1, y1 = circles()
    plt.scatter(x1[:, 0], x1[:, 1], marker="o", c=y1, s=25, edgecolor="k")
    plt.show()