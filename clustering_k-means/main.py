from data_create import blobs, nested, nearby, nested_nearby
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import KMeans
import time


data = [blobs(5), nested(), nearby(), nested_nearby()]  # Массив тестов
for num, ds in enumerate(data):
    X = np.array(ds)
    X = X.T
    start = time.monotonic()
    scores = []
    for i in range(2, 11):                                # Смотрим при каком кол-ве кластеров будет наибольшее значение силуэта
        kmeans = KMeans(n_clusters=i).fit(X)
        y = kmeans.predict(X)
        score = silhouette_score(X, y)
        #print(i, score)
        scores.append(score)

    k = scores.index(max(scores)) + 2                     # Разделяем наши данные на кластеры

    kmeans = KMeans(n_clusters=k).fit(X)
    y = kmeans.predict(X)
    end = time.monotonic()
    print(max(scores), k, end - start)
    ax = plt.subplot(2, 2, num + 1)
    ax.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")

plt.show()