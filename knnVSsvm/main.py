import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from data_create import usual, gaussian, blobs, circles, moons
import time
from sklearn.metrics import f1_score


h = 0.02

names = [
    "KNN",
    "SVM"
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear")
]

datasets = [
    usual(), blobs(2), gaussian(2), moons(), circles()
]

figure = plt.figure(figsize=(45, 27))
i = 1


for ds_cnt, ds in enumerate(datasets):
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,  alpha=0.6, edgecolors="k")

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    i += 1

    for name, clf in zip(names, classifiers):
        start_time = time.monotonic()
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            y = clf.predict(X_test)
            score_f1 = f1_score(y_test, y)
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            y = clf.predict(X_test)
            score_f1 = f1_score(y_test, y)

        end_time = time.monotonic()
        print(name, ds_cnt, end_time - start_time, score, score_f1)


        Z = Z.reshape(xx.shape)
        #print(Z, xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.8)

        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,  edgecolors="k")
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors="k", alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1


plt.tight_layout()
plt.show()