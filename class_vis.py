import warnings
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

warnings.filterwarnings('ignore')


def prettyPicture(clf, X_test, y_test):
    """

    :param clf: machine learning model
    :param X_test: Testing features
    :param y_test: True testing labels
    :return: Plotted image
    """

    x_min = 0.0
    x_max = 1.0
    y_min = 0.0
    y_max = 1.0

    # Plot the decision boundary. For that, we will assign a color to each point in the mesh
    h = .01  # Step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Also plot the test points
    grade_fast = [X_test[i][0] for i in range(len(X_test)) if y_test[i] == 0]
    bumpy_fast = [X_test[i][1] for i in range(len(X_test)) if y_test[i] == 0]
    grade_slow = [X_test[i][0] for i in range(len(X_test)) if y_test[i] == 1]
    bumpy_slow = [X_test[i][1] for i in range(len(X_test)) if y_test[i] == 1]

    plt.scatter(grade_fast, bumpy_fast, color='b', label='fast')
    plt.scatter(grade_slow, bumpy_slow, color='r', label='slow')
    plt.legend()
    plt.xlabel('bumpiness')
    plt.ylabel('grade')

    # The image will be saved in the algorithms folder
    plt.savefig("test.png")

