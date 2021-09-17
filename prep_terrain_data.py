import random


def makeTerrainData(n_points: int = 1000, test_size: float = 0.75):
    """
    :param n_points: Enter the total number of samples
    :param test_size: Enter the fraction of test size
    :return: X_train, y_train, X_test, y_test
    """
    # Make the toy dataset
    random.seed(42)

    # Features of the dataset
    grade = [random.random() for i in range(n_points)]
    bumpy = [random.random() for i in range(n_points)]

    # Error value to be added to the feature variables to vary the respective output
    error = [random.random() for i in range(n_points)]
    y = [round(grade[i] * bumpy[i] + 0.3 + 0.1 * error[i]) for i in range(n_points)]

    for i in range(len(y)):
        if grade[i] > 0.8 or bumpy[i] > 0.8:
            y[i] = 1

    # Splitting the data into Training/Testing sets
    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(test_size * n_points)
    X_train = X[0:split]
    X_test = X[split:]
    y_train = y[0:split]
    y_test = y[split:]

    return X_train, y_train, X_test, y_test


