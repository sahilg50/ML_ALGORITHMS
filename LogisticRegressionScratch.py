"""
Notations —
n →number of features
m →number of training examples
X →input data matrix of shape (m x n)
y →true/ target value (can be 0 or 1 only)
x(i), y(i)→ith training example
w → weights (parameters) of shape (n x 1)
b →bias (parameter), a real number that can be broadcasted.
y_hat(y with a cap/hat)→ hypothesis (outputs values between 0 and 1)
"""

from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2)


# from sklearn.datasets import make_moons
# X, y = make_moons(n_samples=100, noise=0.24)

# Sigmoid Function
# This function will calculate y_hat
def sigmoid(z):
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat


def loss(y, y_hat):
    loss_value = -np.mean(y * (np.log(y_hat)) - (1 - y) * np.log(1 - y_hat))
    return loss_value


def gradient(X, y, y_hat):
    # X --> Input.
    # y --> true/target value.
    # y_hat --> hypothesis/predictions.
    # w --> weights (parameter).
    # b --> bias (parameter).

    # m-> number of training examples.
    m = X.shape[0]

    # Gradient of loss w.r.t weights.
    dw = (1 / m) * np.dot(X.T, (y_hat - y))

    # Gradient of loss w.r.t bias.
    db = (1 / m) * np.sum((y_hat - y))

    return dw, db


def normalize(X):
    # X -> input
    # m -> number of training samples
    # n -> number of features

    m, n = X.shape
    # Normalize=ing all the n features of X
    for i in range(n):
        X[:i] = (X[:i] - np.mean(X[:i])) / np.std(X[:i])

    return X


def train(X, y, bs: int, epochs: int, lr):
    # X --> Input.
    # y --> true/target value.
    # bs --> Batch Size.
    # epochs --> Number of iterations.
    # lr --> Learning rate.

    # m-> number of training examples
    # n-> number of features

    m, n = X.shape

    # Initializing weights and bias to zeros
    w = np.zeros((n, 1))
    b = 0

    # Reshaping y
    y = y.reshape(m, 1)

    # Normalizing the inputs
    X = normalize(X)

    # Empty list to store losses.
    losses = []

    # Training Loop
    for epoch in range(epochs):
        for i in range((m - 1) // bs + 1):
            # Defining batches. SGD
            start_i = i * bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]

            # Calculating hypothesis/prediction
            y_hat = sigmoid(np.dot(xb, w) + b)

            # Getting the gradients from loss w.r.t to w,b
            dw, db = gradient(xb, yb, y_hat)

            # Updating the parameters
            w -= lr * dw
            b -= lr * db

        # Calculating the loss and appending it in the list.
        loss_per_epoch = loss(y, sigmoid(np.dot(X, w) + b))
        losses.append(loss_per_epoch)
        print("<--Epoch: {epoch}, Loss: {loss_per_epoch}-------->".format(epoch=epoch, loss_per_epoch=loss_per_epoch))

    # Returning weights, bias and losses(List)
    return w, b, losses


# Predict function
def predict(X):
    # X --> Input.

    # Normalizing the inputs.
    x = normalize(X)

    # Calculating presictions/y_hat.
    preds = sigmoid(np.dot(X, w) + b)

    # Empty List to store predictions.
    pred_class = []
    # if y_hat >= 0.5 --> round up to 1
    # if y_hat < 0.5 --> round up to 1
    pred_class = [1 if i > 0.5 else 0 for i in preds]

    return np.array(pred_class)


def accuracy(y, y_hat):
    y_hat = y_hat.reshape(y.shape[0])
    accuracy = np.sum(np.abs(y - y_hat)) / len(y)
    return accuracy


# Training the model and find the right attributes
w, b, l = train(X, y, bs=100, epochs=1000, lr=0.01)

# Potting Decision Boundary
acc = accuracy(y, y_hat=predict(X))
print(acc)
