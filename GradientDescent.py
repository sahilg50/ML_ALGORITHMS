import numpy as np
import matplotlib.pyplot as plt

# Creating a function and plotting it.

function = lambda x: (x ** 3) - (3 * (x ** 2)) + 7

# Get 1000 evenly spaced numbers between -1 and 3.
x = np.linspace(-1, 3, 500)

# Plot the curve
plt.plot(x, function(x))
plt.show()


def derivation(x):
    """
    The function takes in a value of x and returns its derivation based on the initial funciton we specified.
    """

    x_deriv = 3 * (x ** 2) - (6 * x)
    return x_deriv


def step(x_new, x_prev, precision, l_r):
    """
    Description: This function takes in an initial or previous value for x, updates it based on steps taken via the learning and outputs the minimum value of x that reaches the precision satisfaction.

    Arguments:
        x_new  - a starting value of x that wil be updated

        x_prev - the previous value of x

        precision - a precision that determines the stop of the stepwise descent

        l_r - the learning rate (size of each descent step)

    Output:
    1. Prints out the latest new value of x which equates to the minimum we are looking for
    2. Prints out the the number of x values which equates to the number of gradient descent steps
    3. Plots a first graph of the function with the gradient descent path
    4. Plots a second graph of the function with a zoomed in gradient descent path in the important area
    """

    # create an empty list where the updated value of x and y will be appended during each iteration

    x_list, y_list = [x_new], [function(x_new)]

    while abs(x_prev- x_new) > precision:
        x_prev = x_new

        d_x = derivation(x_prev)

        x_new = x_prev - (l_r * d_x)

        x_list.append(x_new)

        y_list.append(function(x_new))

    print("Local minimum occurs at: " + str(x_new))
    print("Number of steps: " + str(len(x_list)))

    plt.subplot(1,2,2)
    plt.scatter(x_list,y_list,c="g")
    plt.plot(x,function(x),c="r")
    plt.title("Gradient descent")
    plt.show()

    plt.subplot(1, 2, 1)
    plt.scatter(x_list, y_list, c="g")
    plt.plot(x, function(x), c="r")
    plt.xlim([1.0, 2.1])
    plt.title("Zoomed in Gradient descent to Key Area")
    plt.show()

step(0.5, 0, 0.001, 0.05)