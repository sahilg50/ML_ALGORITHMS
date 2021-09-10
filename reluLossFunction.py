# Relu stands for rectified linear unit

import matplotlib.pyplot as plt


# Defining the relu loss function
def relu(x):
    if x > 0:
        return x
    else:
        return 0


# Plotting the relu function on the graph

plt.style.use('ggplot')
plt.figure(figsize=(10, 5))

# define a series of inputs
input_series = [x for x in range(-19, 19)]

# calculate the outputs for our inputs
output_series = [relu(x) for x in input_series]

# line plot of raw inputs to rectified outputs
plt.plot(input_series, output_series)
plt.show()
