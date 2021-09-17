"""
k-Nearest Neighbors (KNN) is a supervised machine learning algorithm that can be used for either regression or classification tasks. KNN is non-parametric, which means that the algorithm does not make assumptions about the underlying distributions of the data. This is in contrast to a technique like linear regression, which is parametric, and requires us to find a function that describes the relationship between dependent and independent variables.
"""

# Initial imports
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load iris data and store in dataframe
from sklearn import datasets

iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.head()


# Separate X and y data
X = df.drop('target', axis=1)
y = df.target


# calculate the euclidean distance
def euclidean_distance(a, b):

    dim = len(a)
    distance = 0

    for d in range(dim):
        distance += abs(a[d] - b[d])**2

    distance = distance**(1/2)

    return distance


# Define an arbitrary test point
test_pt = [4.8, 2.7, 2.5, 0.7]

# Calculate distance between test_pt and all points in X
distances = []

for i in X.index:

    distances.append(euclidean_distance(test_pt, X.iloc[i]))

df_dists = pd.DataFrame(data=distances, index=X.index, columns=['dist'])
print(df_dists.head())


# Find the 5 nearest neighbors
df_nn = df_dists.sort_values(by=['dist'], axis=0)[:5]
df_nn


# Create counter object to track the labels
counter = Counter(y[df_nn.index])

# Get most common label of all the nearest neighbors
counter.most_common()[0][0]
