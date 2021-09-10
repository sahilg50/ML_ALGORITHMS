"""
Decision Tree is a white box type of ML algorithm.
It shares the internal decision making logic.
The decision tree is a non=parametric method, which does not depend upon the probability distribution assumptions.
Decision trees can handle high dimensional data with good accuracy.
"""

import pandas as pd
import numpy as np
from collections import Counter


class Node:
    """
    Node for creating the node of the decision tree
    """

    def __init__(self, y: list,
                 x: pd.DataFrame,
                 min_samples_split=None,
                 max_depth=None,
                 depth=None,
                 node_type=None,
                 rule=None):
        # Saving the data to the node
        self.Y = y
        self.X = x

        # Saving the hyperParameters
        self.min_sample_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 5

        # Default current depth of the node
        self.depth = depth if depth else 0

        # Extracting all the features
        self.features = list(self.X.columns)

        # Type of the node
        self.node_type = node_type if node_type else "root"

        # Rule for splitting
        self.rule = rule if rule else ""

        # Calculating the counts of Y in the node
        self.counts = Counter(y)

        # Getting the Gini impurity based on Y distribution
        self.gini_impurity = self.get_gini_impurity()

        # Sorting the counts and saving the final prediction of the node
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        # Getting the last item
        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]

        # Saving the object attribute. This node will predict the class with the most frequent class.
        self.yhat = yhat

        # Saving the number of observations in the node
        self.n = len(y)

        # Initiating the left and the right node as empty
        self.left = None
        self.right = None

        # Default value for the split
        self.best_feature = None
        self.best_value = None

    @staticmethod
    def gini_impurity(y1_count: int, y2_count: int) -> float:
        """
        Given the observations a binary class calculate the GINI impurity
        """

        # Ensuring the correct types
        if y1_count is None:
            y1_count = 0

        if y2_count is None:
            y2_count = 0

        # Getting the total number of observations
        n = y1_count + y2_count

        # if n is 0 then we return the lowest possible gini impurity
        if n == 0:
            return o.o

        # Getting the probability of each class
        p1 = y1_count / n
        p2 = y2_count / n

        # Calculating GINI
        gini = 1 - (p1 ** 2 + p2 ** 2)

        # return the gini impurity
        return gini

    @staticmethod
    def ma(x1: np.array, window: int) -> np.array:
        """
        calculates the moving averages of the given list
        """
        return np.convolve(x1, np.ones(window), 'valid') / window

    def get_gini_impurity(self):
        """
        Function to calculate the GINI impurity of the node
        """
        # Getting the 0 and 1 counts
        y1_counts, y2_counts = self.counts.get(0, 0), self.counts.get(1, 0)

        # Getting the Gini Impurity
        return self.gini_impurity(y1_counts, y2_counts)

    def best_split(self) -> tuple:

        """
        Given X features and Y targets calculates the best split for a decision tree
        """
        # Creating a dataset for splitting
        df = self.X.copy()
        df['Y'] = self.Y

        # Getting the gini impurity for the base input
        gini_base = self.get_gini_impurity()

        # Finding which split yields the best GINI gain
        max_gain = 0

        # Default best features and split
        best_feature = None
        best_value = None

        for feature in self.features:
            # Dropping missing values
            Xdf = df.dropna().sort_values(feature)

            # Sorting the values and getting the rolling average
            xmeans = self.ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                # Splitting the dataset
                left_counts = Counter(Xdf[Xdf[feature] < value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature] >= value]['Y'])

                # Getting the Y distribution from the dicts
                y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(1, 0), right_counts.get(0,
                                                                                                                      0), right_counts.get(
                    1, 0)

                # Getting the left and right gini impurities
                gini_left = self.gini_impurity(y0_left, y1_left)
                gini_right = self.gini_impurity(y0_right, y1_right)

                # Getting the obs count from the left and the right data splits
                n_left = y0_left + y1_left
                n_right = y0_right + y1_right

                # Calculating the weights for each of the nodes
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                # Calculating the weighted GINI impurity
                wGINI = w_left * gini_left + w_right * gini_right

                # Calculating the GINI gain
                GINIgain = gini_base - wGINI

                # Checking if this is the best split so far
                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value

                    # Setting the best gain to the current one
                    max_gain = GINIgain

        return best_feature, best_value

    def grow_tree(self):
        """
        Recursive method to create the decision tree
        """
        # Making a df from the data
