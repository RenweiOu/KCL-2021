#--
# solution-to-lab01.py
# Edited by Munkhtulga Battogtokh 11-Dec-2021
# Solutions for "Practical 1: Machine Learning Metrics"
# -----------------------------------------------------
# simple-nn-iris.py
# parsons/20-jan-2018
#
# Simple NN classifiers applied to the Iris dataset.
#
# Borrows code from:
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Manhattan distance
def L1_distance(first, second):
    return abs(first[0] - second[0]) + abs(first[1] - second[1])

# Load the data
iris = load_iris()
X = iris.data [:, [1, 3]] # 1 and 3 are the features we will use here.
y = iris.target

# Create a training and test set.
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)

# The NN classifer is just the training set so there is no learning to do.

#
# Classify based on the 1-NN rule and the training data.
#

y_pred = 0
score  = 0.0
preds1 = []
error1 = []

for i in range(len(X_test)):
    # Calculate the distance between the test element and every training element
    distances = []
    for j in range(len(X_train)):
        distances.append(L1_distance(X_test[i], X_train[j]))

    # Find the index of the closest element
    min_dist = 1000
    min_index = 0
    for j in range(len(X_train)):
        if distances[j] < min_dist:
            min_dist = distances[j]
            min_index = j

    # Classify on this basis
    y_pred = y_train[min_index]

    # Stash the predictions
    preds1.append(y_pred)

    # Is this the right value?
    if y_pred == y_test[i]:
        score += 1
        error1.append(1)
    else:
        error1.append(0)

    # Turn errors into plottable points
    misses1 = []
    missed1 = []
    for i in range(len(error1)):
        if error1[i] == 0:
            misses1.append(1)
            missed1.append(X_test[i])
            
print("My 1-NN classifer scores: ", score/len(y_test))

#
# Classify based on the 5-NN rule and the training data.
#

y_pred = 0
score  = 0.0
preds5 = []

for i in range(len(X_test)):
    # Calculate the distance between the test element and every training element
    distances = []
    for j in range(len(X_train)):
        distances.append(L1_distance(X_test[i], X_train[j]))

    # Find the index of the closest 5 elements
    min_dist = [1000, 1000, 1000, 1000, 1000]
    min_index = [0, 0, 0, 0, 0]
    for j in range(len(X_train)):
        if distances[j] < max(min_dist):
            min_dist[min_dist.index(max(min_dist))] = distances[j]
            min_index[min_dist.index(max(min_dist))] = j
    
    # Classify on this basis

    # Find the predictions of each of the 5 closest elements
    y_plist = [0, 0, 0, 0, 0]
    for j in range(len(y_plist)):
        y_plist[j] = y_train[min_index[j]]

    # Count up the votes for each class
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for j in range(len(y_plist)):
        if  y_plist[j] == 0:
            count_0 = count_0 + 1
        elif y_plist[j] == 1:
            count_1 = count_1 + 1
        else:
            count_2 = count_2 + 1

    # And predict
    if (count_0 >= count_1) and (count_0 >= count_2):
        y_pred = 0
    elif (count_1 >= count_0) and (count_1 >= count_2):
        y_pred = 1
    else:
        y_pred = 2 

    # Is this the right value?
    if y_pred == y_test[i]:
        score += 1

    # Stash the predictions
    preds5.append(y_pred)
    
print("My 5-NN classifer scores: ", score/len(y_test))

#
# Now plot the results
#
# Comparing the test data, the 1-NN and 5-NN results shows where the
# classifiers go wrong.

# Plot the original data on the same axes
plt.subplot( 3, 2, 1 )
plt.scatter(X[:, 0], X[:, 1], c=y.astype(float), cmap='autumn')
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
# Title
plt.title("Original dataset", fontsize=10)


# Plot the training data
plt.subplot( 3, 2, 3 )
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.astype(float), cmap='autumn')
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
# Title
plt.title("Training set", fontsize=10)

# Plot the test data
plt.subplot( 3, 2, 4 )
plt.xlim(1.5, 5.0)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.astype(float), cmap='autumn')
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
# Title
plt.title("Test set", fontsize=10)

# Plot the 1-NN predictions
plt.subplot( 3, 2, 5 )
plt.xlim(1.5, 5.0)
plt.scatter(X_test[:, 0], X_test[:, 1], c=preds1, cmap='autumn')
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
# Title
plt.title("1-NN predictions", fontsize=10)

# Plot the 5-NN predictions
plt.subplot( 3, 2, 6 )
plt.xlim(1.5, 5.0)
plt.scatter(X_test[:, 0], X_test[:, 1], c=preds5, cmap='autumn')
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
# Title
plt.title("5-NN predictions", fontsize=10)

plt.show()
